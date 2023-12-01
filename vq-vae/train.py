import tqdm
import argparse
import wandb
import os
import random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model
from ema_model import EMA
from sam import SAM
from autoclip import AutoClip
from lazy_adam import LazyAdamW
from random_sampler import WeightedRandomSampler


torch.backends.cuda.matmul.allow_tf32 = True


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

def run(model, dataset, word: str, bow=True, eow=True):
    input = torch.tensor(dataset.numericalize(dataset.encode(word)[0], bow=bow, eow=eow)).unsqueeze(0).cuda()
    prediction, indices, perplexity = model.inference(input)

    gold = dataset.ids_to_word(input[0, 1:-1].tolist())
    prediction = dataset.ids_to_word(prediction[0, :].tolist())

    print(f"{gold}\t-> {indices[0][0].item()}, {indices[1][0].item()}, {indices[2][0].item()} -> {prediction}, {(-perplexity).exp().item()*100.0:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook_groups", type=str, default="256,256,256")
    parser.add_argument("--language", type=str, default="czech")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--vq_loss_weight", type=float, default=0.5)
    parser.add_argument("--orthogonal_loss_weight", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--encoder_dropout", type=float, default=0.3)
    parser.add_argument("--codebook_dropout", type=float, default=0.3)
    parser.add_argument("--decoder_dropout", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--steps_per_epoch", type=int, default=2500)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2**12)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    args = parser.parse_args()

    seed_everything(args.seed)
    wandb.init(name=f"{args.language}_regularized_{args.hidden_size}_{args.seed}", config=args, project="vae-tokenizer")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = Dataset(args, path=f"data/{args.language}_train_word_freq.tsv", vocab=None, random=True, split=True)
    valid_dataset = Dataset(args, path=f"data/{args.language}_valid_word_freq.tsv", vocab=train_dataset.vocab, split=False, remove_long_words=True)
    frequent_dataset = Dataset(args, path=f"data/{args.language}_frequent_word_freq.tsv", vocab=train_dataset.vocab, split=False, remove_long_words=True)

    train_sampler = WeightedRandomSampler(
        [freq / math.log(freq + 1) for freq in train_dataset.freqs],
        num_samples=args.steps_per_epoch * args.batch_size, replacement=True
    )
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=4, collate_fn=train_dataset.get_collate_fn(args.max_length), drop_last=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, num_workers=2, collate_fn=valid_dataset.get_collate_fn(args.max_length), drop_last=False)
    frequent_loader = DataLoader(frequent_dataset, args.batch_size, num_workers=2, collate_fn=frequent_dataset.get_collate_fn(args.max_length), drop_last=False)

    model = Model(args, train_dataset).to(device)
    ema_model = EMA(Model(args, train_dataset).to(device), model, beta=0.999, update_after_step=0, update_every=1)
    for vq in model.vqs:
        vq.init_steps = 500

    indices_count = torch.zeros(256*256*256, dtype=torch.float, device=device)

    parameters = list(model.named_parameters())
    no_decay = ['bias', "layer_norm", "embedding"]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    print("No decay params:")
    for n, _ in parameters:
        if any(nd in n for nd in no_decay):
            print(n)
    print(flush=True)

    optimizer = SAM(optimizer_grouped_parameters, LazyAdamW, rho=args.rho, alpha=args.alpha, adaptive=True, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    # optimizer = LazyAdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    grad_clip = AutoClip(model.parameters(), initial_clipping=0.2, history_len=100)

    def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_schedule_with_warmup(optimizer, 500, args.num_steps, 0.1)
    del parameters

    # train loop
    for epoch in range(args.num_steps // args.steps_per_epoch):

        model.train()
        train_iter = tqdm.tqdm(train_loader)
        for step, (words, freqs) in enumerate(train_iter):
            words, freqs = words.to(device), freqs.to(device)
            scheduler.step()
            freq_weights = torch.log(freqs + 1)

            model.eval()
            reconstruction_loss, vq_loss, indices = model(words, freq_weights)
            loss = (reconstruction_loss + args.vq_loss_weight * vq_loss) * freq_weights
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            indices = indices[0] * (256*256) + indices[1] * 256 + indices[2]
            indices = indices.flatten().bincount(minlength=256*256*256)
            indices_count = 0.9999 * indices_count + (1.0 - 0.9999) * indices

            model.train()
            reconstruction_loss_, vq_loss_, _ = model(words, freq_weights)
            loss = (reconstruction_loss_ + args.vq_loss_weight * vq_loss_) * freq_weights
            loss.mean().backward()
            grad_norm, clip_value = grad_clip()
            optimizer.second_step(zero_grad=True)

            ema_model.update()

            wandb.log(
                {
                    "epoch": epoch,
                    "train/reconstruction_loss": reconstruction_loss.mean().item(),
                    "train/vq_loss": vq_loss.mean().item(),
                    "stats/grad_norm": grad_norm.item(),
                    "stats/clip_value": clip_value.item(),
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    **{
                        f"stats/codebook_{i}_perplexity": vq.get_codebook_perplexity()
                        for i, vq in enumerate(model.vqs)
                    },
                    **{
                        f"stats/codebook_{i}_count": vq.get_codebook_count()
                        for i, vq in enumerate(model.vqs)
                    }
                }
            )

        ema_model.eval()
        with torch.no_grad():
            sum_loss, last_sum_correct, sum_correct, last_n_words, n_words = 0, 0, 0, 0, 0
            step_accuracies, next_step = {}, 1
            for n_steps, (words, _) in enumerate(valid_loader):
                words = words.to(device)
                sum_loss += ema_model(words)[0].sum().item()

                prediction, _, _ = ema_model.inference(words)
                words = words[:, 1:]

                if prediction.size(1) > words.size(1):
                    prediction = prediction[:, :words.size(1)]
                elif prediction.size(1) < words.size(1):
                    prediction = F.pad(prediction, (0, words.size(1) - prediction.size(1)), value=train_dataset.pad_id)
                sum_correct += (prediction == words).all(1).sum().item()
                last_sum_correct += (prediction == words).all(1).sum().item()

                n_words += words.size(0)
                last_n_words += words.size(0)

                if next_step == n_steps:
                    step_accuracies[f"valid/accuracy_{n_steps}"] = last_sum_correct / last_n_words
                    last_sum_correct, last_n_words = 0, 0
                    next_step *= 2

            frequent_correct, frequent_n_words = 0, 0
            for words, _ in tqdm.tqdm(frequent_loader):
                words = words.to(device)
                prediction, _, _ = ema_model.inference(words)
                words = words[:, 1:]

                if prediction.size(1) > words.size(1):
                    prediction = prediction[:, :words.size(1)]
                elif prediction.size(1) < words.size(1):
                    prediction = F.pad(prediction, (0, words.size(1) - prediction.size(1)), value=train_dataset.pad_id)
                frequent_correct += (prediction == words).all(1).sum().item()
                frequent_n_words += words.size(0)

            wandb.log(
                {
                    "epoch": epoch,
                    "valid/reconstruction_loss": sum_loss / n_words,
                    "valid/accuracy": sum_correct / n_words,
                    "valid/frequent_accuracy": frequent_correct / frequent_n_words,
                }
            )

            # print some examples for sanity checking

            run(model, valid_dataset, "blackberry")
            run(model, valid_dataset, "black", eow=False)
            run(model, valid_dataset, "blackb", eow=False)
            run(model, valid_dataset, "blacko", eow=False)
            run(model, valid_dataset, "berry", bow=False)
            run(model, valid_dataset, "erry", bow=False)
            run(model, valid_dataset, "berries", bow=False)
            run(model, valid_dataset, "erries", bow=False)
            run(model, valid_dataset, "super", eow=False)
            run(model, valid_dataset, "superc", eow=False)
            print()

        torch.save(
            {
                "model": ema_model.online_model.state_dict(),
                "index_count": indices_count,
                "vocabulary": train_dataset.vocab,
                "args": args
            },
            f"{args.language}_{args.hidden_size}_{args.seed}.bin"
        )
