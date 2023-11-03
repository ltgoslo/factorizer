import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder
from vq import VQ
from apex_wrapper import LayerNorm


class Model(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.pad_id = dataset.pad_id
        self.bos_id = dataset.bos_id
        self.eos_id = dataset.eos_id
        self.skipped_indices = dataset.skipped_indices
        self.max_length = args.max_length
        self.vocab_size = len(dataset.vocab)
        self.vocab_size_remainder = 0
        if self.vocab_size % 8 != 0:
            self.vocab_size_remainder = self.vocab_size - (self.vocab_size % 8)
            self.vocab_size += self.vocab_size_remainder

        self.embedding = nn.Embedding(len(dataset.vocab), args.hidden_size, padding_idx=self.pad_id, sparse=True)
        self.encoder_forward_pos_embedding = nn.Embedding(args.max_length, args.hidden_size, sparse=True)
        self.encoder_backward_pos_embedding = nn.Embedding(args.max_length, args.hidden_size, sparse=True)
        self.decoder_pos_embedding = nn.Embedding(args.max_length + 3 - 1, args.hidden_size, sparse=True)
        self.embedding_layer_norm = LayerNorm(args.hidden_size, elementwise_affine=False)
        self.encoder_dropout = nn.Dropout(args.encoder_dropout)

        self.encoder = Encoder(args.num_layers, args.hidden_size, args.num_heads, args.encoder_dropout, args.attention_dropout)
        self.vqs = nn.ModuleList(
            VQ(args, args.hidden_size, int(codebook_size))
            for codebook_size in args.codebook_groups.split(',')
        )
        self.decoder = Decoder(args.num_layers, args.hidden_size, args.num_heads, args.decoder_dropout, args.attention_dropout)
        self.output = nn.Linear(args.hidden_size, len(dataset.vocab))

        self.embedding.weight.data /= math.sqrt(args.hidden_size)
        self.encoder_forward_pos_embedding.weight.data /= math.sqrt(args.hidden_size)
        self.encoder_backward_pos_embedding.weight.data /= math.sqrt(args.hidden_size)
        self.decoder_pos_embedding.weight.data /= math.sqrt(args.hidden_size)
        self.output.weight.data = self.embedding.weight.data.clone()  # do not tie! just a shallow copy
        self.output.bias.data.zero_()

    def forward(self, words, freq_weights=None):
        mask = words == self.pad_id
        words = words.t()

        context, indices, vq_loss, embedding = self.encode(words, mask, freq_weights)

        encoding = self.decoder(embedding, context)
        output = self.output(encoding)

        mask = mask[:, 1:].t()
        reconstruction_loss = F.cross_entropy(output.flatten(0, 1), words[1:, :].flatten(), ignore_index=self.pad_id, reduction='none').view(-1, words.size(1))
        reconstruction_loss = reconstruction_loss.masked_fill(mask, 0.0).sum(0) / (~mask).sum(0)

        return reconstruction_loss, vq_loss, indices

    @torch.no_grad()
    def inference(self, words):
        mask = words == self.pad_id
        words = words.t()

        context, indices, _, embedding = self.encode(words, mask)
        decoded = self.greedy_search(context)

        encoding = self.decoder(embedding, context)
        output = self.output(encoding)

        mask = mask[:, 1:].t()
        perplexity = F.cross_entropy(output.flatten(0, 1), torch.clamp(words[1:, :].flatten(), min=0), ignore_index=self.pad_id, reduction='none').view(-1, words.size(1))
        perplexity = perplexity.masked_fill(mask, 0.0).sum(0).mean()

        return decoded, indices, perplexity

    def greedy_search(self, context):
        decoder_input = self.embedding_layer_norm(
            self.embedding(torch.full([1, context.size(1)], self.bos_id, device=context.device)) \
                + self.decoder_pos_embedding.weight[3, :]
        )

        stop = torch.zeros(decoder_input.size(1), dtype=torch.bool, device=decoder_input.device)
        decoded = torch.zeros(decoder_input.size(1), 0, dtype=torch.long, device=decoder_input.device)

        for i in range(self.max_length - 1):
            prediction = self.decoder(decoder_input, context)[-1, :, :]
            prediction = self.output(prediction)
            prediction = torch.where(stop, self.pad_id, prediction.argmax(-1))
            stop |= prediction == self.eos_id

            if stop.all() or i == self.max_length-2:
                break

            decoded = torch.cat([decoded, prediction.unsqueeze(1)], dim=1)
            decoder_input = torch.cat([
                decoder_input,
                self.embedding_layer_norm(
                    self.embedding(prediction) + self.decoder_pos_embedding.weight[3 + i + 1, :]
                ).unsqueeze(0)
            ], dim=0)

        return decoded

    def beam_search(self, context, beam_size=4):
        batch_size = context.size(1)
        vocab_size = self.output.weight.size(0)

        decoder_input = self.embedding_layer_norm(
            self.embedding(torch.full([1, batch_size], self.bos_id, device=context.device)) \
                + self.decoder_pos_embedding.weight[3, :]
        )

        candidates = [[] for _ in range(batch_size)]

        prediction = self.decoder(decoder_input, context)[-1, :, :]
        prediction = self.output(prediction)
        prediction = F.log_softmax(prediction, dim=-1)
        prediction = torch.topk(prediction, beam_size, dim=-1)  # shape: [batch, beam]

        context = context.repeat_interleave(beam_size, dim=1)
        decoder_input = torch.cat([
            decoder_input.repeat_interleave(beam_size, dim=1),
            self.embedding_layer_norm(
                self.embedding(prediction.indices.flatten()) + self.decoder_pos_embedding.weight[3 + 1, :]
            ).unsqueeze(0)
        ], dim=0)
        target = prediction.indices.flatten().unsqueeze(1)
        logp = prediction.values  # shape: [batch, beam]

        for i in range(1, self.max_length - 1):
            prediction = self.decoder(decoder_input, context)[-1, :, :]
            prediction = self.output(prediction)
            prediction = prediction.view(batch_size, beam_size, -1)
            prediction = F.log_softmax(prediction, dim=-1)  # shape: [batch, beam, V]
            prediction = logp.unsqueeze(-1) + prediction  # shape: [batch, beam, V]
            prediction = prediction.flatten(1, 2)  # shape: [batch, beam x V]
            prediction = torch.topk(prediction, 16*beam_size, dim=1, sorted=True)  # shape: [batch, beam]

            target = target.cpu()
            next_char = (prediction.indices % vocab_size).tolist()  # shape: [batch, beam]
            previous_beam = (prediction.indices // vocab_size).tolist()  # shape: [batch, beam]
            logp = prediction.values.tolist()  # shape: [batch, beam]
            next_target, next_logps = [], []

            for batch in range(batch_size):
                for ch, beam, score in zip(next_char[batch], previous_beam[batch], logp[batch]):
                    if ch == self.eos_id:
                        if len(candidates[batch]) < beam_size:
                            candidates[batch].append((target[batch * beam_size + beam, :], score / i))
                    else:
                        next_target.append(torch.cat([
                            target[batch * beam_size + beam, :],
                            torch.tensor([ch])
                        ], dim=0))
                        next_logps.append(score)

                        if len(next_target) % beam_size == 0:
                            break

            if all(len(candidate) >= beam_size for candidate in candidates) or (i == self.max_length - 2):
                break

            target = torch.stack(next_target, dim=0).to(context.device)  # shape: [batch x beam, length]
            logp = torch.tensor(next_logps, device=context.device).view(batch_size, -1)  # shape: [batch, beam]

            decoder_input = torch.cat([
                decoder_input,
                self.embedding_layer_norm(
                    self.embedding(target[:, -1]) + self.decoder_pos_embedding.weight[3 + i + 1, :]
                ).unsqueeze(0)
            ], dim=0)

        best_targets = []
        for batch in range(batch_size):
            if len(candidates[batch]) == 0:
                best_targets.append(next_target[batch * beam_size].tolist())
            else:
                best_targets.append(sorted(candidates[batch], key=lambda x: x[1], reverse=True)[0][0].tolist())

        return best_targets

    def encode(self, words, mask, freq_weights=None):
        pos_embedding = self.encoder_dropout(
            F.embedding(
                torch.arange(words.size(0), device=words.device),
                self.encoder_forward_pos_embedding.weight,
                sparse=self.training
            )
        )
        pos_embedding = pos_embedding.unsqueeze(1) + self.encoder_dropout(
            F.embedding(
                torch.clamp(-torch.arange(words.size(0), device=words.device).unsqueeze(1).expand_as(words) + (~mask).long().sum(1).unsqueeze(0) - 1, min=0),
                self.encoder_backward_pos_embedding.weight,
                sparse=self.training
            )
        )
        pos_embedding = pos_embedding.masked_fill(mask.t().unsqueeze(-1), 0.0)
        pos_embedding = F.pad(pos_embedding, (0, 0, 0, 0, 3, 0), value=0.0)

        codebook_ids = torch.arange(3, device=words.device).unsqueeze(1).expand(-1, words.size(1))
        word_embedding = self.encoder_dropout(
            F.embedding(
                torch.cat([codebook_ids, words], dim=0),
                self.embedding.weight,
                padding_idx=self.embedding.padding_idx,
                sparse=self.training
            )
        )
        embedding = word_embedding + pos_embedding
        embedding = self.embedding_layer_norm(embedding)

        encoding = self.encoder(embedding, F.pad(mask, (3, 0), value=False))
        contexts, indices, vq_losses = zip(*[vq(encoding[i, :, :], freq_weights) for i, vq in enumerate(self.vqs)])

        vq_loss = sum(vq_losses) / len(vq_losses)

        word_embedding = self.encoder_dropout(
            F.embedding(
                words[:-1, :],
                self.embedding.weight,
                padding_idx=self.embedding.padding_idx,
                sparse=self.training
            )
        )
        pos_embedding = self.encoder_dropout(
            F.embedding(
                torch.arange(words.size(0) + 3 - 1, device=words.device),
                self.decoder_pos_embedding.weight,
                sparse=self.training
            )
        )
        context = torch.stack(contexts, dim=0) + pos_embedding[:3, :].unsqueeze(1)
        context = self.embedding_layer_norm(context)

        embedding = word_embedding + pos_embedding[3:, :].unsqueeze(1)
        embedding = self.embedding_layer_norm(embedding)

        return context, indices, vq_loss, embedding

    @torch.no_grad()
    def decode(self, indices):
        batch_size = indices.size(0)
        context = [vq.decode(indices[:, i]) for i, vq in enumerate(self.vqs)]
        context = torch.stack(context, dim=0)
        context += self.decoder_pos_embedding.weight[:3, :].unsqueeze(1)
        context = self.embedding_layer_norm(context)

        decoder_input = self.embedding_layer_norm(
            self.embedding(torch.full([1, batch_size], self.bos_id, device=context.device)) \
                + self.decoder_pos_embedding.weight[3, :]
        )

        stop = torch.zeros(decoder_input.size(1), dtype=torch.bool, device=decoder_input.device)
        decoded = [[] for _ in range(decoder_input.size(1))]
        perplexity = [0.0 for _ in range(decoder_input.size(1))]

        for i in range(self.max_length - 1):
            encoding = self.decoder(decoder_input, context)[-1, :, :]
            output = F.log_softmax(self.output(encoding), dim=-1)
            prediction = torch.where(stop, self.pad_id, output.argmax(-1))
            stop |= prediction == self.eos_id

            if stop.all():
                break

            for b, (p, s) in enumerate(zip(prediction.tolist(), stop.tolist())):
                if not s:
                    decoded[b].append(p)
                    perplexity[b] += output[b, :].max().item()

            if i == self.max_length - 2:
                break

            decoder_input = torch.cat([
                decoder_input,
                self.embedding_layer_norm(
                    self.embedding(prediction) + self.decoder_pos_embedding.weight[3 + i + 1, :]
                ).unsqueeze(0)
            ], dim=0)

        return decoded, perplexity

    @torch.no_grad()
    def distance_to_gold(self, words, mask, indices):
        pos_embedding = self.encoder_dropout(
            F.embedding(
                torch.arange(words.size(0), device=words.device),
                self.encoder_forward_pos_embedding.weight,
                sparse=self.training
            )
        )
        pos_embedding = pos_embedding.unsqueeze(1) + self.encoder_dropout(
            F.embedding(
                torch.clamp(-torch.arange(words.size(0), device=words.device).unsqueeze(1).expand_as(words) + (~mask).long().sum(1).unsqueeze(0) - 1, min=0),
                self.encoder_backward_pos_embedding.weight,
                sparse=self.training
            )
        )
        pos_embedding = pos_embedding.masked_fill(mask.t().unsqueeze(-1), 0.0)
        pos_embedding = F.pad(pos_embedding, (0, 0, 0, 0, 3, 0), value=0.0)

        codebook_ids = torch.arange(3, device=words.device).unsqueeze(1).expand(-1, words.size(1))
        word_embedding = self.encoder_dropout(
            F.embedding(
                torch.cat([codebook_ids, words], dim=0),
                self.embedding.weight,
                padding_idx=self.embedding.padding_idx,
                sparse=self.training
            )
        )
        embedding = word_embedding + pos_embedding
        embedding = self.embedding_layer_norm(embedding)

        encoding = self.encoder(embedding, F.pad(mask, (3, 0), value=False))
        distances = sum([
            vq.distance_to_index(encoding[i, :, :], indices[:, i])
            for i, vq in enumerate(self.vqs)
        ]) / len(self.vqs)

        return distances
