import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ(nn.Module):
    def __init__(self, args, hidden_size, codebook_len: int):
        super().__init__()
        self.K = codebook_len
        self.decay = args.ema_decay
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(args.codebook_dropout)

        self.codebook_classifier = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.codebook = nn.Parameter(torch.empty(self.K, hidden_size), requires_grad=False)

        nn.init.uniform_(self.codebook_classifier.weight, -1.0 / self.K, 1.0 / self.K)
        nn.init.uniform_(self.codebook.data, -1.0 / self.K, 1.0 / self.K)

        self.register_buffer('cluster_size_sum', torch.full([self.K], self.batch_size / self.K), persistent=False)
        self.register_buffer('embedding_sum', self.batch_size / self.K * self.codebook.data.clone(), persistent=False)

    def _ema_inplace(self, moving_avg, new, decay):
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def _laplace_smoothing(self, x, n_categories, eps=1e-5):
        return (x + eps) / (x.sum() + n_categories * eps) * x.sum()

    def _distance(self, x, codebook):
        return x.pow(2).sum(1, keepdim=True) - 2 * torch.einsum("bd,kd->bk", x, codebook) + codebook.pow(2).sum(1)

    def forward(self, x, freq_weights=None):
        z_e = self.codebook_classifier(x)
        
        with torch.no_grad():
            if self.training and (self.cluster_size_sum < 1e-3).any():
                i = torch.nonzero(self.cluster_size_sum < 1e-3)[0]

                coverage, _ = self._distance(z_e, self.codebook).min(-1)
                coverage -= coverage.min()
                coverage /= coverage.max()
                p = torch.rand(x.size(0), device=x.device) * coverage
                p = torch.softmax(x.size(0) * p / p.max(), dim=0)  # peaky distribution

                random_combination = torch.einsum("kd,k->d", z_e, p)
                self.codebook.data[i, :] = random_combination
                alpha = self.batch_size / (self.batch_size + 1 - self.cluster_size_sum[i])
                self.embedding_sum *= alpha
                self.cluster_size_sum *= alpha
                self.embedding_sum[i, :] = random_combination
                self.cluster_size_sum[i] = 1

            distance = self._distance(z_e, self.codebook)
            indices = distance.argmin(1)
            z_q = F.embedding(indices, self.codebook)

            if self.training:
                assert freq_weights is not None
                indices_onehot = F.one_hot(indices, self.K).float()
                cluster_size = (indices_onehot * freq_weights.unsqueeze(1)).sum(0)
                embedding_sum = torch.einsum("b,bk,bd->kd", freq_weights, indices_onehot, z_e)

                self._ema_inplace(self.cluster_size_sum, cluster_size, self.decay)
                self._ema_inplace(self.embedding_sum, embedding_sum, self.decay)

                cluster_size = self._laplace_smoothing(self.cluster_size_sum, self.K)
                embedding_normalized = self.embedding_sum / cluster_size.unsqueeze(1)
                self.codebook.data = embedding_normalized

        vq_loss = F.mse_loss(z_q.detach(), z_e, reduction="none").mean(-1)
        z_q = z_e + (z_q - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = self.dropout(z_q)

        return z_q, indices, vq_loss

    def decode(self, indices):
        z_q = F.embedding(indices, self.codebook)
        return z_q

    def distance_to_index(self, x, indices):
        z_e = self.codebook_classifier(x)
        distance = self._distance(z_e, self.codebook)  # shape: [B, V]
        probs = F.log_softmax(-distance, dim=-1)
        probs = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return probs

    def get_codebook_perplexity(self):
        p = self.cluster_size_sum / self.cluster_size_sum.sum()
        p = p * torch.log(p + 1e-12)
        return torch.exp(-p.sum()).item()

    def get_codebook_count(self):
        return (self.cluster_size_sum > 1e-3).sum().item() / self.K * 100.0

    def generate_codebook(self, k, d, lr=256.0, steps=10000):
        print("GENERATING CODEBOOK")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        points = torch.randn(k, d, requires_grad=True, device=device)
        points.data = F.normalize(points.data, dim=-1)

        optimizer = torch.optim.SGD([points], lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

        eye_mask = torch.eye(k, dtype=torch.bool, device=device)
        for i in range(steps + 1):
            points.grad = None
            distance = 2.0 - 2.0 * torch.einsum("ad,bd->ab", points, points)
            distance = distance.masked_fill(eye_mask, float("inf"))
            distance, _ = distance.topk(8, dim=-1, sorted=False, largest=False)
            loss = 1.0 / (distance + 1e-3)
            loss.mean().backward()

            optimizer.step()
            scheduler.step()
            points.data = F.normalize(points.data, dim=-1)

            if i % (steps // 10) == 0:
                print(f"{i}\t{loss.mean().item()}")
        print(flush=True)

        return points.data
