mport torch
import torch.nn as nn


class AutoClip:
    def __init__(self, parameters, initial_clipping=1.0, percentile=75, history_len=1000):
        self.parameters = list(parameters)
        self.grad_history = [torch.full([history_len], initial_clipping) for _ in self.parameters]

        self.index = 0
        self.history_len = history_len
        self.percentile = percentile

    @torch.no_grad()
    def __call__(self):
        grad_norms, clip_values = [], []
        for i, param in enumerate(self.parameters):
            if param.grad is None or param.grad.is_sparse or not param.grad.abs().sum().is_nonzero():
                continue

            self.grad_history[i][self.index] = param.grad.data.norm(2)

            clip_value = self._get_percentile(self.grad_history[i], self.percentile)
            grad_norms.append(nn.utils.clip_grad_norm_(param, clip_value.item()))
            clip_values.append(clip_value)

        self.index = (self.index + 1) % self.history_len
        grad_norm = torch.stack(grad_norms).norm(2)
        clip_value = torch.stack(clip_values).norm(2)

        return grad_norm, clip_value

    def _get_percentile(self, tensor, percentile):
        k = 1 + round(0.01 * percentile * (tensor.numel() - 1))
        return tensor.kthvalue(k).values
