# early_exit.py
import torch, torch.nn as nn, numpy as np

class EarlyExitWrapper(nn.Module):
    """Add confidence–based early‑exit to any multi‑exit net."""
    def __init__(self, core: nn.Module, T_file: str):
        super().__init__()
        self.core = core
        self.T    = torch.tensor(np.load(T_file), dtype=torch.float32)  # K thresholds
        self.sm   = nn.Softmax(dim=1)

    @torch.no_grad()
    def forward(self, x):
        for k, logits in enumerate(self.core(x)):
            conf = self.sm(logits).amax(1)          # per‑sample max‑prob
            if (conf >= self.T[k].to(conf.device)).all():
                return logits                       # exit k fires
        return logits                               # fall‑back = last exit
