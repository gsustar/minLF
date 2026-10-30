import torch
import torch.nn as nn
import torch.nn.functional as F

from .trm import TRMInner


def _round(x):
    return F.one_hot(x.argmax(dim=-1), num_classes=x.size(-1)).float()

def given_mask(c):
    return c.argmax(dim=-1, keepdim=True) > 0

def interpolant(c, ti, x0, x1):
    I_ti = (1.0 - ti) * x0 + ti * x1
    return torch.where(given_mask(c), c, I_ti)

def cnoise(c):
    return torch.where(given_mask(c), c, torch.randn_like(c))


class LoopedFlow(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_trm_blocks=2, bf16=False):
        super().__init__()
        self.denoiser = TRMInner(
            hidden_size=hidden_size, 
            vocab_size=vocab_size,
            num_trm_blocks=num_trm_blocks,
            L_cycles=4,
            H_cycles=3
        )
        self.bf16 = bf16
        self.lmbda_q = 0.5
        self.dummy = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy.device

    def train_step(self, c, x1, optimizer, k=16):
        bs, seq_len, _ = x1.size()
        x0 = cnoise(c).to(self.device)
        ts, _ = torch.rand(c.size(0), k+1, device=self.device).sort() # NOTE: k or k+1?

        z_ti = self.denoiser.init_carry(bs, seq_len)  # z0
        active = torch.ones(bs).to(self.device)
        losses = dict(loss_x=[], loss_q=[], loss=[])
        for i in range(k):
            optimizer.zero_grad()

            ti = ts[:, i].view(-1, 1, 1)
            I_ti = interpolant(c, ti, x0, x1)
            with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type, enabled=self.bf16):
                x_hat_ti, q_hat, z_ti = self.denoiser(I_ti, ti, z_ti)
            q = (x1 == _round(x_hat_ti)).all(dim=(-2, -1)).unsqueeze(-1).float()

            loss_x = F.cross_entropy(
                x_hat_ti.mT, x1.mT, reduction='none'
            ).mean(dim=-1)
            loss_q = F.binary_cross_entropy_with_logits(
                q_hat, q, reduction='none'
            ).squeeze(-1)

            # a convoluted way to do masking
            loss_x = (loss_x * active).sum() / active.sum().clamp_min(1.0)
            loss_q = (loss_q * active).sum() / active.sum().clamp_min(1.0) * self.lmbda_q
            loss = loss_x + loss_q

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            active = active * (q_hat <= 0).squeeze(-1).float()  # same as sigmoid(q_hat) <= 0.5 but in logit space
            losses["loss_x"].append(loss_x)
            losses["loss_q"].append(loss_q)
            losses["loss"].append(loss)

            if not active.any():
                break

        losses["steps"] = i + 1
        losses["loss"] = torch.stack(losses["loss"]).mean()
        losses["loss_x"] = torch.stack(losses["loss_x"]).mean()
        losses["loss_q"] = torch.stack(losses["loss_q"]).mean()
        return losses

    @torch.no_grad()
    def infer_reps(self, c, n=16, gamma=0.0, reps=1):
        for _ in range(reps):
            preds, raw_x = self.infer(c, n=n, gamma=gamma)
            yield preds, raw_x

    @torch.no_grad()
    def infer(self, c, n=16, gamma=0.0):
        bs, seq_len, _ = c.size()
        ts = torch.linspace(0, 1, n+1).repeat(bs, 1).to(self.device)

        x = cnoise(c).to(self.device)  # x0
        z = self.denoiser.init_carry(bs, seq_len)  # z0
        for i in range(n):
            x, z = self.infer_step(x, z, ts, i, c, gamma=gamma)
        return _round(x), x

    @torch.no_grad()
    def infer_step(self, x_ti, z_ti, ts, i, c, gamma=0.0):
        # gamma = 0.0 reduces to forward Euler
        cmask = given_mask(c)
        ti = ts[:, i].view(-1, 1, 1)
        tip1 = ts[:, i + 1].view(-1, 1, 1)
        a = torch.clamp(1 - gamma * (tip1 - ti), min=0.0, max=1.0)
        s = a * ti
        x_s_bar = a * x_ti + torch.sqrt(
            (1.0 - s) ** 2 - (a - s) ** 2
        ) * torch.randn_like(x_ti)
        x_s_bar = torch.where(cmask, c, x_s_bar)
        with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type, enabled=self.bf16):
            x_s_hat, _, z_ti = self.denoiser(x_s_bar, ti, z_ti)
        x_tip1 = x_s_bar + (tip1 - s) * (x_s_hat - x_s_bar) / (1 - s)
        x_tip1 = torch.where(cmask, c, x_tip1)
        return x_tip1, z_ti