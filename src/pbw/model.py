# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The predictor. One sentence, one compound, one dose, one probability.

    p(the instruction is satisfied | sentence, compound, dose)

The sentence goes through a frozen sentence encoder (nli-mpnet-base-v2) and the
compound through its chemistry vector. A shared trunk mixes the two and emits a few
parameters of a dose-response curve, and the dose is applied to that curve rather
than fed in as another input feature. The reason is that the shape in dose is the
part we actually know something about, so it is worth writing down instead of
hoping a network discovers it:

    beta    p = sigmoid(b0 + b1 * z)
            monotone. b1 is the steepness, -b0/b1 is the dose at which the
            probability crosses one half, which is an EC50 in z units.

    window  p = sigmoid(b0 + b1 * z) * sigmoid(c0 - softplus(b3) * z)
            effect times survival. Non-monotone, so the curve can rise and then
            fall. This is not decoration: under the v3 definition, 61 of the 82
            compounds that ever brighten mitochondria in living cells do it at a
            middle dose and not at the top dose, because the top dose kills. A
            monotone curve cannot express that, and will prefer the toxic compound.

softplus keeps the survival term falling with dose, so the two factors cannot swap
roles and fit the window upside down.

Only the small network is trained here; the sentence encoder stays frozen. Fine
tuning its last layers did raise top-1 accuracy, but it degraded refusal, so it is
not the default. See MODEL_CARD.md.
"""
import torch
import torch.nn as nn

N_PARAMS = {'beta': 2, 'window': 4}


class Net(nn.Module):
    def __init__(self, out='window', d_chem=2048, d_text=768, h=256):
        super().__init__()
        if out not in N_PARAMS:
            raise ValueError(f'out must be one of {sorted(N_PARAMS)}')
        self.out = out
        self.text = nn.Sequential(nn.Linear(d_text, h), nn.GELU())
        self.chem = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_chem, h), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, N_PARAMS[out]))

    def curve(self, e, fp):
        """The dose-response parameters for this (sentence, compound) pair."""
        return self.head(self.trunk(torch.cat([self.text(e), self.chem(fp)], -1)))

    def forward(self, e, fp, z):
        prm = self.curve(e, fp)
        effect = torch.sigmoid(prm[:, 0] + prm[:, 1] * z)
        if self.out == 'beta':
            return effect
        survival = torch.sigmoid(prm[:, 2] - nn.functional.softplus(prm[:, 3]) * z)
        return effect * survival


def bce(p, y):
    """Cross-entropy against a SOFT target. y is a probability, not a 0/1 class, so
    a group just past the threshold costs less to get wrong than one far past it."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
