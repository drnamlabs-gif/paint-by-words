# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The output shape is a modelling claim, so it is tested as one."""
import numpy as np
import torch

from pbw.images import count_nuclei, otsu, scale_from, to_grey
from pbw.model import Net, bce


def test_beta_is_monotone_in_dose():
    torch.manual_seed(0)
    net = Net('beta', d_chem=16).eval()
    e, fp = torch.randn(1, 768), torch.randn(1, 16)
    z = torch.linspace(-2, 2, 9)
    p = torch.stack([net(e, fp, z[i:i + 1])[0] for i in range(9)]).detach().numpy()
    assert np.all(np.diff(p) >= -1e-6) or np.all(np.diff(p) <= 1e-6)


def test_window_can_rise_then_fall():
    """A dose window is the reason this output exists: it must be reachable."""
    net = Net('window', d_chem=4).eval()
    with torch.no_grad():                      # b0, b1, c0, b3 -> effect up, survival down
        net.head[-1].weight.zero_()
        net.head[-1].bias.copy_(torch.tensor([-2.0, 3.0, 2.0, 1.0]))
    e, fp = torch.zeros(1, 768), torch.zeros(1, 4)
    z = torch.linspace(-2, 2, 21)
    p = torch.stack([net(e, fp, z[i:i + 1])[0] for i in range(21)]).detach().numpy()
    assert p.argmax() not in (0, len(p) - 1), 'the maximum must be at a middle dose'


def test_the_loss_is_minimised_at_the_soft_target():
    """A soft label of 0.7 asks the model to predict 0.7, not 1.0: a group just past
    the threshold should not be pushed as hard as one far past it."""
    y = torch.tensor([0.7])
    at_target = bce(torch.tensor([0.7]), y)
    assert at_target < bce(torch.tensor([0.9]), y)
    assert at_target < bce(torch.tensor([0.5]), y)
    assert at_target < bce(torch.tensor([1.0 - 1e-6]), y)


def test_otsu_separates_a_two_level_image():
    """On a perfectly bimodal image any cut between the two levels is optimal, so
    what matters is that the mask separates them, not the exact value returned."""
    img = np.concatenate([np.full(500, 20), np.full(500, 200)]).astype(np.uint8)
    t = otsu(img)
    assert 20 <= t < 200
    assert (img > t).sum() == 500


def _blob(img, r, c, size=5, value=200):
    img[r:r + size, c:c + size] = value


def test_nuclei_counting_uses_area_not_components():
    """Two touching nuclei merge into ONE connected component, so component counting
    undercounts exactly in the dense wells that matter. The area estimate must not.
    (The area estimate needs at least five clean nuclei to size one nucleus, which
    is why this image has eight.)"""
    img = np.zeros((96, 96), np.uint8)
    for i in range(8):
        _blob(img, 5 + 11 * i, 5)             # eight separated nuclei
    _blob(img, 60, 60, size=5)
    _blob(img, 60, 65, size=5)                # a touching pair: 1 component, 2 nuclei
    n_obj, n_area = count_nuclei(img)
    assert n_obj == 9, 'the merged pair is counted once by components'
    assert n_area > n_obj, 'the area estimate recovers the merged nucleus'


def test_the_brightness_scale_comes_from_the_reference_only():
    """Both wells must be mapped by the SAME scale, or a dim treated well and a
    bright reference well render identically. That bug made an earlier vision round
    score at chance on four of eight behaviours."""
    rng = np.random.default_rng(0)
    reference = rng.integers(60, 180, (64, 64)).astype(np.uint8)
    treated = (reference // 4).astype(np.uint8)
    a, b = scale_from(reference)
    assert to_grey(treated, a, b).mean() < to_grey(reference, a, b).mean() - 50
    # the failure mode: scaling each image by its own percentiles hides the difference
    ta, tb = scale_from(treated)
    assert abs(float(to_grey(treated, ta, tb).mean())
               - float(to_grey(reference, a, b).mean())) < 10
