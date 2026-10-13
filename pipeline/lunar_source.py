"""Kaguya source encoding, validity and relative detail shared by every stage.

Sources are stored as uint16 codes linear in the label DN: code = 2*DN for
valid samples (label DN 2..32766) and 0 for DUMMY (0), low saturation (1) and
high saturation (32767). A code therefore equals 0 exactly where the archive
flags the sample as unusable, so validity never depends on a per-tile stretch.

Detail is relative reflectance (v - local mean) / local mean with one fixed
gain, so every product and every tile share the same contrast scale.
"""
import numpy as np
from scipy import ndimage as ndi

DUMMY, LOW_SATURATION, HIGH_SATURATION = 0, 1, 32767
CODE_PER_DN = 2
# Detail model (log reflectance, fixed for every tile):
#   factor = exp(MID_GAIN * (G2 - Gs) + FINE_GAIN * (log v - G2))
# where Gs is the local log mean at the caller's scale (32 px for the base
# blend) and G2 a 2 px log mean. A mid gain near 1 keeps Kaguya structure
# without unsharp-mask halos (LROC carries ~1.2x Kaguya relative contrast where
# both resolve); the fine gain restores crispness of sub-2px features only.
# Detail is returned as d = (factor - 1) / BLEND, so callers keep the existing
# value = base * (1 + BLEND * d) combination.
MID_GAIN = 1.5
FINE_GAIN = 2.2
FINE_SIGMA = 2.0
BLEND = .75
FACTOR_MAX = 3.0
# Reflectance floor for the logarithm, in code/65535 units (about DN 25).
MEAN_FLOOR = 2 * 25 / 65535


def encode(raw):
    """Label DN (uint16) -> source code (uint16); 0 marks unusable samples."""
    raw = np.asarray(raw)
    if raw.dtype.kind != 'u' or raw.dtype.itemsize != 2:
        raise ValueError('Expected uint16 label DN.')
    code = raw.astype(np.uint16) * np.uint16(CODE_PER_DN)
    code[(raw <= LOW_SATURATION) | (raw >= HIGH_SATURATION)] = 0
    return code


def code_to_dn(code):
    return np.asarray(code, np.float32) / CODE_PER_DN


def sample(source, sy, sx):
    """Cubic-spline sample of a code array at (sy, sx).

    Only the bounding window of the requested coordinates is prefiltered.
    A sample is valid inside the source where no unusable code lies within the
    spline support. Bilinear sampling was avoided: its blur depends on the
    sub-pixel phase, which is visible as a texture grid after detail gain."""
    n0, n1 = source.shape
    m = 4
    y0 = max(0, int(np.floor(np.min(sy))) - m); y1 = min(n0, int(np.ceil(np.max(sy))) + m + 1)
    x0 = max(0, int(np.floor(np.min(sx))) - m); x1 = min(n1, int(np.ceil(np.max(sx))) + m + 1)
    window = np.asarray(source[y0:y1, x0:x1])
    coeff = ndi.spline_filter(window.astype(np.float32), order=3, mode='mirror', output=np.float32)
    ly = sy - y0; lx = sx - x0
    v = ndi.map_coordinates(coeff, [ly, lx], order=3, mode='mirror', prefilter=False, output=np.float32) / 65535
    valid = (sy >= 0) & (sx >= 0) & (sy <= n0 - 1) & (sx <= n1 - 1)
    bad = window == 0
    if bad.any():
        bad = ndi.binary_dilation(bad, structure=np.ones((5, 5), bool))
        yi = np.clip(np.rint(ly), 0, bad.shape[0] - 1).astype(np.intp)
        xi = np.clip(np.rint(lx), 0, bad.shape[1] - 1).astype(np.intp)
        valid &= ~bad[yi, xi]
    return v, valid


def valid_code(v):
    """Validity of an already-sampled code value scaled to [0, 1]."""
    return v > 0


def _masked_mean(a, w, sigma):
    from lunar_filters import gaussian_filter
    num = gaussian_filter(a * w, sigma)
    return num / np.maximum(gaussian_filter(w, sigma), 1e-5)


def detail(v, valid, sigma, gain=1.0):
    """Multiplicative detail of v about its local mean at scale sigma.

    valid is a 0..1 weight; invalid samples do not enter the local means and
    receive zero detail. gain is a per-source contrast match (1 for Seamless)."""
    w = np.asarray(valid, np.float32)
    lv = np.log(np.maximum(v, MEAN_FLOOR)).astype(np.float32)
    coarse = _masked_mean(lv, w, sigma)
    fine = _masked_mean(lv, w, FINE_SIGMA)
    log_factor = gain * (MID_GAIN * (fine - coarse) + FINE_GAIN * (lv - fine))
    factor = np.minimum(np.exp(log_factor), FACTOR_MAX)
    return ((factor - 1) / BLEND * (w > 0)).astype(np.float32)


def preview(code):
    """0.5/99.5 percentile stretch of valid codes for registration/detection only."""
    from kaguya_raw import matlab_uint16_nonnegative
    valid = code[code > 0]
    if valid.size == 0:
        return np.zeros(code.shape, np.uint16)
    lo, hi = np.percentile(valid[::max(1, valid.size // 4000000)], [0.5, 99.5])
    hi = max(hi, lo + 1)
    lut = matlab_uint16_nonnegative(np.clip((np.arange(65536) - lo) / (hi - lo), 0, 1) * 65535)
    lut[0] = 0
    return lut[code]


def resize_code(code, size):
    """Bicubic resize of a code array; any output pixel whose footprint touches
    an unusable sample is unusable."""
    from PIL import Image
    code = np.asarray(code)
    scale = size / code.shape[0]
    out = np.empty((size, size), np.uint16)
    # Cubic spline with pixel-area alignment (as PIL resize), in row bands.
    rows = 1024
    for start in range(0, size, rows):
        end = min(size, start + rows)
        yy = (np.arange(start, end, dtype=np.float32) + .5) / scale - .5
        xx = (np.arange(size, dtype=np.float32) + .5) / scale - .5
        sy, sx = np.meshgrid(yy, xx, indexing='ij')
        v, _ = sample(code, np.clip(sy, 0, code.shape[0] - 1), np.clip(sx, 0, code.shape[1] - 1))
        out[start:end] = np.clip(np.rint(v * 65535), 1, 65535).astype(np.uint16)
    bad = code == 0
    if bad.any():
        bad = ndi.binary_dilation(bad, iterations=2)
        bad = np.asarray(Image.fromarray(bad.astype(np.uint8) * 255).resize((size, size), Image.Resampling.BILINEAR)) > 0
        out[bad] = 0
    return out


def save_gains(folder, gains):
    import json
    from pathlib import Path
    (Path(folder) / 'detail_gains.json').write_text(json.dumps(gains))


def load_gains(folder):
    import json
    from pathlib import Path
    path = Path(folder) / 'detail_gains.json'
    return json.loads(path.read_text()) if path.exists() else {}


def align_reference(orbiter, seamless_code, target, size=12288):
    """Resample the Orbiter reference onto the Seamless grid.

    Kaguya tiles abut exactly on their own grid, so Seamless is kept fixed and
    the low-resolution reference is shifted instead (about one reference pixel).
    Shifting Kaguya would leave edge strips with no source data in every tile.
    Returns (path, transform) or (orbiter, None) if registration is rejected."""
    from PIL import Image
    from lunar_registration import estimate_registration, RegistrationRejected
    with Image.open(orbiter) as image:
        if np.asarray(image).dtype != np.uint8:
            raise ValueError('Orbiter must be uint8.')
        reference = np.asarray(image.convert('L'))
    view = preview(seamless_code)
    view = np.clip(np.rint(np.asarray(Image.fromarray(view).convert('F').resize((size, size), Image.Resampling.BICUBIC))), 0, 65535).astype('uint16')
    try:
        transform, _ = estimate_registration(reference, view, 'Seamless')
    except RegistrationRejected as error:
        print('Reference alignment skipped:', error, flush=True)
        return orbiter, None
    del view
    n = reference.shape[0]
    k = n / size
    U, V = np.mgrid[:n, :n].astype(np.float64)
    # Output-pixel coordinates of each reference pixel centre.
    y = (U + .5) / k - .5; x = (V + .5) / k - .5
    sy = transform[0, 0] + transform[1, 0] * y + transform[2, 0] * x
    sx = transform[0, 1] + transform[1, 1] * y + transform[2, 1] * x
    shifted = ndi.map_coordinates(reference.astype(np.float32), [U + sy * k, V + sx * k], order=3, mode='nearest')
    Image.fromarray(np.clip(np.rint(shifted), 0, 255).astype(np.uint8), 'L').save(target)
    print('Reference aligned to Seamless; shift at centre (px):',
          [round(float(transform[0, i] + (transform[1, i] + transform[2, i]) * size / 2), 2) for i in (0, 1)], flush=True)
    return str(target), np.asarray(transform)
