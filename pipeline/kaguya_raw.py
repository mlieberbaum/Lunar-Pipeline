"""Port of the attached loadKaguyaHiresSet(..., 'stretch') input operations.

All uint16 samples participate, including zero and special values. No DN
scaling factor, nodata removal, flip, gain, gamma, or resampling is added.
"""
from pathlib import Path
import numpy as np

SIZES = {'seamless': 10800, 'morning': 12288, 'evening': 12288}


def matlab_uint16_nonnegative(values):
    """MATLAB nearest rounding, positive halfway values upward (not bankers')."""
    values = np.clip(values, 0, 65535)
    whole = np.floor(values)
    # Avoid floor(x + 0.5), which can itself round a value just below a tie.
    whole += (values - whole >= 0.5)
    return whole.astype(np.uint16)


def stretch_uint16(raw):
    """Return the decoded uint16 PNG-equivalent array and its double limits."""
    if raw.dtype.kind != 'u' or raw.dtype.itemsize != 2 or raw.ndim != 2:
        raise ValueError('Expected a two-dimensional uint16 array.')
    # MATLAB prctile default midpoint/exact = Hazen, NOT NumPy default linear.
    # A temporary double array mirrors double(rawImg(:)); order is irrelevant
    # because percentiles cover the entire image and are computed by sorting.
    samples = raw.astype(np.float64).reshape(-1)
    limits = np.percentile(samples, [0.5, 99.5], method='hazen', overwrite_input=True)
    del samples
    if limits[1] > limits[0]:
        # Evaluate the original double arithmetic once for each possible DN.
        # Elementwise lookup changes neither arithmetic order nor rounding.
        values = (np.arange(65536, dtype=np.float64) - limits[0]) / (limits[1] - limits[0])
        values = np.minimum(np.maximum(values, 0), 1)
        lookup = matlab_uint16_nonnegative(values * 65535)
        scaled = lookup[raw]
    else:
        scaled = np.zeros(raw.shape, dtype=np.uint16)
    return scaled, limits


def read_raw_img(path, product):
    """Read headerless MSB uint16 row-major IMG as label DN (no stretch)."""
    key = str(product).lower()
    if key not in SIZES:
        raise ValueError('Product must be Seamless, Morning, or Evening.')
    path = Path(path)
    size = SIZES[key]
    expected_bytes = 2 * size * size
    if path.stat().st_size != expected_bytes:
        raise ValueError(f'Unexpected IMG size: {path}; expected {expected_bytes} bytes.')
    # C-order reshape matches fread([N N], '*uint16').\' in the MATLAB loader.
    raw = np.fromfile(path, dtype='>u2')
    if raw.size != size * size:
        raise ValueError(f'Incomplete IMG read: {path}')
    return raw.reshape(size, size).astype(np.uint16)


def read_scaled_img(path, product):
    """Display-only 0.5/99.5 stretch, as in the MATLAB preview loader."""
    scaled, limits = stretch_uint16(read_raw_img(path, product))
    print(f'{product}: stretch limits {limits[0]:.17g}, {limits[1]:.17g}', flush=True)
    return scaled
