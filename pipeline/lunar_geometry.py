"""Local geometric alignment between Kaguya products. Conventional, deterministic.

A single translation per product is not enough: Morning/Evening drift by
several pixels across a tile relative to Seamless, and Seamless mosaic strips
can be offset from each other (tears at seams). Displacements are measured on
a regular grid by normalized cross-correlation of edge images, filtered
robustly and interpolated smoothly. Seamless strips are fitted separately so a
tear is corrected on each side without smoothing across the join.

A displacement d(y, x) means: content at output (y, x) is found in the
source at (y - dy, x - dx) in output-pixel units (same sign as transform_yx).
"""
import numpy as np
from scipy import ndimage as ndi
import lunar_source as ls

N = 12288
CELL = 384          # window size (output px)
STEP = 512          # grid spacing (output px)
SEARCH = 32         # maximum displacement searched (output px)
MIN_SCORE = .25


def _edges(v, ok):
    a = np.log(np.maximum(v, ls.MEAN_FLOOR))
    a = ndi.gaussian_filter(a, 1.5)
    g = np.hypot(ndi.sobel(a, 0), ndi.sobel(a, 1))
    w = ndi.binary_erosion(ok, iterations=3)
    if not w.any():
        return None, w
    g = np.minimum(g, np.percentile(g[w], 97)) * w
    return g, w


def _window(code, cy, cx, size, scale, shift=(0.0, 0.0), name=None):
    """Sample a window; the source is read at output (y + shift_y, x + shift_x),
    plus the registered field of `name` if one is set."""
    h = size // 2
    yy, xx = np.mgrid[cy - h:cy + h, cx - h:cx + h].astype(np.float32)
    if name is not None and name in FIELDS:
        yy, xx = source_coords(name, None, yy, xx)
    yy = yy + shift[0]; xx = xx + shift[1]
    return ls.sample(code, (yy + .5) * scale - .5, (xx + .5) * scale - .5)


def _match(a, wa, b, wb):
    """Displacement of b relative to a (b content at a position + d), score."""
    if a is None or b is None or wa.mean() < .4 or wb.mean() < .4:
        return None, 0.0
    a = (a - a[wa].mean()) * wa; b = (b - b[wb].mean()) * wb
    n = a.shape[0]
    fa = np.fft.rfft2(a, (2 * n, 2 * n)); fb = np.fft.rfft2(b, (2 * n, 2 * n))
    cc = np.fft.irfft2(fa * np.conj(fb), (2 * n, 2 * n))
    # Normalize by overlap energy so partial overlaps are not favoured.
    ea = np.fft.irfft2(np.fft.rfft2(a * a, (2 * n, 2 * n)) * np.conj(np.fft.rfft2(wb.astype(float), (2 * n, 2 * n))), (2 * n, 2 * n))
    eb = np.fft.irfft2(np.fft.rfft2(wa.astype(float), (2 * n, 2 * n)) * np.conj(np.fft.rfft2(b * b, (2 * n, 2 * n))), (2 * n, 2 * n))
    score = cc / np.sqrt(np.maximum(ea * eb, 1e-12))
    score = np.fft.fftshift(score)
    c = n
    sub = score[c - SEARCH:c + SEARCH + 1, c - SEARCH:c + SEARCH + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    best = float(sub[iy, ix])
    def refine(m, p, q):
        d = m - 2 * p + q
        return 0.0 if abs(d) < 1e-12 else .5 * (m - q) / d
    dy = refine(sub[iy - 1, ix], sub[iy, ix], sub[iy + 1, ix]) if 0 < iy < sub.shape[0] - 1 else 0.0
    dx = refine(sub[iy, ix - 1], sub[iy, ix], sub[iy, ix + 1]) if 0 < ix < sub.shape[1] - 1 else 0.0
    # Peak at index means a(p) ~ b(p - d): b content appears at a position + d.
    return np.array([iy - SEARCH + dy, ix - SEARCH + dx]), best


def grid_points(size=N):
    return np.arange(STEP // 2, size, STEP)


def _avoid_cuts(cx, cuts, h, cy=0, margin=24):
    """Move a window centre so the window lies entirely on its side of seams."""
    for c in cuts or []:
        if isinstance(c, dict):
            ys = np.arange(cy - h, cy + h, 16)
            span = np.interp(ys, c['rows'], c['low']).min(), np.interp(ys, c['rows'], c['high']).max()
        else:
            span = (c, c)
        if cx - h - margin < span[1] and span[0] < cx + h + margin:
            mid = .5 * (span[0] + span[1])
            cx = span[0] - h - margin if cx < mid else span[1] + h + margin
    return cx


def measure(ref_code, ref_scale, mov_code, mov_scale, mov_field=None, ref_name=None, cuts=None):
    """Displacements of mov relative to ref on the grid; NaN where unreliable.

    The result f means mov is read at output (y + f) to match ref at y."""
    pts = grid_points()
    shifts = np.full((len(pts), len(pts), 2), np.nan, np.float32)
    scores = np.zeros((len(pts), len(pts)), np.float32)
    h = CELL // 2
    for i, cy in enumerate(pts):
        for j, cx in enumerate(pts):
            cy_ = int(np.clip(cy, h, N - h)); cx_ = int(np.clip(_avoid_cuts(cx, cuts, h, cy_), h, N - h))
            va, oka = _window(ref_code, cy_, cx_, CELL, ref_scale, name=ref_name)
            prior = (0.0, 0.0) if mov_field is None else tuple(mov_field[i, j])
            vb, okb = _window(mov_code, cy_, cx_, CELL, mov_scale, prior)
            ea, wa = _edges(va, oka); eb, wb = _edges(vb, okb)
            d, s = _match(ea, wa, eb, wb)
            scores[i, j] = s
            if d is not None and s >= MIN_SCORE:
                shifts[i, j] = np.asarray(prior) - d
    return shifts, scores


def robust_field(shifts, groups=None, tolerance=2.0):
    """Reject outliers against a local median and fill gaps.

    groups: optional integer label per grid column; filtering and filling never
    mix labels (used for Seamless strips separated by seams)."""
    out = shifts.copy()
    labels = np.zeros(shifts.shape[1], int) if groups is None else np.asarray(groups)
    for label in np.unique(labels):
        cols = labels == label
        part = out[:, cols]
        for k in range(2):
            p = part[..., k]
            good = np.isfinite(p)
            if not good.any():
                p[:] = 0.0; continue
            med = ndi.generic_filter(np.where(good, p, np.nan), np.nanmedian, size=3, mode='nearest')
            med = np.where(np.isfinite(med), med, np.nanmedian(p))
            p[good & (np.abs(p - med) > tolerance)] = np.nan
            good = np.isfinite(p)
            if not good.any():
                p[:] = np.nanmedian(med); continue
            idx = ndi.distance_transform_edt(~good, return_distances=False, return_indices=True)
            p[:] = p[tuple(idx)]
            p[:] = ndi.uniform_filter(p, 3, mode='nearest')
        out[:, cols] = part
    return out.astype(np.float32)


def _cut(c, y):
    """x position of a seam at rows y (a traced path or a fixed column)."""
    if isinstance(c, dict):
        return np.interp(y, c['rows'], .5 * (np.asarray(c['low']) + np.asarray(c['high'])))
    return np.full(np.shape(y), float(c))


def _mean_cut(c):
    return float(np.mean(_cut(c, np.asarray(c['rows'])))) if isinstance(c, dict) else float(c)


def strip_groups(strips, size=N, paths=None):
    """Grid-column label per mosaic strip. Seams are traced paths when given
    (sorted by mean position), else interior x-axis strip centres."""
    pts = grid_points(size)
    if paths is not None:
        cuts = sorted(paths, key=_mean_cut)
    else:
        cuts = sorted(.5 * (r['start_fraction'] + r['end_fraction']) * size for r in strips
                      if r['axis'] == 'x' and .025 < r['start_fraction'] and r['end_fraction'] < .975)
    return np.searchsorted([_mean_cut(c) for c in cuts], pts), cuts


def field_at(field, yy, xx, cuts=None, size=N):
    """Interpolate a grid field at output coordinates.

    With cuts (Seamless seams), each pixel uses only grid columns of its own
    strip: values are interpolated along y, and along x within the strip."""
    pts = grid_points(size)
    gy = (np.asarray(yy, np.float32) - pts[0]) / STEP
    if not cuts:
        gx = (np.asarray(xx, np.float32) - pts[0]) / STEP
        return [ndi.map_coordinates(field[..., k], [gy, gx], order=1, mode='nearest') for k in range(2)]
    labels = np.searchsorted([_mean_cut(c) for c in cuts], pts)
    xx = np.asarray(xx, np.float32); yy = np.asarray(yy, np.float32)
    strip = np.zeros(xx.shape, int)
    for c in cuts:
        strip += xx > _cut(c, yy)
    out = [np.zeros(np.shape(xx), np.float32) for _ in range(2)]
    for label in np.unique(strip):
        cols = np.flatnonzero(labels == label)
        sel = strip == label
        if not len(cols):
            # A sliver without its own grid column takes the nearest strip.
            present = np.unique(labels)
            cols = np.flatnonzero(labels == present[np.argmin(np.abs(present - label))])
        sub = field[:, cols]
        local = (np.asarray(xx, np.float32)[sel] - pts[cols[0]]) / STEP
        local = np.clip(local, 0, len(cols) - 1)
        for k in range(2):
            out[k][sel] = ndi.map_coordinates(sub[..., k], [gy[sel], local], order=1, mode='nearest')
    return out


# Registered displacement fields, set once per pipeline call.
FIELDS = {}


def clear():
    FIELDS.clear()
    if 'SEAM_PATHS' in globals():
        SEAM_PATHS.clear()


def set_field(name, grid, cuts=None):
    FIELDS[name] = (np.asarray(grid, np.float32), list(cuts or []))


def source_coords(name, transform, yy, xx):
    """Source read position (output-pixel units) for output (yy, xx)."""
    if name in FIELDS:
        grid, cuts = FIELDS[name]
        fy, fx = field_at(grid, yy, xx, cuts)
        return yy + fy, xx + fx
    tr = transform
    sy = tr[0, 0] + tr[1, 0] * yy + tr[2, 0] * xx
    sx = tr[0, 1] + tr[1, 1] * yy + tr[2, 1] * xx
    return yy - sy, xx - sx


def _poly_fit(shifts, iterations=5):
    """Robust quadratic surface through the grid (per component)."""
    pts = grid_points() / N - .5
    gy, gx = np.meshgrid(pts, pts, indexing='ij')
    A = np.stack([np.ones_like(gy), gy, gx, gy * gy, gy * gx, gx * gx], -1).reshape(-1, 6)
    out = np.zeros_like(shifts)
    for k in range(2):
        b = shifts[..., k].ravel()
        keep = np.isfinite(b)
        if keep.sum() < 12:
            out[..., k] = np.nanmedian(b) if keep.any() else 0.0
            continue
        for _ in range(iterations):
            coef = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
            r = np.abs(A @ coef - b)
            limit = max(1.0, 3 * 1.4826 * np.nanmedian(r[keep]))
            new = np.isfinite(b) & (r <= limit)
            if np.array_equal(new, keep) or new.sum() < 12:
                break
            keep = new
        out[..., k] = (A @ coef).reshape(shifts.shape[:2])
    return out


SEAM_PATHS = []


def align_products(sources, strips, scale_seamless, report=None, min_reliable=.5):
    """Measure displacement fields and choose the geometric anchor.

    Morning (a single orbit, smooth geometry consistent with the Orbiter
    reference) defines the output grid when it registers reliably. Seamless
    mosaic strips carry their own along-track errors (tens of pixels varying
    across each strip, steps at seams), so Seamless is warped into the Morning
    frame with a field filtered only within each strip. Evening gets its own
    field. Returns (anchor_name, report). Without a reliable Morning, Seamless
    stays the anchor and only Evening is measured against it."""
    clear()
    report = {} if report is None else report
    S = sources['Seamless']
    SEAM_PATHS[:] = trace_seams(S, strips, scale_seamless)
    interior = [p for p in SEAM_PATHS if 200 < _mean_cut(p) < N - 200]
    groups, cuts = strip_groups(strips, paths=interior)
    report['seams_traced'] = [{'detected': p['detected'], 'low_range': [min(p['low']), max(p['low'])],
                               'high_range': [min(p['high']), max(p['high'])]} for p in SEAM_PATHS]
    anchor = 'Seamless'
    if 'Morning' in sources:
        raw, scores = measure(S, scale_seamless, sources['Morning'], 1.0, cuts=cuts)
        good = np.isfinite(raw[..., 0])
        report['seamless_vs_morning_reliable_cells_percent'] = 100 * float(good.mean())
        report['seamless_vs_morning_median_score'] = float(np.median(scores))
        if good.mean() >= min_reliable:
            field = robust_field(raw, groups, tolerance=2.0)
            # Morning read at y + f matches Seamless at y, so in the Morning
            # frame Seamless is read at u - f(u) (fields are smooth and small).
            set_field('Seamless', -field, cuts)
            anchor = 'Morning'
            report['seamless_field_range_yx'] = [[float(-field[..., k].max()), float(-field[..., k].min())] for k in range(2)]
    if 'Evening' in sources:
        # Evening is measured against Seamless (already warped into the
        # anchor frame): Seamless has data nearly everywhere, whereas Morning
        # is dark in exactly the shadowed areas where Evening is needed.
        raw, scores = measure(S, scale_seamless, sources['Evening'], 1.0,
                              ref_name='Seamless' if 'Seamless' in FIELDS else None)
        good = np.isfinite(raw[..., 0])
        report['evening_reliable_cells_percent'] = 100 * float(good.mean())
        report['evening_median_score'] = float(np.median(scores))
        if good.mean() >= .3:
            field = robust_field(raw, tolerance=2.0)
            set_field('Evening', field)
            report['evening_field_range_yx'] = [[float(field[..., k].min()), float(field[..., k].max())] for k in range(2)]
    report['anchor'] = anchor
    return anchor, report


TRACE_BLOCK = 64    # rows per seam-trace sample (output px)
TRACE_MARGIN = 96   # search beyond the detected interval (output px)


def trace_seams(code, strips, scale, size=N):
    """Follow each vertical seam row by row.

    Mosaic seams follow orbit tracks and are slanted, so the axis-aligned
    interval from detect_strips only matches them over part of the tile. For
    each block of rows the column profile of mean log reflectance is searched
    for brightness steps near the detected interval; the outermost steps give
    the excluded span (covering narrow sliver strips between two seams).
    Returns a list of {'axis': 'x', 'kind': 'path', 'rows', 'low', 'high'}."""
    paths = []
    rows = np.arange(TRACE_BLOCK // 2, size, TRACE_BLOCK)
    for r in strips:
        if r['axis'] != 'x':
            continue
        lo = r['start_fraction'] * size; hi = r['end_fraction'] * size
        x0 = int(max(0, lo - TRACE_MARGIN)); x1 = int(min(size, hi + TRACE_MARGIN))
        if x1 - x0 < 32:
            continue
        lows = np.full(len(rows), np.nan); highs = np.full(len(rows), np.nan)
        for i, cy in enumerate(rows):
            yy, xx = np.mgrid[cy - TRACE_BLOCK // 2:cy + TRACE_BLOCK // 2, x0:x1].astype(np.float32)
            v, ok = ls.sample(code, (yy + .5) * scale - .5, (xx + .5) * scale - .5)
            if ok.mean() < .8:
                continue
            lv = np.where(ok, np.log(np.maximum(v, ls.MEAN_FLOOR)), np.nan)
            # Two column profiles: brightness (median log) and texture contrast
            # (median absolute horizontal log gradient). A sliver of a low-sun
            # strip can match its neighbours in brightness but not in contrast.
            grad = np.abs(np.diff(lv, axis=1, append=lv[:, -1:]))
            k = 6
            strong = np.zeros(0, int)
            for p in (np.nanmedian(lv, axis=0), np.log(np.nanmedian(grad, axis=0) + 1e-3)):
                if not np.all(np.isfinite(p)):
                    p = np.where(np.isfinite(p), p, np.nanmedian(p))
                c = np.cumsum(np.concatenate([[0], p]))
                left = (c[k:-k] - c[:-2 * k]) / k; right = (c[2 * k:] - c[k:-k]) / k
                step = np.abs(right - left)
                base = np.median(step); spread = 1.4826 * np.median(np.abs(step - base)) + 1e-4
                strong = np.concatenate([strong, np.flatnonzero(step > base + 6 * spread)])
            if len(strong):
                lows[i] = x0 + k + strong.min(); highs[i] = x0 + k + strong.max()
        good = np.isfinite(lows)
        if good.sum() < max(4, .1 * len(rows)):
            # Untraceable: fall back to the detected interval.
            lows[:] = lo; highs[:] = hi
        else:
            # Robust smoothing along the seam; fill untraced blocks.
            for arr in (lows, highs):
                med = ndi.median_filter(np.where(good, arr, np.nanmedian(arr[good])), 7, mode='nearest')
                bad = ~good | (np.abs(arr - med) > 24)
                arr[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), arr[~bad]) if (~bad).any() else med[bad]
                arr[:] = ndi.median_filter(arr, 5, mode='nearest')
        paths.append({'axis': 'x', 'kind': 'path', 'rows': rows.tolist(),
                      'low': lows.tolist(), 'high': highs.tolist(),
                      'detected': [float(lo), float(hi)]})
    return paths


def path_masks(paths, guard=16.0, feather=8.0):
    return [dict(p, guard=guard, feather=feather) for p in paths]


def path_cut_at(path, y):
    return np.interp(y, path['rows'], .5 * (np.asarray(path['low']) + np.asarray(path['high'])))
