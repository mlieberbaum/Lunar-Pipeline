from lunar_filters import gaussian_filter, gaussian_filter1d
"""Experimental local repair of suppressed measured detail; no terrain synthesis.

Uses existing output/diagnostics to preserve accepted pixels exactly. All masks,
alignment and repair eligibility are image-derived; there are no tile identifiers
or hardcoded geographic repair locations in this module.
"""
from pathlib import Path
from dataclasses import asdict
import argparse, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from lunar_texture_auto import load_source, source_detail, save_output
from lunar_detect import Settings, register, pixel_transform, thumbnail, edge_image
from lunar_mask_policy import apply_masks, masks_from_regions
from lunar_io import read_array, write_array, blend_fields
import lunar_source as ls

N = 12288


def refined_masks(regions):
    """Seamless seam masks: only the detected join is excluded."""
    return ls.narrow_masks(regions, N)


def _legacy_refined_masks(regions):
    """v4: exclude detected intervals plus 32 px, with a 64 px transition."""
    masks = []
    for region in regions:
        low, high = region['start_fraction'] * N, region['end_fraction'] * N
        common = {'axis': region['axis'], 'feather': 64.0}
        if low < .025 * N:
            masks.append(dict(common, kind='low_edge', core=high + 32))
        elif high > .975 * N:
            masks.append(dict(common, kind='high_edge', core=N - low + 32))
        else:
            masks.append(dict(common, kind='interior', center=(low + high) / 2,
                              halfwidth=(high - low) / 2 + 32))
    return masks


def coordinates(transform, yy, xx, name=None):
    """Source read position for output (yy, xx); measured local fields
    (lunar_geometry) take precedence over the affine transform."""
    import lunar_geometry
    return lunar_geometry.source_coords(name, transform, yy, xx)


def local_agreement(seamless, evening, seamless_transform, evening_transform):
    """Conservative local gradient agreement, not a proof of scientific validity."""
    size = 1536
    yy, xx = np.mgrid[:size, :size].astype(np.float32) * (N / size)
    images = []
    for name, source, transform in [('Seamless', seamless, seamless_transform), ('Evening', evening, evening_transform)]:
        sy, sx = coordinates(transform, yy, xx, name)
        coords = [sy * size / N, sx * size / N]
        v = ndi.map_coordinates(thumbnail(ls.preview(np.asarray(source)), size) / 65535, coords, order=1, mode='nearest')
        ok = ndi.map_coordinates(thumbnail((np.asarray(source) > 0).astype(np.uint8) * 255, size) / 255,
                                 coords, order=1, mode='nearest') > .999
        images.append((edge_image(v), ok))
    (a, va), (b, vb) = images
    weight = (va & vb).astype(np.float32)
    average = lambda image: ndi.uniform_filter(image, size=41)
    denominator = average(weight)
    safe = np.maximum(denominator, 1e-6)
    ma, mb = average(a * weight) / safe, average(b * weight) / safe
    covariance = average(a * b * weight) / safe - ma * mb
    var_a = np.maximum(average(a * a * weight) / safe - ma * ma, 0)
    var_b = np.maximum(average(b * b * weight) / safe - mb * mb, 0)
    correlation = covariance / np.maximum(np.sqrt(var_a * var_b), 1e-6)
    return (np.clip((correlation - .25) / .3, 0, 1)
            * np.clip((denominator - .2) / .3, 0, 1)).astype(np.float32)


def recovered_detail(name, source, transform, masks, yy, xx, confidence=None, gain=1.0):
    sy, sx = coordinates(transform, yy, xx, name)
    v, ok = ls.sample(source, sy, sx)
    valid = apply_masks(ok.astype(np.float32), masks, sy, sx, N)
    # Keep broad Evening illumination changes out of the new contribution.
    sigma = 8 if name == 'Evening' else 32
    denominator = gaussian_filter(valid, sigma)
    detail = ls.detail(v, valid, sigma, (-gain if name == 'Evening' else gain) * ls.gain_at(yy, xx, N))
    # No contribution from invalid samples or directly along their boundaries.
    distance = ndi.distance_transform_edt(np.pad(valid > 0, 32, constant_values=True))[32:-32, 32:-32]
    weight = valid * np.clip(distance / 32, 0, 1) * np.clip(denominator / .7, 0, 1)
    if confidence is not None:
        weight *= ndi.map_coordinates(confidence, [yy / 8, xx / 8], order=1, mode='nearest')
    return detail, weight


def combine(details, weights, base):
    ws = weights['Seamless']
    wm = (1 - ws) * weights['Morning']
    we = (1 - ws) * (1 - weights['Morning']) * weights['Evening']
    total = ws + wm + we
    detail = (ws * details['Seamless'] + wm * details['Morning'] + we * details['Evening']) / np.maximum(total, 1e-5)
    detail *= np.clip(total / .8, 0, 1)
    return np.clip(base * (1 + .75 * detail), 0, 1), total


def repair(orbiter, seamless, morning, evening, processed, diagnostics, output):
    output = Path(output).resolve()
    inputs = [Path(p).resolve() for p in [orbiter, seamless, morning, evening, processed, diagnostics]]
    if output in inputs:
        raise ValueError('Output cannot overwrite an input.')
    if not output.parent.is_dir():
        raise ValueError('Output directory must exist.')
    companion = output.with_suffix('.diagnostics.json')
    extension = '.npy' if output.suffix.lower()=='.npy' else '.png'
    mask_path = output.with_suffix('.repair-mask'+extension)
    support_path = output.with_suffix('.old-detail-support'+extension)
    measured_path = output.with_suffix('.measured-support'+extension)
    if any(p.exists() for p in [output, companion, mask_path, support_path, measured_path]):
        raise FileExistsError('Output or a companion file already exists.')
    original_report = json.loads(Path(diagnostics).read_text())
    settings = Settings(**original_report['settings'])
    accepted = read_array(processed)
    if accepted.dtype != np.uint16 or accepted.shape != (N, N):
        raise ValueError('Processed image must be 12288-square grayscale uint16.')
    sources = {name: load_source(path, name) for name, path in
               [('Seamless', seamless), ('Morning', morning), ('Evening', evening)]}
    unscaled_seamless = sources['Seamless']
    from lunar_io import resize_source
    sources['Seamless'] = resize_source(unscaled_seamless,N)
    del unscaled_seamless
    reference = np.asarray(Image.open(orbiter).convert('L'))
    base = np.asarray(Image.fromarray(reference).convert('F').resize((N, N), Image.Resampling.BICUBIC), np.float32) / 255
    recorded = original_report['sources']
    if not recorded['Seamless'].get('enabled', True):
        raise ValueError('This experimental repair requires registered Seamless.')
    transforms = {name: np.array(s['transform_yx']) for name, s in recorded.items()
                  if s.get('enabled', True)}
    report = {'algorithm': 'experimental-local-detail-repair-1',
              'original_diagnostics': str(Path(diagnostics).resolve()),
              'policy': {'eligibility_detail_gain_below': .8, 'full_repair_gain_below': .4,
                         'seam_guard': 32, 'seam_feather': 64, 'evening_sigma': 8,
                         'evening_shadow_margin': 32, 'agreement_window_at_1536': 41,
                         'agreement_zero_below': .25, 'agreement_full_above': .55},
              'limitations': ['Provisional settings tested on two tiles only.',
                              'Local gradient agreement can reject real detail under different illumination.',
                              'No recovery of detail missing from all sources.',
                              'Geographic tears are not repaired by narrowing masks.'],
              'versions': original_report['versions'], 'evening_recovery': {}}
    evening_transform = transforms.get('Evening')
    if evening_transform is None:
        try:
            measured = register(reference, ls.preview(np.asarray(sources['Evening'])), Settings(analysis_size=768,
                                patch_size=128, grid_points=6, search_radius=6, minimum_correlation=.4))
            evening_transform = pixel_transform(measured, N)
            report['evening_recovery'] = {'registration': measured,
                                         'transform_yx': evening_transform.tolist()}
        except ValueError as error:
            report['evening_recovery'] = {'rejected': str(error)}
    confidence = None
    if evening_transform is not None:
        report['evening_recovery']['transform_yx'] = evening_transform.tolist()
        confidence = local_agreement(sources['Seamless'], sources['Evening'], transforms['Seamless'], evening_transform)
        report['evening_recovery']['local_confidence_above_half_percent'] = 100 * float(np.mean(confidence > .5))
    out = accepted.copy()
    repair_mask = np.zeros((N, N), np.uint8)
    support_map = np.zeros((N, N), np.uint8)
    measured_map = np.zeros((N, N), np.uint16)
    changed = eligible = protected_changed = 0
    baseline_max_delta = 0
    original_zero = new_zero = 0
    cached = blend_fields()
    for start in range(0, N, 1024):
        end = min(N, start + 1024)
        lo, hi = max(0, start - 160), min(N, end + 160)
        yy, xx = np.mgrid[lo:hi, :N].astype(np.float32)
        if cached is None:
            details, weights = {}, {}
            for name, source in sources.items():
                if name in transforms:
                    details[name], weights[name] = source_detail(name, source, transforms[name],
                             recorded[name]['masks'], yy, xx, N, 32, settings, recorded[name].get('detail_gain', 1.0))
                else:
                    details[name], weights[name] = np.zeros_like(yy), np.zeros_like(yy)
            smooth_base = gaussian_filter(base[lo:hi], 32)
            old, old_total = combine(details, weights, smooth_base)
        else:
            # Combination is pixelwise; only the core rows survive this block.
            # Cached halo values from neighboring blocks cannot affect that core.
            smooth_base=cached['base'][lo:hi]
            old=cached['value'][lo:hi];old_total=cached['total'][lo:hi]
            details={'Morning':cached['morning_detail'][lo:hi],
                     'Seamless':np.zeros_like(yy),'Evening':np.zeros_like(yy)}
            weights={'Morning':cached['morning_weight'][lo:hi],
                     'Seamless':np.zeros_like(yy),'Evening':np.zeros_like(yy)}
        new_details, new_weights = details.copy(), weights.copy()
        new_details['Seamless'], new_weights['Seamless'] = recovered_detail('Seamless', sources['Seamless'],
                   transforms['Seamless'], _legacy_refined_masks(recorded['Seamless']['strips']), yy, xx)
        if evening_transform is not None:
            new_details['Evening'], new_weights['Evening'] = recovered_detail('Evening', sources['Evening'],
                     evening_transform, masks_from_regions(recorded['Evening']['strips']), yy, xx, confidence,
                     recorded['Evening'].get('detail_gain', 1.0))
        new, new_total = combine(new_details, new_weights, smooth_base)
        gain = np.clip(old_total / .8, 0, 1)
        alpha = np.clip((.8 - gain) / .4, 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        alpha *= new_total > old_total + 1e-6
        sl = slice(start - lo, end - lo)
        alpha = alpha[sl]
        current = accepted[start:end]
        reconstructed = np.rint(old[sl] * 65535).astype(np.uint16)
        baseline_max_delta = max(baseline_max_delta, int(np.max(np.abs(reconstructed.astype(np.int32) - current.astype(np.int32)))))
        # Apply only the proposed difference to accepted pixels. Protected pixels
        # bypass all float conversion and are byte-for-byte copied.
        active = alpha > 0
        block = current.copy()
        correction = 65535 * alpha * (new[sl] - old[sl])
        block[active] = np.rint(np.clip(current[active].astype(np.float64) + correction[active], 0, 65535)).astype(np.uint16)
        out[start:end] = block
        changed += int(np.count_nonzero(block != current))
        eligible += int(np.count_nonzero(active))
        protected_changed += int(np.count_nonzero(block[~active] != current[~active]))
        original_zero += int(np.count_nonzero(old_total[sl] == 0))
        effective_total = (1 - alpha) * old_total[sl] + alpha * new_total[sl]
        new_zero += int(np.count_nonzero(effective_total == 0))
        repair_mask[start:end] = np.rint(alpha * 255).astype(np.uint8)
        support_map[start:end] = np.rint(gain[sl] * 255).astype(np.uint8)
        measured_map[start:end] = np.rint(np.clip(effective_total / .8, 0, 1) * 65535).astype(np.uint16)
        print(f'Rows {start}:{end}', flush=True)
    if baseline_max_delta > 3:
        raise ValueError(f'Recorded baseline is not reproduced within tolerance: max delta {baseline_max_delta}. No output published.')
    report.update(baseline_max_uint16_delta=baseline_max_delta,
                  repair_eligible_percent=100 * eligible / (N * N), changed_pixel_percent=100 * changed / (N * N),
                  protected_pixels_changed=protected_changed,
                  original_full_fallback_percent=100 * original_zero / (N * N),
                  effective_full_fallback_percent=100 * new_zero / (N * N))
    save_output(out, output, report)
    write_array(mask_path, repair_mask)
    write_array(support_path, support_map)
    write_array(measured_path, measured_map)
    print(json.dumps({k: v for k, v in report.items() if k.endswith('percent') or k.endswith('delta') or k == 'protected_pixels_changed'}), flush=True)
    return report


if 'orbiterFile' in globals():
    repair(orbiterFile, seamlessFile, morningFile, eveningFile, processedFile, diagnosticsFile, outputFile)
elif __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['orbiter', 'seamless', 'morning', 'evening', 'processed', 'diagnostics', 'output']:
        parser.add_argument(name)
    args = parser.parse_args()
    repair(**vars(args))
