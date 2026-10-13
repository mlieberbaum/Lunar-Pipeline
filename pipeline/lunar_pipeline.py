"""Run base blending, measured-detail recovery and the complete refinement."""
from pathlib import Path
import contextlib
import json
import tempfile
import time
from lunar_detect import Settings
from lunar_io import reuse_sources, release_blend_fields
from lunar_workspace import mapping_scope, claim_output


def _keep_debug(work):
    """Opt-in: LUNAR_DEBUG_DIR keeps base image, support and repair masks."""
    import os, shutil
    target = os.environ.get('LUNAR_DEBUG_DIR')
    if not target:
        return
    target = Path(target); target.mkdir(parents=True, exist_ok=True)
    names = ['base.npy', 'candidate.npy', 'candidate.repair-mask.npy', 'candidate.old-detail-support.npy',
             'candidate.measured-support.npy', 'cache/availability.npy', 'cache/big_core.npy',
             'cache/small_core.npy', 'cache/quilt_core.npy', 'cache/tone_alpha.npy', 'cache/output.npy',
             'cache/changed.npy', 'cache/detail_gains.json']
    for name in names:
        source = work / name
        if source.exists():
            shutil.copyfile(source, target / name.replace('/', '_'))


def _build_texture(orbiter, seamless, morning, evening, output, quiet=False,
                  settings=Settings(), source_workers=1):
    from lunar_texture_auto import build_base_texture
    from repair_lunar_detail import repair
    from full_tile_trial import Trial

    output = Path(output).resolve()
    paths = [Path(p).resolve() for p in [orbiter, seamless, morning, evening]]
    if output in paths:
        raise ValueError('Output cannot overwrite an input.')
    if output.exists() or output.with_suffix('.diagnostics.json').exists():
        raise FileExistsError('Output or diagnostics already exist.')
    if not output.parent.is_dir():
        raise ValueError('Output folder does not exist.')
    if source_workers not in (1, 2, 3):
        raise ValueError('source_workers must be 1, 2, or 3.')
    orbiter, seamless, morning, evening = map(str, paths)
    with tempfile.TemporaryDirectory(prefix='lunar_work_', dir=output.parent) as folder, mapping_scope(), reuse_sources(Path(folder)/'sources'):
        work = Path(folder)
        base = work / 'base.npy'
        candidate = work / 'candidate.npy'
        # Stream quiet logs to disk rather than retaining them in memory.
        with open(work / 'processing.log', 'w') as log:
            with contextlib.redirect_stdout(log) if quiet else contextlib.nullcontext():
                before=time.perf_counter()
                from lunar_texture_auto import load_source
                from lunar_source import align_reference
                import lunar_geometry, lunar_source
                from lunar_detect import detect_strips
                codes = {'Seamless': load_source(seamless, 'Seamless'), 'Morning': load_source(morning, 'Morning'),
                         'Evening': load_source(evening, 'Evening')}
                seams, _ = detect_strips(lunar_source.preview(codes['Seamless']), settings)
                anchor, geometry_report = lunar_geometry.align_products(codes, seams, 10800 / 12288)
                lunar_source.ANCHOR = anchor
                orbiter, reference_transform = align_reference(orbiter, codes[anchor], work / 'reference_aligned.png', name=anchor)
                del codes
                import numpy as np, lunar_source
                gains = lunar_source.gain_field(orbiter, load_source(seamless, 'Seamless'))
                lunar_source.set_gain_field(gains)
                np.save(work / 'detail_gain_field.npy', gains)
                base_report = build_base_texture(orbiter, seamless, morning, evening,
                    base, quiet=quiet, settings=settings, source_workers=source_workers)
                base_seconds=time.perf_counter()-before;before=time.perf_counter()
                repair_report = repair(orbiter, seamless, morning, evening, base,
                    base.with_suffix('.diagnostics.json'), candidate)
                recovery_seconds=time.perf_counter()-before
                release_blend_fields()
                config = dict(orbiter=orbiter, seamless=seamless, morning=morning,
                    evening=evening, current=str(candidate),
                    old_support=str(candidate.with_suffix('.old-detail-support.npy')),
                    support=str(candidate.with_suffix('.measured-support.npy')),
                    repair_strength=str(candidate.with_suffix('.repair-mask.npy')),
                    original_diagnostics=str(base.with_suffix('.diagnostics.json')),
                    repair_diagnostics=str(candidate.with_suffix('.diagnostics.json')),
                    work_folder=str(work / 'cache'), output=str(output),
                    size=12288, filter_workers=source_workers)
                trial = Trial(config)
                trial.run()
                _keep_debug(work)
        report = trial.report
        report['base_processing'] = base_report
        report['geometry'] = geometry_report
        report['reference_alignment_transform_yx'] = None if reference_transform is None else reference_transform.tolist()
        report['detail_gain_field_percentiles_5_50_95'] = np.percentile(gains, [5, 50, 95]).tolist()
        report['measured_detail_recovery'] = repair_report
        report['stage_seconds']['base_processing']=base_seconds
        report['stage_seconds']['measured_detail_recovery']=recovery_seconds
        report['algorithm'] = 'lunar-complete-pipeline-4'
        report['runtime_version'] = '4.1'
        output.with_suffix('.diagnostics.json').write_text(json.dumps(report, indent=2))
    return report

def build_texture(orbiter,seamless,morning,evening,output,quiet=False,settings=Settings(),source_workers=1):
    from threadpoolctl import threadpool_limits
    started=time.perf_counter()
    with claim_output(Path(output).resolve()), threadpool_limits(limits=1):
        report=_build_texture(orbiter,seamless,morning,evening,output,quiet,settings,source_workers)
    report['total_elapsed_seconds']=time.perf_counter()-started
    import lunar_source, lunar_geometry
    lunar_source.set_gain_field(None)
    lunar_geometry.clear()
    lunar_source.ANCHOR = 'Seamless'
    Path(output).with_suffix('.diagnostics.json').write_text(json.dumps(report,indent=2))
    return report
