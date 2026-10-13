from lunar_workspace import load_array, open_array, unlink_array
from lunar_filters import gaussian_filter, gaussian_filter1d
"""Local lunar texture processing from four source images."""
from pathlib import Path
from lunar_workers import ThreadPoolExecutor
from dataclasses import asdict
import argparse, contextlib, io, json, os, tempfile
from PIL import Image
import numpy as np
import PIL, scipy
from scipy import ndimage as ndi
from kaguya_raw import read_scaled_img
from lunar_detect import Settings, detect_strips
from lunar_registration import estimate_registration, RegistrationRejected
from lunar_mask_policy import MaskPolicy, masks_from_regions, apply_masks
from lunar_io import cached_source, blend_fields
Image.MAX_IMAGE_PIXELS=None

def load_source(path, name):
    return cached_source(path, name, _load_source)

def _load_source(path, name):
    if Path(path).suffix.lower()=='.img':return read_scaled_img(path,name)
    with Image.open(path) as image:
        a=np.array(image)
    expected=10800 if name=='Seamless' else 12288
    if a.dtype!=np.uint16 or a.shape!=(expected,expected):
        raise ValueError(f'{name}: expected {expected} square uint16 PNG or raw IMG.')
    return a

def prepare(orbiter, paths, settings=Settings(), quiet=False):
    N=12288
    with Image.open(orbiter) as image:
        reference=np.asarray(image.convert('L'))
        if np.asarray(image).dtype!=np.uint8:raise ValueError('Orbiter must be uint8.')
    report={'schema_version':1,'algorithm':'lunar-local-processing-2','settings':asdict(settings),'mask_policy':asdict(MaskPolicy()),
            'versions':{'Pillow':PIL.__version__,'NumPy':np.__version__,'SciPy':scipy.__version__},
            'sources':{},'limitations':['Persistent axis-aligned strips only; visual review required.', 'Mask-policy defaults require validation across additional tiles.',
            'Evening detail uses reversed polarity.']}
    sources={};transforms={}
    for name,path in paths.items():
        scaled=load_source(path,name)
        strips,profiles=detect_strips(scaled,settings)
        if name=='Seamless':
            from lunar_io import resize_source
            sources[name]=resize_source(scaled,N)
        else:sources[name]=scaled
        try:
            transform,registration=estimate_registration(reference,sources[name],name)
        except RegistrationRejected as error:
            del sources[name]
            report['sources'][name] = {
                'input_path': str(Path(path).resolve()),
                'input_size_bytes': Path(path).stat().st_size,
                'enabled': False, 'rejection_reason': str(error),
                'registration': error.patches,
                'strips': strips, 'strip_profiles': profiles}
            if not quiet:
                print('Excluded source:', error, flush=True)
            continue
        transforms[name]=transform
        report['sources'][name]={'input_path':str(Path(path).resolve()),
            'input_size_bytes':Path(path).stat().st_size,'enabled':True,'registration':registration,
            'transform_yx':transform.tolist(),'strips':strips,
            'masks':masks_from_regions(strips,N),'strip_profiles':profiles}
        if not quiet:print(name,'registered:',len(registration),'patches;',len(strips),'suspect strips',flush=True)
    return sources,transforms,report

def source_detail(name,a,transform,strips,yy,xx,N,sigma,settings):
    tr = transform
    sy = tr[0,0]+tr[1,0]*yy+tr[2,0]*xx
    sx = tr[0,1]+tr[1,1]*yy+tr[2,1]*xx
    rawx = xx-sx
    v = ndi.map_coordinates(a,[yy-sy,rawx],order=1,mode='nearest').astype(np.float32)/65535
    valid = np.clip((v-.01)/.035,0,1)*np.clip((.995-v)/.035,0,1)
    valid = apply_masks(valid,strips,yy-sy,rawx,N)
    denom = gaussian_filter(valid,sigma)
    mean = gaussian_filter(v*valid,sigma)/np.maximum(denom,1e-5)
    d = np.clip((v-mean)/(mean+.15),-.6,.6)
    # Evening relief has opposite illumination; use its fine relief with reversed polarity.
    if name == 'Evening':
        d = -d

    weight = gaussian_filter(valid,4)*np.clip(denom/.7,0,1)
    return d, weight

def blend(orbiter,sources,transforms,report,settings=Settings(),quiet=False,source_workers=1,output_size=12288):
    N=output_size
    ref = Image.open(orbiter).convert('L').convert('F').resize((N,N), Image.Resampling.BICUBIC)
    base = np.asarray(ref,dtype=np.float32)/255
    del ref
    out = np.empty((N,N),dtype=np.uint16)
    cached=blend_fields(N)
    fallback_pixels = 0
    block, halo, sigma = 1024, 160, 32
    with ThreadPoolExecutor(max_workers=source_workers) as pool:
        for start in range(0,N,block):
            end = min(N,start+block)
            lo, hi = max(0,start-halo), min(N,end+halo)
            yy,xx = np.mgrid[lo:hi,0:N].astype(np.float32)
            details, weights = {}, {}
            futures={name:pool.submit(source_detail,name,a,transforms[name],report['sources'][name]['masks'],yy,xx,N,sigma,settings) for name,a in sources.items()}
            for name,future in futures.items():
                details[name],weights[name]=future.result()
            for name in ['Seamless', 'Morning', 'Evening']:
                if name not in details:
                    details[name] = np.zeros(yy.shape, dtype=np.float32)
                    weights[name] = np.zeros(yy.shape, dtype=np.float32)
            ws=weights['Seamless']
            wm=(1-ws)*weights['Morning']
            we=(1-ws)*(1-weights['Morning'])*weights['Evening']
            total=ws+wm+we
            detail=(ws*details['Seamless']+wm*details['Morning']+we*details['Evening'])/np.maximum(total,1e-5)
            detail *= np.clip(total/.8,0,1)
            sl=slice(start-lo,end-lo)
            fallback_pixels += int(np.count_nonzero(total[sl] == 0))
            smooth_base=gaussian_filter(base[lo:hi],sigma)[sl]
            value=np.clip(smooth_base*(1+.75*detail[sl]),0,1)
            out[start:end] = np.rint(value*65535).astype(np.uint16)
            if cached is not None:
                cached['value'][start:end]=value;cached['total'][start:end]=total[sl]
                cached['base'][start:end]=smooth_base
                cached['morning_detail'][start:end]=details['Morning'][sl]
                cached['morning_weight'][start:end]=weights['Morning'][sl]
            if not quiet: print('Processed rows',start,end,flush=True)
    report['orbiter_fallback_pixels'] = fallback_pixels
    if cached is not None:
        for field in cached.values():field.flush()
    report['orbiter_fallback_percent'] = 100.0 * fallback_pixels / (N * N)
    if not quiet:
        print(f"Orbiter-only fallback: {report['orbiter_fallback_percent']:.2f}% of output pixels", flush=True)
    return out

def save_output(out,output,report):
    output=Path(output)
    report_path=output.with_suffix('.diagnostics.json')
    if output.exists() or report_path.exists():raise FileExistsError('Output or diagnostics already exist.')
    array_output=output.suffix.lower()=='.npy'
    fd,temp=tempfile.mkstemp(prefix=output.stem+'.',suffix='.npy' if array_output else '.png',dir=output.parent);os.close(fd)
    try:
        if array_output:
            np.save(temp,out,allow_pickle=False)
            decoded=load_array(temp,mmap_mode='r',allow_pickle=False)
            if decoded.dtype!=out.dtype or not np.array_equal(decoded,out):
                raise RuntimeError('Saved output does not match computed pixels.')
        else:
            Image.fromarray(out).save(temp,compress_level=2)
            with Image.open(temp) as image:
                decoded=np.asarray(image)
                if decoded.dtype!=out.dtype or not np.array_equal(decoded,out):
                    raise RuntimeError('Saved output does not match computed pixels.')
        del decoded
        # Publish the complete PNG atomically without replacing an existing file.
        # Requires a filesystem supporting hard links (e.g. local NTFS/ext4).
        os.link(temp,output)
        with open(report_path,'x') as dst:json.dump(report,dst,indent=2)
    finally:unlink_array(temp)

def build_base_texture(orbiter,seamless,morning,evening,output,quiet=False,settings=Settings(),source_workers=1):
    output=Path(output).resolve()
    inputs=[Path(p).resolve() for p in [orbiter,seamless,morning,evening]]
    if output in inputs:raise ValueError('Output cannot overwrite an input.')
    report_path=output.with_suffix('.diagnostics.json')
    if output.exists() or report_path.exists():raise FileExistsError('Output or diagnostics already exist.')
    if not output.parent.is_dir():raise ValueError('Output folder does not exist.')
    if source_workers not in (1,2,3):raise ValueError('source_workers must be 1, 2, or 3.')
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture) if quiet else contextlib.nullcontext():
        sources,transforms,report=prepare(orbiter,dict(zip(['Seamless','Morning','Evening'],[seamless,morning,evening])),settings,quiet)
        out=blend(orbiter,sources,transforms,report,settings,quiet,source_workers)
    report['source_workers']=source_workers
    del sources
    save_output(out,output,report)
    if not quiet:print('Saved:',output,flush=True)
    return report

def build_texture(orbiter,seamless,morning,evening,output,quiet=False,settings=Settings(),source_workers=1):
    from lunar_pipeline import build_texture as run_pipeline
    return run_pipeline(orbiter,seamless,morning,evening,output,quiet,settings,source_workers)

if 'orbiterFile' in globals():
    build_texture(orbiterFile,seamlessFile,morningFile,eveningFile,outputFile,bool(globals().get('quiet',False)),source_workers=int(globals().get('sourceWorkers',1)))
elif __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['orbiter','seamless','morning','evening','output']:parser.add_argument(name)
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--source-workers',type=int,choices=[1,2,3],default=1)
    args=parser.parse_args();build_texture(args.orbiter,args.seamless,args.morning,args.evening,args.output,args.quiet,source_workers=args.source_workers)
