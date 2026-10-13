from lunar_filters import gaussian_filter, gaussian_filter1d
"""Deterministic registration and persistent strip detection. No models/services."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi, optimize
from PIL import Image

@dataclass(frozen=True)
class Settings:
    analysis_size: int = 1536
    patch_size: int = 256
    grid_points: int = 6
    search_radius: float = 8.0
    minimum_correlation: float = 0.40
    strip_z_threshold: float = 2.5
    strip_persistence: float = 0.50
    strip_guard_fraction: float = 0.015625
    strip_feather_fraction: float = 0.013020833333333334

def thumbnail(a, n):
    return np.asarray(Image.fromarray(a).convert('F').resize((n,n), Image.Resampling.BICUBIC), dtype=np.float32)

def edge_image(a):
    # Magnitude tolerates illumination polarity changes; compress outliers.
    a = gaussian_filter(a.astype(np.float32), 0.8)
    g = np.hypot(ndi.sobel(a,0), ndi.sobel(a,1))
    scale = max(float(np.percentile(g,75)), 1e-6)
    return np.log1p(g/scale)

def correlation(a,b):
    a=a-a.mean(); b=b-b.mean()
    return float(np.sum(a*b)/max(float(np.sqrt(np.sum(a*a)*np.sum(b*b))),1e-12))

def patch_shift(ref, src, y, x, size, radius):
    margin=int(np.ceil(radius))+2
    yy,xx=np.mgrid[y+margin:y+size-margin,x+margin:x+size-margin].astype(np.float32)
    target=ref[y+margin:y+size-margin,x+margin:x+size-margin]
    def objective(s):
        # Positive displacement means sample source at destination minus shift.
        sample=ndi.map_coordinates(src,[yy-s[0],xx-s[1]],order=1,mode='nearest')
        return -correlation(target,sample)
    # Integer search avoids locking the subpixel fit onto a nearby local maximum.
    best=min(((objective((dy,dx)),dy,dx) for dy in range(-int(radius),int(radius)+1)
              for dx in range(-int(radius),int(radius)+1)), key=lambda v:v[0])
    fit=optimize.minimize(objective,[best[1],best[2]],method='Powell',
                          bounds=[(-radius,radius)]*2, options={'xtol':0.01,'ftol':1e-6})
    return fit.x, -float(fit.fun)

def register(reference, source, settings=Settings()):
    n=settings.analysis_size; size=settings.patch_size
    ref=edge_image(thumbnail(reference,n)); src=edge_image(thumbnail(source,n))
    positions=np.rint(np.linspace(24,n-size-24,settings.grid_points)).astype(int)
    measurements=[]
    for y in positions:
        for x in positions:
            shift,score=patch_shift(ref,src,int(y),int(x),size,settings.search_radius)
            measurements.append({'center_yx':[float(y+size/2),float(x+size/2)],
                                 'shift_yx':shift.tolist(),'correlation':score})
    points=np.array([m['center_yx'] for m in measurements])
    shifts=np.array([m['shift_yx'] for m in measurements])
    scores=np.array([m['correlation'] for m in measurements])
    keep=(scores>=settings.minimum_correlation)&(np.max(np.abs(shifts),axis=1)<settings.search_radius-0.3)
    if keep.sum()<6: raise ValueError('Registration uncertain: fewer than six reliable patches.')
    A=np.column_stack([np.ones(len(points)),points/n])
    for _ in range(5):
        tr=np.linalg.lstsq(A[keep]*np.sqrt(scores[keep,None]),shifts[keep]*np.sqrt(scores[keep,None]),rcond=None)[0]
        residual=np.linalg.norm(A@tr-shifts,axis=1)
        med=float(np.median(residual[keep])); mad=float(np.median(np.abs(residual[keep]-med)))
        next_keep=keep & (residual<=max(0.6,med+3*1.4826*mad))
        if np.array_equal(next_keep,keep) or next_keep.sum()<6:break
        keep=next_keep
    for m,k,r in zip(measurements,keep,residual):m.update(accepted=bool(k),residual=float(r))
    if np.linalg.matrix_rank(A[keep])<3 or np.median(residual[keep])>1.2:
        raise ValueError('Registration uncertain: insufficient coverage or inconsistent shifts.')
    return {'transform_normalized_yx':tr.tolist(),'analysis_size':n,
            'accepted_patches':int(keep.sum()),'median_correlation':float(np.median(scores[keep])),
            'median_residual_analysis_px':float(np.median(residual[keep])), 'patches':measurements}

def detect_strips(source, settings=Settings()):
    """Identify long, anomalous derivative lines; return source-relative regions.

    This first detector handles persistent axis-aligned discontinuities only.
    Suspect strips are downweighted, never synthesized or inpainted.
    """
    n=settings.analysis_size
    a=thumbnail(source,n); a=a/max(float(np.max(a)),1e-6)
    regions=[]; profiles={}
    for axis,label in [(1,'x'),(0,'y')]:
        work=a if axis==1 else a.T
        smooth=gaussian_filter(work,0.7)
        derivative=np.abs(np.diff(smooth,axis=1))
        # Require the anomaly to recur across many independent sections.
        section_scores=[]
        for section in np.array_split(derivative,12,axis=0):
            p=np.median(section,axis=0)
            baseline=ndi.median_filter(p,size=81,mode='reflect')
            spread=ndi.median_filter(np.abs(p-baseline),size=81,mode='reflect')
            section_scores.append((p-baseline)/np.maximum(1.4826*spread,0.00015))
        scores=np.array(section_scores)
        persistence=np.mean(scores>settings.strip_z_threshold,axis=0)
        suspect=persistence>=settings.strip_persistence
        suspect=ndi.binary_closing(suspect,iterations=2)
        labels,count=ndi.label(suspect)
        for i in range(1,count+1):
            indices=np.flatnonzero(labels==i); low=int(indices[0]); high=int(indices[-1])+1
            regions.append({'axis':label,'start_fraction':low/n,'end_fraction':high/n,
                            'persistence':float(np.max(persistence[indices])),
                            'peak_z':float(np.median(scores[:,indices],axis=0).max())})
        profiles[label]={'persistence':persistence.tolist(),'median_z':np.median(scores,axis=0).tolist()}
    return regions,profiles

def strip_weight(regions, y, x, size, settings=Settings()):
    weight=np.ones(np.broadcast_shapes(y.shape,x.shape),dtype=np.float32)
    guard=settings.strip_guard_fraction*size
    feather=settings.strip_feather_fraction*size
    for region in regions:
        coordinate=x if region['axis']=='x' else y
        low=region['start_fraction']*size-guard
        high=region['end_fraction']*size+guard
        distance=np.maximum(low-coordinate,coordinate-high)
        weight*=np.clip(distance/max(feather,1e-6),0,1)
    return weight

def pixel_transform(registration, output_size):
    tr=np.array(registration['transform_normalized_yx'],dtype=float)
    tr*=output_size/registration['analysis_size']
    tr[1:]/=output_size
    return tr
