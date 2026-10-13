from lunar_filters import gaussian_filter, gaussian_filter1d
"""Accepted balanced residual texture, applied in map-derived fixed windows. No AI."""
import numpy as np
from scipy import ndimage as nd
from frozen_algorithm import mincut,smoothstep,large,outside_distance

def residual(image,old,support,repair):
    original=image.copy()
    v=original.astype('float32')/65535
    coarse=gaussian_filter(v,64);fine=v-gaussian_filter(v,8)
    rms=np.sqrt(gaussian_filter(fine*fine,24))
    trusted=(support>.98)&(old>.95)&(repair<.025)
    # Contrast references use similar brightness, avoiding bright-to-shadow transfer.
    edges=np.linspace(0,1,21);idx=np.clip(np.digitize(coarse,edges)-1,0,19)
    reference=np.zeros(20,'float32')
    for i in range(20):
        pool=rms[trusted&(idx==i)]
        reference[i]=np.median(pool) if len(pool)>500 else np.nan
    known=np.isfinite(reference)
    reference=np.interp(np.arange(20),np.flatnonzero(known),reference[known]).astype('float32')
    ratio=rms/np.maximum(reference[idx],.002)
    weak=(support<.9)&(ratio<.80)&(coarse>.09)&(coarse<.85)
    mask=large(nd.binary_closing(weak,iterations=16),4096)
    mask=nd.binary_fill_holes(mask)
    if not mask.any():
        return original,np.zeros(image.shape,bool),mask,{"placements":0,"mask_percent":0,"protected_pixels_changed":0}
    good=trusted&(outside_distance(mask)>96)&(coarse>.07)&(coarse<.88)
    texture=(v-coarse)/(coarse+.12)
    patch=192;overlap=48;step=144;h,w=v.shape
    coverage=nd.uniform_filter(good.astype('float32'),patch)
    donors=[]
    for y in range(0,h-patch+1,48):
        for x in range(0,w-patch+1,48):
            if coverage[y+patch//2,x+patch//2]>.985:
                donors.append((y,x,float(np.mean(coarse[y:y+patch,x:x+patch]))))
    donors=np.array(donors);assert len(donors)>32
    rng=np.random.Generator(np.random.PCG64(20261008))
    canvas=texture.copy();visited=np.zeros(mask.shape,bool);used=[]
    for y in range(0,h,step):
        for x in range(0,w,step):
            ph,pw=min(patch,h-y),min(patch,w-x);roi=np.s_[y:y+ph,x:x+pw];hole=mask[roi]
            if not hole.any():continue
            target=canvas[roi];known=visited[roi]|~hole;brightness=float(np.mean(coarse[roi][hole]))
            eligible=np.flatnonzero(np.abs(donors[:,2]-brightness)<.07)
            if len(eligible)<32:eligible=np.argsort(np.abs(donors[:,2]-brightness),kind='stable')[:64]
            choices=rng.choice(eligible,min(48,len(eligible)),replace=False);best=None
            for i in choices:
                sy,sx=donors[i,:2].astype(int);donor=texture[sy:sy+ph,sx:sx+pw]
                score=float(np.mean((donor[known]-target[known])**2)) if known.any() else abs(donors[i,2]-brightness)*.001
                if best is None or score<best[0]:best=(score,int(i),donor)
            _,i,donor=best;replace=hole.copy()
            if x>0:
                z=min(overlap,pw);replace[:,:z]&=mincut((donor[:,:z]-target[:,:z])**2)
            if y>0:
                z=min(overlap,ph);replace[:z]&=mincut(((donor[:z]-target[:z])**2).T).T
            # Always cover an unvisited missing pixel, even at a patch overlap.
            replace|=hole&~visited[roi]
            target[replace]=donor[replace];visited[roi]|=hole;used.append(i)
    assert np.all(visited[mask])
    current8=gaussian_filter(texture,8);current32=gaussian_filter(texture,32)
    donor8=gaussian_filter(canvas,8);donor32=gaussian_filter(canvas,32)
    distance=outside_distance(mask);cap=1-smoothstep(distance/96)
    wf=gaussian_filter(mask.astype('float32'),12)*cap
    wm=gaussian_filter(mask.astype('float32'),24)*cap
    fine_change=wf*((canvas-donor8)-(texture-current8))
    medium_change=wm*((donor8-donor32)-(current8-current32))
    # No donor structure above 32px; the crater's broad shading stays the baseline.
    delta=(coarse+.12)*(fine_change+.7*medium_change)
    result=original.copy();active=cap>0
    result[active]=np.rint(np.clip(v[active]+delta[active],0,1)*65535).astype('uint16')
    assert np.array_equal(result[~active],original[~active])
    return result,active,mask,{'donors':len(donors),'placements':len(used),'distinct_donors':len(set(used)),
        'seed':20261008,'mask_percent':float(mask.mean()*100),'protected_pixels_changed':0}
