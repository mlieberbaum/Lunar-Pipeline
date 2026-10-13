from lunar_filters import gaussian_filter, gaussian_filter1d
"""Deterministic shading and conventional exemplar-quilting helpers.
No AI. Exemplar patches introduce explicitly invented placement of real texture.
Inputs are already aligned, normalized grayscale arrays and diagnostic maps.
"""
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.linalg import cg

def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)

def large(mask,minimum=16384):
    labels,n=ndi.label(mask);counts=np.bincount(labels.ravel(),minlength=n+1)
    keep=counts>=minimum;keep[0]=False
    return keep[labels]

def outside_distance(mask):
    return ndi.distance_transform_edt(~mask) if np.any(mask) else np.full(mask.shape,np.inf)

def harmonic(field,mask,screen=.01):
    mask=mask.copy();mask[[0,-1],:]=False;mask[:,[0,-1]]=False
    y,x=np.where(mask);n=len(y)
    if n==0:return field.copy(),{'unknowns':0,'converged':True}
    lookup=np.full(mask.shape,-1,np.int32);lookup[y,x]=np.arange(n)
    rows=[np.arange(n)];cols=[np.arange(n)];values=[np.full(n,4+screen)]
    rhs=np.zeros(n,np.float64)
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        neighbor=lookup[y+dy,x+dx];inside=neighbor>=0
        rows.append(np.flatnonzero(inside));cols.append(neighbor[inside]);values.append(-np.ones(np.count_nonzero(inside)))
        rhs[~inside]+=field[y[~inside]+dy,x[~inside]+dx]
    matrix=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsr()
    solution,status=cg(matrix,rhs,rtol=1e-7,atol=1e-10,maxiter=2000)
    if status!=0:raise RuntimeError(f'Shading solve did not converge: {status}')
    output=field.copy();output[y,x]=solution
    relative=float(np.linalg.norm(matrix@solution-rhs)/max(np.linalg.norm(rhs),1e-12))
    return output,{'unknowns':n,'converged':True,'relative_residual':relative,'screen':screen}

def mincut(error):
    """New patch goes to the right of a minimum-cost vertical seam."""
    height,width=error.shape;cost=error.astype(np.float64).copy();back=np.zeros(error.shape,np.int16)
    columns=np.arange(width);choices=np.empty((3,width),np.float64)
    for y in range(1,height):
        choices[0,0]=np.inf;choices[0,1:]=cost[y-1,:-1]
        choices[1]=cost[y-1]
        choices[2,:-1]=cost[y-1,1:];choices[2,-1]=np.inf
        best=np.argmin(choices,axis=0)
        back[y]=columns+best-1
        cost[y]+=choices[best,columns]
    path=np.empty(height,np.int32);path[-1]=np.argmin(cost[-1])
    for y in range(height-1,0,-1):path[y-1]=back[y,path[y]]
    return np.arange(width)[None,:]>=path[:,None]

def quilt(texture,mask,good,seed=20261007,patch=128,overlap=32,candidates_per_patch=32):
    if not np.any(mask):
        return texture.copy(), {"donor_patch_count":0,"placed_patch_count":0,"distinct_donors_used":0,"seed":seed,"patch":patch,"overlap":overlap}
    h,w=texture.shape;step=patch-overlap
    coverage=ndi.uniform_filter(good.astype(np.float32),patch)
    candidates=[]
    for y in range(0,h-patch+1,64):
        for x in range(0,w-patch+1,64):
            if coverage[y+patch//2,x+patch//2]>.985:
                donor=texture[y:y+patch,x:x+patch]
                deviation=float(np.std(donor))
                if .001<deviation<.15:candidates.append((y,x,deviation))
    if len(candidates)<8:raise ValueError(f'Only {len(candidates)} adequate donor patches; expand the crop.')
    rng=np.random.Generator(np.random.PCG64(seed));canvas=texture.copy();visited=np.zeros(mask.shape,bool)
    used=[]
    for y in range(0,h,step):
        for x in range(0,w,step):
            ph,pw=min(patch,h-y),min(patch,w-x)
            hole=mask[y:y+ph,x:x+pw]
            if not np.any(hole):continue
            target=canvas[y:y+ph,x:x+pw]
            known=visited[y:y+ph,x:x+pw]|~hole
            indices=rng.choice(len(candidates),size=min(candidates_per_patch,len(candidates)),replace=False)
            best=None
            for index in indices:
                sy,sx,std=candidates[int(index)];donor=texture[sy:sy+ph,sx:sx+pw]
                # Fit against known terrain and previously laid texture, without
                # treating missing-region pixels as measured constraints.
                score=float(np.mean((donor[known]-target[known])**2)) if np.any(known) else std*.001
                if best is None or score<best[0]:best=(score,int(index),donor)
            score,index,donor=best;replace=np.ones(hole.shape,bool)
            if x>0:
                width=min(overlap,pw)
                replace[:,:width]&=mincut((donor[:,:width]-target[:,:width])**2)
            if y>0:
                height=min(overlap,ph)
                replace[:height]&=mincut(((donor[:height]-target[:height])**2).T).T
            replace&=hole
            target[replace]=donor[replace];visited[y:y+ph,x:x+pw]|=hole
            used.append(index)
    return canvas,{'donor_patch_count':len(candidates),'placed_patch_count':len(used),
                   'distinct_donors_used':len(set(used)),'seed':seed,'patch':patch,'overlap':overlap}

def bands(v):
    blurred=[gaussian_filter(v,s) for s in [2,8,24,64]]
    return [v-blurred[0],blurred[0]-blurred[1],blurred[1]-blurred[2],blurred[2]-blurred[3]]

def refine(current,orbiter,old_support,support,repair_strength,seed=20261007,fill_weak_repaired_detail=False):
    """Return shading-only and shading-plus-quilt variants and exact masks."""
    if current.dtype!=np.uint16 or any(a.shape!=current.shape for a in [orbiter,old_support,support,repair_strength]):
        raise ValueError('Aligned matching arrays are required; current must be uint16.')
    v=current.astype(np.float32)/65535
    old_bad=large(ndi.binary_closing(old_support<.85,iterations=8))
    handoff=large(ndi.binary_opening(repair_strength>5/255,iterations=2))
    remaining=large(ndi.binary_closing(support<.85,iterations=8))
    tone_region=outside_distance(old_bad|handoff)<96
    tone_region[[0,-1],:]=False;tone_region[:,[0,-1]]=False
    base=gaussian_filter(orbiter.astype(np.float32),32)
    low=gaussian_filter((v-base)/(base+.15),64)
    small_low=low[::8,::8];small_mask=tone_region[::8,::8]
    extended,solve=harmonic(small_low,small_mask)
    yy,xx=np.mgrid[:v.shape[0],:v.shape[1]].astype(np.float32)/8
    target=ndi.map_coordinates(extended,[yy,xx],order=1,mode='nearest')
    # Subtract the sampled/reinterpolated original, avoiding sampling error in
    # known terrain. Correction has zero support outside the allowed region.
    interpolated=ndi.map_coordinates(small_low,[yy,xx],order=1,mode='nearest')
    tone_alpha=smoothstep(ndi.distance_transform_edt(tone_region)/64)
    correction=np.clip(target-interpolated,-.10,.10)*(base+.15)*tone_alpha
    toned=v+correction
    tone=current.copy();active=tone_alpha>0
    tone[active]=np.rint(np.clip(toned[active],0,1)*65535).astype(np.uint16)
    ribbon=(outside_distance(handoff)<32)&(outside_distance(~handoff)<32) if np.any(handoff) else np.zeros(v.shape,bool)
    weak=np.zeros(v.shape,bool)
    if fill_weak_repaired_detail:
        fine=v-gaussian_filter(v,8)
        local_std=np.sqrt(np.maximum(gaussian_filter(fine*fine,32)-gaussian_filter(fine,32)**2,0))
        trusted=((support>.95)&(old_support>.95)).astype(np.float32)
        expected=gaussian_filter(local_std*trusted,192)/np.maximum(gaussian_filter(trusted,192),1e-6)
        weak=large(ndi.binary_closing((old_bad|handoff)&(local_std<.55*expected),iterations=8),4096)
    texture_region=ndi.binary_closing(remaining|ribbon|weak,iterations=16)
    good=(support>.95)&(old_support>.95)&(outside_distance(texture_region)>64)&(v>.035)&(v<.96)
    coarse=gaussian_filter(v,64)
    texture=(v-coarse)/(coarse+.12)
    synthesized,quilting=quilt(texture,texture_region,good,seed)
    current_bands,synthetic_bands=bands(texture),bands(synthesized)
    change=np.zeros(v.shape,np.float32)
    cap=1-smoothstep((outside_distance(texture_region)-128)/64)
    weights=[]
    for original,new,sigma in zip(current_bands,synthetic_bands,[8,16,32,48]):
        weight=gaussian_filter(texture_region.astype(np.float32),sigma)*cap
        weights.append(weight);change+=weight*(new-original)
    proposed=toned+(coarse+.12)*change
    synthetic=tone.copy();synth_active=np.logical_or.reduce([weight>0 for weight in weights])
    synthetic[synth_active]=np.rint(np.clip(proposed[synth_active],0,1)*65535).astype(np.uint16)
    assert np.array_equal(tone[~active],current[~active])
    assert np.array_equal(synthetic[~synth_active],tone[~synth_active])
    report={'algorithm':'crop-guided-shading-and-exemplar-quilting-1','shading_solve':solve,'quilting':quilting,
        'fill_weak_repaired_detail':fill_weak_repaired_detail,'weak_detail_mask_percent':100*float(np.mean(weak)),
        'tone_changed_percent':100*float(np.mean(tone!=current)),
        'synthetic_changed_percent_relative_to_tone':100*float(np.mean(synthetic!=tone)),
        'protected_tone_pixels_changed':0,'protected_synthetic_pixels_changed':0,
        'limitations':['Exemplar placements are synthetic, not recovered lunar features.',
                       'Donor selection is local to this development crop; full-tile equivalence is not established.',
                       'Settings are provisional and may remove genuine detail inside a seam ribbon.']}
    return tone,synthetic,(tone!=current),(synthetic!=tone),report
