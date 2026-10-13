"""Match tone and frequency contrast across the whole repair perimeter."""
import numpy as np
from scipy import ndimage as nd
from lunar_filters import gaussian_filter
from frozen_algorithm import harmonic,outside_distance,smoothstep

def blend(image,mask,reference=None,width=160,full_texture=False):
    if not mask.any():return image.copy(),np.zeros(mask.shape,bool),{'changed_pixels':0}
    v=image.astype('float32')/65535;coarse=gaussian_filter(v,64)
    texture=(v-coarse)/(coarse+.12);low=gaussian_filter(texture,8)
    fine=texture-low;medium=low-gaussian_filter(texture,32)
    outside=outside_distance(mask);inside=outside_distance(~mask)
    unknown=(outside<width) if full_texture else ((outside<width)&(inside<width))
    unknown_small=unknown[::8,::8]
    strength=1-smoothstep(outside/width) if full_texture else 1-smoothstep(np.maximum(outside,inside)/width)
    # Inside/outside distance is zero on the opposite side; the maximum is
    # the distance from the boundary in either direction.
    correction=np.zeros(v.shape,'float32');stats=[]
    yy,xx=np.mgrid[:v.shape[0],:v.shape[1]].astype('float32')/8
    for name,band,limit in [('fine',fine,3 if full_texture else 1.6),('medium',medium,3)]:
        rms=np.sqrt(np.maximum(gaussian_filter(band*band,24)-gaussian_filter(band,24)**2,0))
        log=np.log(np.maximum(rms[::8,::8],1e-5));target,solve=harmonic(log,unknown_small,screen=0)
        gain=np.exp(nd.map_coordinates(target-log,[yy,xx],order=1,mode='nearest'))
        gain=np.clip(gain,.75,limit)
        correction+=strength*(gain-1)*band*(coarse+.12)
        stats.append({'band':name,'gain_p99':float(np.percentile(gain[unknown],99)),'solve':solve})
    if reference is not None and not full_texture:
        guide=gaussian_filter(reference.astype('float32')/255,32)
        mismatch=(gaussian_filter(v,32)-guide)/(guide+.15)
        sampled=mismatch[::8,::8];target,solve=harmonic(sampled,unknown_small,screen=0)
        correction+=strength*np.clip(nd.map_coordinates(target-sampled,[yy,xx],order=1,mode='nearest'),-.035,.035)*(guide+.15)
    active=(strength>0)&(np.abs(correction)>1e-10);out=image.copy()
    out[active]=np.rint(np.clip(v[active]+correction[active],0,1)*65535).astype('uint16')
    assert np.array_equal(out[~active],image[~active])
    return out,active,{'changed_pixels':int(np.count_nonzero(out!=image)),'width':width,'full_texture':full_texture,'bands':stats}
