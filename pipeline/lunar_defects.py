from lunar_workspace import load_array, open_array, unlink_array
"""Whole-image, brightness-conditioned detection without size exclusions."""
from pathlib import Path
import gc
import numpy as np
from scipy import ndimage as nd
from lunar_filters import gaussian_filter

def detect(image,old,support,repair,folder):
    root=Path(folder);root.mkdir(parents=True,exist_ok=True);n=image.shape[0];stride=8
    shape=(n//stride,n//stride)
    rms=open_array(root/'rms.npy',mode='w+',dtype='float32',shape=image.shape)
    brightness=np.empty(shape,'float32')
    for y in range(0,n,1024):
        ey=min(n,y+1024);lo=max(0,y-192);hi=min(n,ey+192);sl=slice(y-lo,ey-lo)
        v=image[lo:hi].astype('float32')/65535;fine=v-gaussian_filter(v,8)
        mean=gaussian_filter(fine,8)
        rms[y:ey]=np.sqrt(np.maximum(gaussian_filter(fine*fine,8)-mean*mean,0))[sl]
        brightness[y//8:ey//8]=gaussian_filter(v,32)[sl][::8,::8]
    rms.flush()
    trusted=((old[::8,::8]>.95*255)&(support[::8,::8]>.98*65535)&(repair[::8,::8]<.025*255))
    samples=np.asarray(rms[::8,::8]);fallback=float(np.median(samples[trusted])) if trusted.any() else .005
    planes=[];centers=(np.arange(20,dtype='float32')+.5)/20
    for center in centers:
        weights=np.maximum(0,1-np.abs(brightness-center)*20)*trusted
        selected=trusted&(np.abs(brightness-center)<.05)
        bin_fallback=float(np.median(samples[selected])) if np.count_nonzero(selected)>=64 else fallback*float(np.clip((center/.45)**2,.1,1))
        den=gaussian_filter(weights,24)
        estimate=gaussian_filter(samples*weights,24)/np.maximum(den,1e-6)
        estimate=np.where(den>.005,estimate,bin_fallback).astype('float32');planes.append(estimate)
    planes=np.stack(planes)
    z=np.clip(brightness*20-.5,0,19);lower=np.floor(z).astype('int32');upper=np.minimum(lower+1,19);fraction=z-lower
    yy,xx=np.indices(shape)
    expected=(planes[lower,yy,xx]*(1-fraction)+planes[upper,yy,xx]*fraction).astype('float32')
    np.save(root/'expected_small.npy',expected);del planes,weights,den,trusted,samples;gc.collect()
    weak=open_array(root/'weak.npy',mode='w+',dtype='bool',shape=image.shape)
    seed=open_array(root/'seed.npy',mode='w+',dtype='bool',shape=image.shape)
    isolated=open_array(root/'isolated.npy',mode='w+',dtype='bool',shape=image.shape)
    for y in range(0,n,1024):
        ey=min(n,y+1024);lo=max(0,y-32);hi=min(n,ey+32);sl=slice(y-lo,ey-lo)
        yy,xx=np.mgrid[lo:hi,:n].astype('float32')/8
        e=nd.map_coordinates(expected,[yy,xx],order=1,mode='nearest');b=nd.map_coordinates(brightness,[yy,xx],order=1,mode='nearest')
        history=(old[lo:hi]<.85*255)|(repair[lo:hi]>.025*255)|(support[lo:hi]<.9*65535)
        candidate=history&(b>.09)&(b<.85)&(rms[lo:hi]<.65*e)&(e>.002)
        severe=candidate&(rms[lo:hi]<(e*.25*np.clip((b/.28)**2,.15,1)))
        weak[y:ey]=candidate[sl];seed[y:ey]=((nd.uniform_filter(severe.astype('float32'),8)>.8)&candidate)[sl]
        isolated[y:ey]=(candidate&(rms[lo:hi]<.05*e))[sl]
    weak.flush();seed.flush();isolated.flush()
    labels,count=nd.label(seed);counts=np.bincount(labels.ravel(),minlength=count+1)
    keep=counts>=64;keep[0]=False;expansion=keep[labels]
    core=nd.binary_propagation(expansion,mask=weak)|seed|isolated
    path=root/'residual_core.npy';np.save(path,core)
    report={'detected_pixels':int(core.sum()),'seed_pixels':int(seed.sum()),'small_seed_pixels_retained':int(np.count_nonzero(seed&~expansion)),
        'policy':'Every seeded component and isolated severe pixel is retained. Size controls expansion, never repair eligibility.'}
    del labels,expansion,core,weak,seed,isolated,rms;gc.collect()
    return path,root/'expected_small.npy',report
