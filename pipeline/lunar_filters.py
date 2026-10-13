"""Single-thread finite Gaussian kernels; preserve separable float rounding."""
import numpy as np
from scipy import ndimage, signal, fft
from functools import lru_cache

@lru_cache(maxsize=128)
def _kernel_spectrum(sigma,truncate,length,axis):
    radius=int(truncate*sigma+.5)
    x=np.arange(-radius,radius+1,dtype=np.float64)
    kernel=np.exp(-.5*x*x/(sigma*sigma));kernel/=kernel.sum()
    shape=[1,1];shape[axis]=len(kernel)
    return fft.rfftn(kernel.reshape(shape),[length],axes=[axis],workers=1)

def gaussian_filter1d(a,sigma,axis=-1,order=0,mode='reflect',truncate=4,**kwargs):
    if sigma<16 or order or mode!='reflect' or a.ndim!=2 or a.dtype not in (np.float32,np.float64) or kwargs:
        return ndimage.gaussian_filter1d(a,sigma,axis=axis,order=order,mode=mode,truncate=truncate,**kwargs)
    axis%=2;radius=int(truncate*sigma+.5)
    length=fft.next_fast_len(a.shape[axis]+4*radius,real=True)
    spectrum=_kernel_spectrum(sigma,truncate,length,axis)
    result=np.empty_like(a)
    with fft.set_workers(1):
        for start in range(0,a.shape[1-axis],64):
            sl=[slice(None),slice(None)];sl[1-axis]=slice(start,start+64)
            v=np.asarray(a[tuple(sl)],dtype=np.float64)
            padding=[(0,0),(0,0)];padding[axis]=(radius,radius)
            v=np.pad(v,padding,mode='symmetric')
            transformed=fft.rfftn(v,[length],axes=[axis],workers=1)
            convolved=fft.irfftn(transformed*spectrum,[length],axes=[axis],workers=1)
            valid=[slice(None),slice(None)];valid[axis]=slice(2*radius,2*radius+a.shape[axis])
            result[tuple(sl)]=convolved[tuple(valid)]
    if np.min(a)>=0:np.maximum(result,0,out=result)
    elif np.max(a)<=0:np.minimum(result,0,out=result)
    return result

def gaussian_filter(a,sigma,order=0,mode='reflect',truncate=4,**kwargs):
    if not np.isscalar(sigma) or sigma<16 or order!=0 or mode!='reflect' or a.ndim!=2 or a.dtype not in (np.float32,np.float64) or kwargs:
        return ndimage.gaussian_filter(a,sigma,order=order,mode=mode,truncate=truncate,**kwargs)
    return gaussian_filter1d(gaussian_filter1d(a,sigma,axis=0,truncate=truncate),sigma,axis=1,truncate=truncate)
