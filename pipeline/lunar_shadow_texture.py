from lunar_filters import gaussian_filter, gaussian_filter1d
"""Conventional photometric restraint for quilted low-information shadows."""
from lunar_workers import ThreadPoolExecutor
import numpy as np
from scipy import ndimage as ndi
from frozen_algorithm import smoothstep, outside_distance, bands

def adjustment(current, texture, quilt, big_core, expected_fine, origin_yx,workers=1):
    v=current.astype('float32')/65535
    with ThreadPoolExecutor(max_workers=workers) as pool:
        old_bands,new_bands=list(pool.map(bands,[texture,quilt]))
        coarse=gaussian_filter(v,64)
    shadow=1-smoothstep((coarse-.22)/.13)
    fine=new_bands[0]+new_bands[1]
    rms=np.sqrt(np.maximum(gaussian_filter(fine*fine,16)-gaussian_filter(fine,16)**2,0))
    y,x=np.mgrid[:v.shape[0],:v.shape[1]].astype('float32')
    y=(y+origin_yx[0])/8;x=(x+origin_yx[1])/8
    expected=ndi.map_coordinates(expected_fine,[y,x],order=1,mode='nearest')
    gain=np.clip(expected/np.maximum((coarse+.12)*rms,1e-6),.025,1)
    cap=1-smoothstep((outside_distance(big_core)-128)/64)
    correction=np.zeros(v.shape,'float32')
    for i,(old,new,sigma) in enumerate(zip(old_bands,new_bands,[8,16,32,48])):
        weight=gaussian_filter(big_core.astype('float32'),sigma)*cap*shadow
        # Match fine granulation to measured texture. Keep existing broad
        # shading rather than transplant broad donor features into shadows.
        change=(gain-1)*new if i<2 else -(new-old)
        correction+=weight*change
    return (coarse+.12)*correction
