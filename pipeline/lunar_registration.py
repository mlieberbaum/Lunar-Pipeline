from lunar_filters import gaussian_filter, gaussian_filter1d
"""Original image-derived registration, recovered from GPT_register.py."""
import numpy as np
from scipy import ndimage as ndi, optimize
from PIL import Image

class RegistrationRejected(ValueError):
    def __init__(self, message, patches):
        super().__init__(message)
        self.patches = patches


def estimate_registration(reference, source, name):
    size=1536
    ref=np.asarray(Image.fromarray(reference).convert('F').resize((size,size),Image.Resampling.LANCZOS))/255
    moving=np.asarray(Image.fromarray(source).convert('F').resize((size,size),Image.Resampling.LANCZOS))/65535
    def edges(a):
        a=gaussian_filter(a,1.3)
        g=np.hypot(ndi.sobel(a,0),ndi.sobel(a,1))
        return np.minimum(g,np.percentile(g,95))
    ref=edges(ref);moving=edges(moving);patches=[]
    # Predetermined sampling windows, not predetermined shifts.
    for y,x in [(100,100),(100,900),(650,400),(900,900)]:
        sl=(slice(y,y+400),slice(x,x+400));target=ref[sl]
        def loss(shift):
            src=ndi.shift(moving,shift,order=1,mode='nearest')[sl]
            return -float(np.corrcoef(target.ravel(),src.ravel())[0,1])
        r=optimize.minimize(loss,[0,0],method='Powell',bounds=[(-8,8),(-8,8)],options={'xtol':.02,'ftol':1e-5})
        if not np.all(np.isfinite(r.x)) or not np.isfinite(r.fun):
            raise RegistrationRejected(name+': registration did not produce finite measurements.', patches)
        patches.append({'patch_yx':[y,x],'shift_yx_at_1536':r.x.tolist(),'edge_correlation':-r.fun})
    confidence = float(np.median([p['edge_correlation'] for p in patches]))
    if confidence < 0.4:
        raise RegistrationRejected(f'{name}: median registration correlation {confidence:.4f} is below 0.4.', patches)
    A=np.array([[1,(p['patch_yx'][0]+200)*8,(p['patch_yx'][1]+200)*8] for p in patches])
    shifts=np.array([p['shift_yx_at_1536'] for p in patches])*8
    transform=np.linalg.lstsq(A,shifts,rcond=None)[0] if name=='Seamless' else np.array([np.median(shifts,axis=0),[0,0],[0,0]])
    return transform,patches
