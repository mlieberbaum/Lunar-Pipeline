from lunar_filters import gaussian_filter, gaussian_filter1d
"""Local brightness-matched multiscale contrast calibration. No AI."""
import numpy as np
from scipy import ndimage as nd
from frozen_algorithm import smoothstep,outside_distance
def calibrate(baseline,result,old,support,repair,mask,strength=1):
    if not np.any(mask):
        return result.copy(),np.zeros(mask.shape,bool),[]
    before=baseline.astype('float32')/65535;v=result.astype('float32')/65535
    coarse=gaussian_filter(before,64);trusted=(old>.95)&(support>.98)&(repair<.025)
    distance=outside_distance(mask);cap=1-smoothstep(distance/96)
    weight=gaussian_filter(mask.astype('float32'),24)*cap
    delta=np.zeros_like(v);reports=[];yy,xx=np.mgrid[:v.shape[0],:v.shape[1]].astype('float32')/8
    for scale in ['fine','medium']:
        b=before-gaussian_filter(before,8) if scale=='fine' else gaussian_filter(before,8)-gaussian_filter(before,32)
        band=v-gaussian_filter(v,8) if scale=='fine' else gaussian_filter(v,8)-gaussian_filter(v,32)
        source_rms=np.sqrt(gaussian_filter(b*b,24));current_rms=np.sqrt(gaussian_filter(band*band,24))
        target=np.zeros(coarse[::8,::8].shape,'float32');norm=np.zeros_like(target)
        brightness=coarse[::8,::8];good=trusted[::8,::8];energy=source_rms[::8,::8]**2
        # Compactly truncated Gaussian neighborhoods and overlapping brightness
        # bins restrict references to nearby terrain at comparable illumination.
        for center in np.arange(.025,1,.05):
            membership=np.maximum(0,1-np.abs(brightness-center)/.075)
            trusted_weight=good*membership
            den=gaussian_filter(trusted_weight,64,truncate=3)
            num=gaussian_filter(energy*trusted_weight,64,truncate=3)
            estimate=np.sqrt(num/np.maximum(den,1e-7))
            contribution=membership*(den>.001)
            target+=estimate*contribution;norm+=contribution
        target/=np.maximum(norm,1e-6)
        expected=nd.map_coordinates(target,[yy,xx],order=1,mode='nearest')
        supported=nd.map_coordinates(norm,[yy,xx],order=1,mode='nearest')>.05
        gain=np.where(supported,np.clip(expected/np.maximum(current_rms,.002),1,1.6 if scale=='fine' else 1.35),1)
        delta+=strength*weight*(gain-1)*band
        reports.append({'band':scale,'median_reference_rms_in_mask':float(np.median(expected[mask])),
            'median_current_rms_in_mask':float(np.median(current_rms[mask])),
            'median_gain_in_mask':float(np.median(gain[mask]))})
    out=result.copy();active=cap>0
    out[active]=np.rint(np.clip(v[active]+delta[active],0,1)*65535).astype('uint16')
    assert np.array_equal(out[~active],result[~active])
    return out,active,reports

