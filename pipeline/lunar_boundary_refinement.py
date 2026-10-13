from lunar_filters import gaussian_filter, gaussian_filter1d
"""Local boundary-tone and contrast repair; no AI or geographical coordinates."""
import numpy as np
from scipy import ndimage as nd
def smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)
def refine(image,old_support,correct_tone=True):
    v=image.astype('float32')/65535;h,w=v.shape
    labels,n=nd.label(old_support<.85);objects=nd.find_objects(labels)
    choices=[(s[0].stop-s[0].start,i+1) for i,s in enumerate(objects) if s and s[0].stop-s[0].start>.4*h]
    if not choices:raise ValueError('No tall previous-repair component in this crop.')
    component=labels==max(choices)[1];present=component.any(axis=1)
    left=np.argmax(component,axis=1).astype('float32')
    rows=np.arange(h);left=np.interp(rows,rows[present],left[present]);left=gaussian_filter1d(left,12)
    distance=np.arange(w)[None,:]-left[:,None]
    # An irregular edge corridor: width follows the measured boundary geometry
    # and a smooth deterministic modulation, with a strict finite outer cap.
    width=160+24*np.sin(rows/137)+12*np.sin(rows/53)
    t=(distance+width[:,None])/(2*width[:,None]);inside=np.abs(distance)<width[:,None]
    yfade=smooth(rows/128)*smooth((h-1-rows)/128)*present
    weight=(1-smooth(np.abs(distance)/width[:,None]))*yfade[:,None]
    coarse=gaussian_filter(v,32);slope=gaussian_filter(v,32,order=(0,1))
    lo=left-width;hi=left+width
    def sample(a,x):return nd.map_coordinates(a,[rows,x],order=1,mode='nearest')
    a=sample(coarse,lo);b=sample(coarse,hi);m0=sample(slope,lo);m1=sample(slope,hi)
    # Limit endpoint slopes to prevent interpolation overshoot in a shadow.
    sec=(b-a)/(2*width);limit=3*np.abs(sec)
    m0=np.clip(m0,-limit,limit);m1=np.clip(m1,-limit,limit)
    m0=np.where(m0*sec>=0,m0,0);m1=np.where(m1*sec>=0,m1,0)
    u=np.clip(t,0,1);u2=u*u;u3=u2*u
    target=(2*u3-3*u2+1)*a[:,None]+(u3-2*u2+u)*(2*width*m0)[:,None]+(-2*u3+3*u2)*b[:,None]+(u3-u2)*(2*width*m1)[:,None]
    tone=weight*np.clip(target-coarse,-.035,.035) if correct_tone else np.zeros_like(v)
    # Preserve real local texture, matching its amplitude rather than replacing it.
    fine=v-gaussian_filter(v,8);medium=gaussian_filter(v,8)-coarse
    delta=np.zeros_like(v)
    for band in [fine,medium]:
        rms=np.sqrt(np.maximum(gaussian_filter(band*band,24),1e-8))
        r0=gaussian_filter1d(sample(rms,lo),32);r1=gaussian_filter1d(sample(rms,hi),32)
        desired=(1-u)*r0[:,None]+u*r1[:,None]
        gain=np.clip(desired/rms,.75,1.35)
        delta+=weight*(gain-1)*band
    result=image.copy();active=inside&(yfade[:,None]>0)
    result[active]=np.rint(np.clip(v[active]+tone[active]+delta[active],0,1)*65535).astype('uint16')
    assert np.array_equal(result[~active],image[~active])
    return result,active,{'component':int(max(choices)[1]),'left_edge_median_x':float(np.median(left[present])),
        'corridor_half_width_pixels':[124,196],'maximum_tone_correction_normalized':.035,
        'contrast_gain_limits':[.75,1.35],'protected_pixels_changed':0,
        'changed_percent':float(np.mean(result!=image)*100),
        'tone_correction_abs_p99_active':float(np.percentile(np.abs(tone[active]),99))}
