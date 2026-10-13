"""Exact taxicab distance implementation of repeated cross morphology."""
import numpy as np
from scipy import ndimage as nd

def dilate(mask,iterations=1):
    if iterations<1:raise ValueError('Positive iteration count required.')
    if not np.any(mask):return np.zeros(mask.shape,bool)
    return nd.distance_transform_cdt(~mask,metric='taxicab')<=iterations

def erode(mask,iterations=1):
    if iterations<1:raise ValueError('Positive iteration count required.')
    # scipy's default erosion sees false pixels immediately outside the image.
    padded=np.pad(mask,1,constant_values=False)
    distance=nd.distance_transform_cdt(padded,metric='taxicab')
    return distance[1:-1,1:-1]>iterations

def opening(mask,iterations=1):
    if iterations<=2:return nd.binary_opening(mask,iterations=iterations)
    return dilate(erode(mask,iterations),iterations)
def closing(mask,iterations=1):
    if iterations<=2:return nd.binary_closing(mask,iterations=iterations)
    return erode(dilate(mask,iterations),iterations)

def near(mask,radius,block=1024):
    """Exact finite Euclidean-distance threshold without a full float64 map."""
    result=np.zeros(mask.shape,bool);halo=int(np.ceil(radius))
    for start in range(0,mask.shape[0],block):
        end=min(start+block,mask.shape[0]);lo=max(0,start-halo);hi=min(mask.shape[0],end+halo)
        local=mask[lo:hi]
        if not np.any(local):continue
        distance=nd.distance_transform_edt(~local)
        result[start:end]=distance[start-lo:end-lo]<radius
    return result
