"""Build feathered source masks from detected strip regions."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class MaskPolicy:
    anchor_grid: float = 80.0
    guard: float = 170.0
    width_grid: float = 10.0
    feather: float = 160.0
    edge_grid: float = 80.0
    edge_feather: float = 140.0
    edge_zone_fraction: float = 0.025

def masks_from_regions(regions,size=12288,policy=MaskPolicy()):
    """Anchor interior guards at quantized leading edges; grow with strip span."""
    factor=size/12288;masks=[]
    for r in regions:
        low=r['start_fraction']*size;high=r['end_fraction']*size
        axis=r['axis']
        if r['start_fraction']<policy.edge_zone_fraction:
            core=np.ceil(high/(policy.edge_grid*factor))*policy.edge_grid*factor
            masks.append({'axis':axis,'kind':'low_edge','core':float(core),
                          'feather':policy.edge_feather*factor})
        elif r['end_fraction']>1-policy.edge_zone_fraction:
            core=np.ceil((size-low)/(policy.edge_grid*factor))*policy.edge_grid*factor
            masks.append({'axis':axis,'kind':'high_edge','core':float(core),
                          'feather':policy.edge_feather*factor})
        else:
            grid=policy.anchor_grid*factor
            center=np.ceil(low/grid)*grid
            width_grid=policy.width_grid*factor
            halfwidth=policy.guard*factor+np.rint((high-low)/2/width_grid)*width_grid
            masks.append({'axis':axis,'kind':'interior','center':float(center),
                          'halfwidth':float(halfwidth),'feather':policy.feather*factor})
    return masks

def apply_masks(valid,masks,y,x,size):
    for mask in masks:
        coordinate=x if mask['axis']=='x' else y
        if mask['kind']=='interior':
            factor=np.clip((np.abs(coordinate-mask['center'])-mask['halfwidth'])/mask['feather'],0,1)
        elif mask['kind']=='low_edge':
            factor=np.clip((coordinate-mask['core'])/mask['feather'],0,1)
        else:
            factor=np.clip((size-coordinate-mask['core'])/mask['feather'],0,1)
        valid*=factor
    return valid
