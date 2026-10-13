"""Cache global frequency bands; render bounded repair neighborhoods."""
from lunar_workspace import unlink_array
import gc,time
from pathlib import Path
import numpy as np
from scipy import ndimage as nd
from lunar_filters import gaussian_filter
from frozen_algorithm import smoothstep,outside_distance,harmonic
import lunar_source as ls

def render(t):
    started=time.perf_counter();n=t.n;block=512;halo=192
    current=t.load('current');texture=t.load('texture');quilt=t.load('quilting')
    big=t.load('big_core');small=t.load('small_core');tiny=t.load('tiny_gap');available=t.load('availability')
    tone=t.load('tone_alpha');shade=t.load('shading_delta_small');target=t.load('target_rms')
    gap=t.create('_render_gap','bool');measured=t.create('_render_measured','bool');smalltex=t.create('_render_small_texture')
    for y in range(0,n,1024):
        gap[y:y+1024]=~available[y:y+1024]&~big[y:y+1024]&~tiny[y:y+1024]
        measured[y:y+1024]=small[y:y+1024]&available[y:y+1024]
        smalltex[y:y+1024]=texture[y:y+1024]
    aligned={name:t.load('aligned_'+name) for name in t.source_names}
    validmaps={name:t.load('valid_'+name) for name in t.source_names}
    for y in range(0,n,block):
        ey=min(n,y+block)
        for x in range(0,n,block):
            ex=min(n,x+block)
            seed=np.s_[max(0,y-64):min(n,ey+64),max(0,x-64):min(n,ex+64)]
            if not np.any(measured[seed]):continue
            lo=max(0,y-halo);hi=min(n,ey+halo);left=max(0,x-halo);right=min(n,ex+halo)
            roi=np.s_[lo:hi,left:right];view=np.s_[y-lo:ey-lo,x-left:ex-left]
            yy,xx=np.mgrid[lo:hi,left:right].astype('float32')/8
            desired=nd.map_coordinates(target,[yy,xx],order=1,mode='nearest')
            fused=np.zeros(yy.shape,'float32');total=np.zeros_like(fused)
            for name in t.source_names:
                raw=aligned[name][roi].astype('float32')/65535;valid=validmaps[name][roi].astype('float32')
                den=gaussian_filter(valid,16);mean=gaussian_filter(raw*valid,16)/np.maximum(den,1e-6)
                gy,gx=np.mgrid[lo:hi,left:right]
                detail=ls.detail(raw,valid,16,(-1 if name=='Evening' else 1)*ls.load_gains(t.root).get(name,1.0)*ls.gain_at(gy,gx,n))
                std=np.sqrt(gaussian_filter(detail*detail*valid,32)/np.maximum(gaussian_filter(valid,32),1e-6))
                detail*=np.clip(desired/np.maximum(std,1e-6),.025,1)
                fade=np.clip(nd.distance_transform_edt(valid>0)/32,0,1)
                weight=valid*fade*(1-total);fused+=weight*detail;total+=weight
            fused/=np.maximum(total,1e-6)
            alpha=gaussian_filter((measured[roi]&(total>0)).astype('float32'),16)
            old=np.asarray(texture[roi]);replacement=old+alpha*(fused-old)
            smalltex[y:ey,x:ex]=replacement[view]
        t.log('Measured texture rows',y,ey)
    for y in range(0,n,1024):smalltex[y:y+1024][gap[y:y+1024]]=quilt[y:y+1024][gap[y:y+1024]]
    smalltex.flush();gap.flush();measured.flush()
    original=t.render_bands(texture,'render_original');synthetic=t.render_bands(quilt,'render_quilt');recovered=t.render_bands(smalltex,'render_small')
    output=t.create('output','uint16');changed=t.create('changed','bool');allowed=t.create('allowed','bool');proposal=t.create('_render_proposed')
    coarse=t.load('render_coarse');base=t.load('render_base');changed_count=protected=0
    for y in range(0,n,block):
        ey=min(n,y+block)
        for x in range(0,n,block):
            ex=min(n,x+block);core=np.s_[y:ey,x:ex]
            output[core]=current[core];changed[core]=False;allowed[core]=False;proposal[core]=current[core].astype('float32')/65535
            seed=np.s_[max(0,y-192):min(n,ey+192),max(0,x-192):min(n,ex+192)]
            if not np.any(big[seed]|small[seed]|gap[seed]|tiny[seed]) and not np.any(tone[core]>0):continue
            lo=max(0,y-halo);hi=min(n,ey+halo);left=max(0,x-halo);right=min(n,ex+halo)
            roi=np.s_[lo:hi,left:right];view=np.s_[y-lo:ey-lo,x-left:ex-left]
            b=big[roi];s=small[roi]|gap[roi];db=outside_distance(b);ds=outside_distance(s)
            delta=np.zeros((ey-y,ex-x),'float32')
            b_float=b.astype('float32');s_float=s.astype('float32')
            for old,new_big,new_small,sigma in zip(original,synthetic,recovered,[8,16,32,48]):
                wb=gaussian_filter(b_float,sigma)*(1-smoothstep((db-128)/64))
                ws=gaussian_filter(s_float,sigma)*(1-smoothstep((ds-128)/64))
                if sigma==48:ws*=gaussian_filter(gap[roi].astype('float32'),32)
                delta+=wb[view]*(new_big[core]-old[core])+ws[view]*(new_small[core]-old[core])
            yy,xx=np.mgrid[y:ey,x:ex].astype('float32')/8
            correction=np.clip(nd.map_coordinates(shade,[yy,xx],order=1,mode='nearest'),-.1,.1)*(base[core]+.15)*tone[core]
            values=current[core].astype('float32')/65535+correction+(coarse[core]+.12)*delta
            proposal[core]=values
            active=(tone[core]>0)|(db[view]<192)|(ds[view]<192)|tiny[core]
            pixels=current[core].copy();pixels[active]=np.rint(np.clip(values[active],0,1)*65535).astype('uint16')
            output[core]=pixels;changed[core]=pixels!=current[core];allowed[core]=active
            changed_count+=int(np.count_nonzero(changed[core]));protected+=int(np.count_nonzero((pixels!=current[core])&~active))
        t.log('Rendered rows',y,ey)
    output.flush();proposal.flush()
    if tiny.any():
        labels,count=nd.label(tiny);objects=nd.find_objects(labels)
        for label,box in enumerate(objects,1):
            if box is None:continue
            sy,sx=box;roi=np.s_[max(0,sy.start-2):min(n,sy.stop+2),max(0,sx.start-2):min(n,sx.stop+2)]
            mask=labels[roi]==label;values=np.array(proposal[roi])
            if mask[[0,-1],:].any() or mask[:,[0,-1]].any():
                indices=nd.distance_transform_edt(mask,return_distances=False,return_indices=True);replacement=values[tuple(indices)]
            else:replacement,_=harmonic(values,mask,screen=0)
            output[roi][mask]=np.rint(np.clip(replacement[mask],0,1)*65535).astype('uint16');proposal[roi][mask]=replacement[mask]
            changed[roi]=output[roi]!=current[roi]
        del labels,objects
        changed_count=sum(int(np.count_nonzero(changed[y:y+1024])) for y in range(0,n,1024))
    output.flush();changed.flush();allowed.flush()
    if protected:raise RuntimeError('Renderer changed protected pixels.')
    t.report.update(protected_pixels_changed=protected,changed_percent=100*changed_count/(n*n),no_usable_source_pixels_assigned=int(np.count_nonzero(~available)),no_usable_source_pixels_unassigned=0,measured_detail_core_percent=100*int(measured.sum())/(n*n),block_rows=block,render_halo=halo,filter_workers=1,render_mode='repair-neighborhoods',render_seconds=time.perf_counter()-started)
    for a in original+synthetic+recovered:a._mmap.close()
    del original,synthetic,recovered,smalltex,gap,measured,proposal;gc.collect()
    for p in t.root.iterdir():
        if p.is_file() and any(key in p.name for key in ['render_original_','render_quilt_','render_small_','_render_']):unlink_array(p)
    t.save()
