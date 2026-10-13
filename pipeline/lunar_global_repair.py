from lunar_workspace import load_array, open_array, unlink_array
"""Repair every detected residual using real detail or conventional quilting."""
from pathlib import Path
import gc,json,time
import numpy as np
from scipy import ndimage as nd
from lunar_filters import gaussian_filter
from frozen_algorithm import harmonic,smoothstep,outside_distance
from lunar_defects import detect
import lunar_source as ls

def area_mean(image,scale):
    n=image.shape[0];out=np.empty((n//8,n//8),'float32')
    for y in range(0,n,1024):
        v=image[y:y+1024].astype('float32')/scale
        out[y//8:(y+len(v))//8]=v.reshape(len(v)//8,8,n//8,8).mean(axis=(1,3))
    return out

def run(image_path,old_path,support_path,repair_path,folder,reference_path=None,source_folder=None,core_path=None,expected_path=None):
    from full_tile_trial import Trial
    started=time.perf_counter();root=Path(folder);root.mkdir(parents=True,exist_ok=True)
    image=load_array(image_path,mmap_mode='r');n=image.shape[0]
    t=Trial({'work_folder':str(root),'size':n})
    for name,path in [('current',image_path),('old_support',old_path),('support',support_path),('repair_strength',repair_path)]:t.cache_png(name,path)
    old=t.load('old_support');support=t.load('support');repair=t.load('repair_strength')
    if core_path is None:core_path,expected_path,report=detect(image,old,support,repair,root/'detect')
    else:report={'detected_pixels':int(load_array(core_path,mmap_mode='r').sum())}
    core=load_array(core_path,mmap_mode='r');expected=load_array(expected_path,mmap_mode='r')
    out=t.create('result','uint16');out[:]=image;allowed=t.create('allowed','bool');allowed[:]=False
    labels,count=nd.label(core);counts=np.bincount(labels.ravel(),minlength=count+1)
    ids=counts<=64;ids[0]=False
    report['tiny_pixels']=int(np.count_nonzero(ids[labels]))
    mask=np.array(core,dtype=bool)
    del labels,counts,ids;gc.collect()
    if not mask.any():
        out.flush();allowed.flush();report.update(changed_pixels=int(np.count_nonzero(out!=image)),protected_pixels_changed=0,unassigned_pixels=0,seconds=time.perf_counter()-started)
        (root/'report.json').write_text(json.dumps(report,indent=2));return root/'result.npy',root/'allowed.npy',report
    history=(old<.85*255)|(repair>.025*255)|(support<.9*65535)
    mask|=nd.binary_dilation(mask,iterations=24)&history;del history
    coarse=t.gaussian_rows(image,64,'coarse',scale=65535);texture=t.create('texture');original=t.create('original_texture')
    for y in range(0,n,1024):texture[y:y+1024]=(image[y:y+1024].astype('float32')/65535-coarse[y:y+1024])/(coarse[y:y+1024]+.12)
    original[:]=texture;measured=t.create('measured','bool');measured[:]=False
    source_names=[]
    if source_folder:
        source_names=[name for name in ['Seamless','Morning','Evening'] if (Path(source_folder)/('aligned_'+name+'.npy')).exists()]
    aligned={name:load_array(Path(source_folder)/('aligned_'+name+'.npy'),mmap_mode='r') for name in source_names}
    validmaps={name:load_array(Path(source_folder)/('valid_'+name+'.npy'),mmap_mode='r') for name in source_names}
    for y in range(0,n,512):
        ey=min(n,y+512)
        for x in range(0,n,512):
            ex=min(n,x+512);core_roi=np.s_[y:ey,x:ex]
            if not mask[core_roi].any():continue
            lo=max(0,y-192);hi=min(n,ey+192);left=max(0,x-192);right=min(n,ex+192)
            roi=np.s_[lo:hi,left:right];view=np.s_[y-lo:ey-lo,x-left:ex-left]
            yy,xx=np.mgrid[lo:hi,left:right].astype('float32')/8
            desired=nd.map_coordinates(expected,[yy,xx],order=1,mode='nearest')/(coarse[roi]+.12)
            chosen=np.zeros(desired.shape,bool);new=np.array(texture[roi])
            for name in source_names:
                raw=aligned[name][roi].astype('float32')/65535
                valid=validmaps[name][roi]
                den=gaussian_filter(valid.astype('float32'),16);mean=gaussian_filter(raw*valid,16)/np.maximum(den,1e-6)
                gy,gx=np.mgrid[lo:hi,left:right]
                detail=ls.detail(raw,valid,16,(-1 if name=='Evening' else 1)*ls.load_gains(source_folder).get(name,1.0)*ls.gain_at(gy,gx,n))
                fine=detail-gaussian_filter(detail,8);std=np.sqrt(np.maximum(gaussian_filter(fine*fine,8)-gaussian_filter(fine,8)**2,0))
                good=valid&(std>.65*desired)&~chosen
                detail*=np.clip(desired/np.maximum(std,1e-6),.5,2)
                new[good]=detail[good];chosen|=good
            m=mask[core_roi]&chosen[view];texture[core_roi][m]=new[view][m];measured[core_roi]=m
    texture.flush();measured.flush();synthesized=mask&~measured
    t.store('quilt_core',synthesized);t.store('availability',~synthesized);t.store('tiny_gap',np.zeros(image.shape,bool))
    t.store('target_rms',expected/(coarse[::8,::8]+.12))
    if synthesized.any():t.quilt();canvas=t.load('quilting');assigned=t.load('assigned')
    else:canvas=texture;assigned=np.zeros(image.shape,bool)
    # Fit local tone through the complete defect and its rim. Broad shading
    # follows the reference; no synthesized structure larger than 64 pixels.
    small=gaussian_filter(area_mean(image,65535),2)
    guide=gaussian_filter(area_mean(load_array(reference_path,mmap_mode='r'),255),2) if reference_path else small.copy()
    mismatch=(small-guide)/(guide+.15)
    reduced=mask.reshape(n//8,8,n//8,8).any(axis=(1,3))
    solved,solve=harmonic(mismatch,nd.binary_dilation(reduced,iterations=16),screen=0)
    tone_delta=np.clip(solved-mismatch,-.10,.10);del solved,small,mismatch;gc.collect()
    for y in range(0,n,512):
        ey=min(n,y+512)
        for x in range(0,n,512):
            ex=min(n,x+512);core_roi=np.s_[y:ey,x:ex]
            seed=np.s_[max(0,y-128):min(n,ey+128),max(0,x-128):min(n,ex+128)]
            if not mask[seed].any():continue
            lo=max(0,y-192);hi=min(n,ey+192);left=max(0,x-192);right=min(n,ex+192)
            roi=np.s_[lo:hi,left:right];view=np.s_[y-lo:ey-lo,x-left:ex-left]
            m=mask[roi];distance=outside_distance(m);oldtex=np.array(original[roi]);newtex=np.array(texture[roi])
            syn=synthesized[roi];newtex[syn]=canvas[roi][syn]
            old8=gaussian_filter(oldtex,8);new8=gaussian_filter(newtex,8)
            fine_old=oldtex-old8;fine_new=newtex-new8
            med_old=old8-gaussian_filter(oldtex,32);med_new=new8-gaussian_filter(newtex,32)
            yy,xx=np.mgrid[lo:hi,left:right].astype('float32')/8
            desired=nd.map_coordinates(expected,[yy,xx],order=1,mode='nearest')/(coarse[roi]+.12)
            fine_std=np.sqrt(np.maximum(gaussian_filter(fine_new*fine_new,8)-gaussian_filter(fine_new,8)**2,0))
            med_std=np.sqrt(np.maximum(gaussian_filter(med_new*med_new,16)-gaussian_filter(med_new,16)**2,0))
            fine_new*=np.where(syn,np.clip(desired/np.maximum(fine_std,1e-6),.5,3),1)
            med_new*=np.where(syn,np.clip(2*desired/np.maximum(med_std,1e-6),.5,3),1)
            cap=1-smoothstep(distance/128)
            fine_weight=gaussian_filter(m.astype('float32'),12)*cap
            med_weight=gaussian_filter(m.astype('float32'),24)*cap
            delta=fine_weight*(fine_new-fine_old)+med_weight*(med_new-med_old)
            cy,cx=np.mgrid[y:ey,x:ex].astype('float32')/8-.4375
            tone_c=nd.map_coordinates(tone_delta,[cy,cx],order=1,mode='nearest')*(nd.map_coordinates(guide,[cy,cx],order=1,mode='nearest')+.15)
            tone_c*=1-smoothstep(distance[view]/48)
            proposal=image[core_roi].astype('float32')/65535+tone_c+(coarse[core_roi]+.12)*delta[view]
            active=distance[view]<128
            out[core_roi][active]=np.rint(np.clip(proposal[active],0,1)*65535).astype('uint16');allowed[core_roi]|=active
        t.log('Residual repair rows',y,ey)
    missing=int(np.count_nonzero(core&~(measured|assigned)))
    if missing:raise RuntimeError(f'{missing} detected residual pixels unassigned.')
    out.flush();allowed.flush();changed=protected=0
    for y in range(0,n,1024):
        different=out[y:y+1024]!=image[y:y+1024];changed+=int(different.sum());protected+=int(np.count_nonzero(different&~allowed[y:y+1024]))
    if protected:raise RuntimeError('Residual repair changed protected pixels.')
    report.update(changed_pixels=changed,protected_pixels_changed=protected,unassigned_pixels=missing,measured_pixels=int(measured.sum()),exemplar_pixels=int(synthesized.sum()),quilting=t.report.get('quilting',{}),tone_solve=solve,seconds=time.perf_counter()-started)
    (root/'report.json').write_text(json.dumps(report,indent=2))
    return root/'result.npy',root/'allowed.npy',report
