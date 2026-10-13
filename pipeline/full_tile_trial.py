from lunar_workspace import load_array, open_array, unlink_array
from lunar_morphology import opening, closing, near
from lunar_filters import gaussian_filter, gaussian_filter1d
"""Deterministic, memory-bounded whole-tile repair trial. No generative AI.

Inputs: aligned-stage candidate, original sources/registration and support maps.
Global masks, donor pool, raster quilting and screened shading solve are fixed
before streaming the final pixels. Processing has no manual geographic repairs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gc,json,hashlib,time
from lunar_workers import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from frozen_algorithm import smoothstep,large,outside_distance,harmonic,mincut,bands
from repair_lunar_detail import coordinates,refined_masks,local_agreement
from lunar_mask_policy import apply_masks,masks_from_regions
from lunar_io import sources_reused,resize_source
Image.MAX_IMAGE_PIXELS=None

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

class Trial:
    def __init__(self,config):
        self.config=config;self.root=Path(config['work_folder']);self.root.mkdir(parents=True,exist_ok=True)
        self.source_names=['Seamless','Morning','Evening']
        self.n=int(config.get('size',12288));self.shape=(self.n,self.n);self.seed=20261007
        self.report={'algorithm':'lunar-refinement-4','size':self.n,'seed':self.seed,
          'large_component_minimum':262144,'small_artifact_minimum':256,'gap_minimum':0,
          'donor_policy':'One sorted whole-tile 128px donor pool; grid 64; global quilt raster step 96. Donors restricted to .6–1.6 times local trusted texture RMS where possible.',
          'validity_definition':'Registered, in bounds, intensity .01 to .995, outside zero seam cores; Evening requires positive local agreement. Processing usability, not scientific IMG no-data flags.',
          'limitations':['Trial on two tiles, not validated for arbitrary tiles or neighboring geographic boundaries.','Synthetic exemplar placements are invented texture, not recovered lunar geography.']}
    def log(self,*args):print(*args,flush=True)
    def create(self,name,dtype='float32',shape=None):
        return open_array(self.root/(name+'.npy'),mode='w+',dtype=dtype,shape=shape or self.shape)
    def load(self,name,mode='r'):
        return load_array(self.root/(name+'.npy'),mmap_mode=mode)
    def store(self,name,a):
        dst=self.create(name,a.dtype,a.shape);dst[:]=a;dst.flush();del dst;gc.collect()
    def stripes(self,halo=0,block=384):
        for start in range(0,self.n,block):
            end=min(start+block,self.n);lo=max(0,start-halo);hi=min(self.n,end+halo)
            yield start,end,lo,hi,slice(start-lo,end-lo)
    def cache_png(self,name,path,resize=False,reference=False):
        if (self.root/(name+'.npy')).exists():return self.load(name)
        if Path(path).suffix.lower()=='.npy' and not resize and not reference:
            array=load_array(path,mmap_mode='r',allow_pickle=False)
            if array.shape!=self.shape:raise ValueError(f'{name}: wrong dimensions {array.shape}')
            del array
            try:os.link(path,self.root/(name+'.npy'))
            except OSError:
                import shutil
                shutil.copyfile(path,self.root/(name+'.npy'))
            return self.load(name)
        if Path(path).suffix.lower()=='.img' or (sources_reused() and name in ['Seamless','Morning','Evening']):
            from lunar_texture_auto import load_source
            a=load_source(path,name)
            if resize:
                a=resize_source(a,self.n)
            if a.shape!=self.shape:raise ValueError(f'{name}: wrong dimensions {a.shape}')
            self.store(name,a)
            return self.load(name)
        with Image.open(path) as im:
            if reference:im=im.convert('L').resize((self.n,self.n),Image.Resampling.BICUBIC)
            elif resize:im=im.convert('F').resize((self.n,self.n),Image.Resampling.BICUBIC)
            a=np.asarray(im)
            if resize:a=np.clip(np.rint(a),0,65535).astype('uint16')
            if a.shape!=self.shape:raise ValueError(f'{name}: wrong dimensions {a.shape}')
            self.store(name,a)
        return self.load(name)
    def prepare(self):
        cfg=self.config
        for name in ['current','old_support','support','repair_strength']:
            self.cache_png(name,cfg[name]);self.log('Cached',name)
        self.cache_png('reference',cfg['orbiter'],reference=True)
        original=json.loads(Path(cfg['original_diagnostics']).read_text())['sources']
        recovered=json.loads(Path(cfg['repair_diagnostics']).read_text())['evening_recovery']
        transforms={name:np.array(original[name]['transform_yx']) for name in ['Seamless','Morning'] if original[name].get('enabled',True)}
        if 'transform_yx' in recovered:transforms['Evening']=np.array(recovered['transform_yx'])
        self.source_names=list(transforms)
        sources={name:self.cache_png(name,cfg[name.lower()],resize=name=='Seamless') for name in self.source_names}
        confidence=local_agreement(sources['Seamless'],sources['Evening'],transforms['Seamless'],transforms['Evening']) if 'Evening' in sources else None
        availability=self.create('availability','bool');availability[:]=False
        for name in self.source_names:
            aligned=self.create('aligned_'+name,'uint16');validmap=self.create('valid_'+name,'bool')
            masks=(refined_masks(original[name]['strips']) if name=='Seamless' else masks_from_regions(original[name]['strips']) if name=='Evening' else original[name]['masks'])
            for start,end,lo,hi,sl in self.stripes(block=512):
                yy,xx=np.mgrid[start:end,:self.n].astype(np.float32)
                sy,sx=coordinates(transforms[name],yy,xx)
                values=ndi.map_coordinates(sources[name],[sy,sx],order=1,mode='nearest')
                v=values.astype('float32')/65535
                valid=((sy>=0)&(sx>=0)&(sy<self.n-1)&(sx<self.n-1)&(v>.01)&(v<.995))
                valid=apply_masks(valid.astype('float32'),masks,sy,sx,self.n)>0
                if name=='Evening':valid&=ndi.map_coordinates(confidence,[yy/8,xx/8],order=1,mode='nearest')>0
                aligned[start:end]=values;validmap[start:end]=valid;availability[start:end]|=valid
            aligned.flush();validmap.flush();del aligned,validmap
            self.log('Aligned',name)
        availability.flush();del availability,sources;gc.collect()
        self.report['transforms_yx']={k:v.tolist() for k,v in transforms.items()}
        current=self.load('current');reference=self.load('reference');old=self.load('old_support');support=self.load('support')
        texture=self.create('texture');fine_std=self.create('fine_std')
        coarse=self.gaussian_rows(current,64,'render_coarse',scale=65535)
        fine=self.gaussian_rows(current,8,'_analysis_fine',scale=65535)
        for start,end,*_ in self.stripes(block=1024):
            v=current[start:end].astype('float32')/65535
            texture[start:end]=(v-coarse[start:end])/(coarse[start:end]+.12)
            fine[start:end]=v-fine[start:end]
        texture.flush();fine.flush();del coarse
        base=self.gaussian_rows(reference,32,'render_base',scale=255)
        low=self.create('_analysis_low')
        for start,end,*_ in self.stripes(block=1024):
            v=current[start:end].astype('float32')/65535
            low[start:end]=(v-base[start:end])/(base[start:end]+.15)
        low.flush();del base
        low_small=self.gaussian_rows(low,64,'low_small',step=8)
        self.discard_array(low)
        mean_fine=self.gaussian_rows(fine,16,'_analysis_fine_mean')
        energy=self.gaussian_rows(fine,16,'_analysis_fine_energy',squared=True)
        for start,end,*_ in self.stripes(block=1024):
            fine_std[start:end]=np.sqrt(np.maximum(energy[start:end]-mean_fine[start:end]**2,0))
            self.log('Analysis rows',start,end)
        fine_std.flush()
        self.discard_array(fine);self.discard_array(mean_fine);self.discard_array(energy)
        trusted=((old[::8,::8]>.95*255)&(support[::8,::8]>.95*65535)).astype('float32')
        sample=texture[::8,::8]
        expected=np.sqrt(gaussian_filter(sample*sample*trusted,8)/np.maximum(gaussian_filter(trusted,8),1e-6))
        weak_expected=gaussian_filter(fine_std[::8,::8]*trusted,24)/np.maximum(gaussian_filter(trusted,24),1e-6)
        self.store('target_rms',expected);self.store('weak_expected',weak_expected)
        del texture,fine_std,current,reference;gc.collect()
    def gaussian_rows(self,source,sigma,name,scale=1,squared=False,step=1):
        # scipy.gaussian_filter applies axis 0 then axis 1, with float32
        # rounding after each pass. Retain exactly that order and precision.
        radius=int(4*sigma+.5)
        destination=self.create(name,shape=(self.n//step,self.n//step))
        for start,end,lo,hi,sl in self.stripes(halo=radius,block=1024):
            values=np.asarray(source[lo:hi],dtype='float32')
            if scale!=1:values=values/scale
            if squared:values=values*values
            vertical=gaussian_filter1d(values,sigma,axis=0)
            result=gaussian_filter1d(vertical[sl],sigma,axis=1)
            destination[start//step:end//step]=result[::step,::step]
        destination.flush()
        return destination
    def discard_array(self,array):
        path=Path(array.filename);array._mmap.close();unlink_array(path)
    def masks(self):
        old=self.load('old_support');support=self.load('support');strength=self.load('repair_strength')
        hand=opening(strength>5,iterations=2)
        # Size routing is derived from complete connected repair components.
        large_hand=large(hand,262144);self.store('large_hand',large_hand)
        del hand
        old_bad=closing(old<.85*255,iterations=8)
        large_old=large(old_bad,262144)
        gap=~self.load('availability');self.report['no_usable_source_pixels']=int(np.count_nonzero(gap))
        labels,n=ndi.label(gap);counts=np.bincount(labels.ravel(),minlength=n+1)
        ids=counts<=64;ids[0]=False;tiny=ids[labels];del labels,counts,ids
        self.store('tiny_gap',tiny)
        big_remaining=large(closing(support<.85*65535,iterations=8),262144)
        large_gap=large(gap,262144)
        large_anchor=large_old|large_hand|big_remaining|large_gap
        ribbon=(near(large_hand,32))&(near(~large_hand,32))
        big_core=closing(big_remaining|ribbon|large_gap,iterations=16)
        del large_gap,large_old,large_hand,big_remaining,ribbon
        self.store('big_core',big_core)
        small_hand=large(opening(strength>5,iterations=1),256)
        remaining=large(closing(support<.9*65535,iterations=2),256)
        fine_std=self.load('fine_std');expected=self.load('weak_expected');weak=self.create('weak_raw','bool')
        for start,end,lo,hi,sl in self.stripes():
            yy,xx=np.mgrid[start:end,:self.n].astype(np.float32)/8
            e=ndi.map_coordinates(expected,[yy,xx],order=1,mode='nearest')
            weak[start:end]=(old_bad[start:end]|small_hand[start:end])&(fine_std[start:end]<.65*e)
        weak.flush();weak_small=large(closing(weak,iterations=2),256);del weak,fine_std
        small_ribbon=(near(small_hand,16))&(near(~small_hand,16))
        small_core=remaining|weak_small|small_ribbon|gap
        del remaining,weak_small,small_ribbon
        # Keep the complete accepted large-repair footprint on its own branch.
        big_zone=near(large_anchor,192)
        small_core&=~big_zone
        del big_zone
        all_core=big_core|small_core|gap
        self.store('small_core',small_core);self.store('all_core',all_core)
        del all_core
        self.store('quilt_core',big_core|(gap&~tiny))
        big_tone=near(large_anchor,96)
        del large_anchor
        small_tone=near(large(old_bad,256)|small_hand|small_core|gap,48)
        del old_bad,small_hand
        tone_region=big_tone|small_tone
        tone_region[[0,-1],:]=False;tone_region[:,[0,-1]]=False
        alpha=ndi.distance_transform_edt(tone_region).astype('float32')
        alpha/=64;alpha[~big_tone]*=2
        np.clip(alpha,0,1,out=alpha);factor=3-2*alpha;alpha*=alpha;alpha*=factor;del factor
        del big_tone,small_tone
        self.store('tone_alpha',alpha)
        low=self.load('low_small');extended,solve=harmonic(np.array(low),tone_region[::8,::8])
        self.store('shading_delta_small',extended-low);self.report['shading_solve']=solve
        self.report['large_core_percent']=100*float(np.mean(big_core));self.report['small_core_percent']=100*float(np.mean(small_core))
        self.report['tiny_gap_pixels']=int(np.count_nonzero(tiny))
        self.log('Masks:',self.report['large_core_percent'],self.report['small_core_percent'],'percent; gaps',self.report['no_usable_source_pixels'])
        del alpha,tone_region,gap,tiny,small_core,big_core;gc.collect()
    def quilt(self):
        texture=self.load('texture');mask=self.load('quilt_core');old=self.load('old_support');support=self.load('support');current=self.load('current')
        candidates=[];patch=128;overlap=32;step=96
        for y in range(0,self.n-patch+1,64):
            for x in range(0,self.n-patch+1,64):
                sl=np.s_[y:y+patch,x:x+patch]
                good=(old[sl]>.95*255)&(support[sl]>.95*65535)&(current[sl]>.035*65535)&(current[sl]<.96*65535)&~mask[sl]
                if np.mean(good)>.985:
                    sd=float(np.std(texture[sl]))
                    if .001<sd<.15:candidates.append((y,x,sd))
        if len(candidates)<8:raise ValueError('Insufficient whole-tile donor patches.')
        donors=np.asarray(candidates);self.store('donor_pool',donors)
        canvas=self.create('quilting');canvas[:]=texture
        assigned=self.create('assigned','bool');assigned[:]=False
        expected=self.load('target_rms');rng=np.random.Generator(np.random.PCG64(self.seed));placements=0;used=set()
        for y in range(0,self.n,step):
            for x in range(0,self.n,step):
                ph,pw=min(patch,self.n-y),min(patch,self.n-x);sl=np.s_[y:y+ph,x:x+pw]
                hole=mask[sl]
                if not np.any(hole):continue
                target=canvas[sl];done=assigned[sl];known=done|~hole
                desired=float(expected[min((y+ph//2)//8,expected.shape[0]-1),min((x+pw//2)//8,expected.shape[1]-1)])
                pool=np.flatnonzero((donors[:,2]>=.6*desired)&(donors[:,2]<=1.6*desired))
                if len(pool)<32:pool=np.argsort(np.abs(donors[:,2]-desired))[:min(128,len(donors))]
                indices=rng.choice(pool,min(32,len(pool)),replace=False);best=None
                for index in indices:
                    sy,sx,std=candidates[int(index)];donor=texture[sy:sy+ph,sx:sx+pw]
                    score=float(np.mean((donor[known]-target[known])**2)) if np.any(known) else (std-desired)**2
                    if best is None or score<best[0]:best=(score,int(index),donor)
                _,index,donor=best;replace=np.ones(hole.shape,bool)
                if x>0:
                    width=min(overlap,pw);replace[:,:width]&=mincut((donor[:,:width]-target[:,:width])**2)
                if y>0:
                    height=min(overlap,ph);replace[:height]&=mincut(((donor[:height]-target[:height])**2).T).T
                replace=(replace|~done)&hole;target[replace]=donor[replace];done[replace]=True
                placements+=1;used.add(index)
            if y%960==0:self.log('Quilt rows',y,'patches',placements)
        canvas.flush();assigned.flush()
        missing=0
        for start,end,*_ in self.stripes():missing+=int(np.count_nonzero(mask[start:end]&~assigned[start:end]))
        if missing:raise RuntimeError(f'{missing} quilt pixels unassigned.')
        availability=self.load('availability');tiny=self.load('tiny_gap')
        for start,end,*_ in self.stripes():
            if np.any((~availability[start:end])&~assigned[start:end]&~tiny[start:end]):raise RuntimeError('Gap missing its explicit repair route.')
        self.report['quilting']={'donor_count':len(candidates),'placements':placements,'distinct_donors':len(used),'unassigned_pixels':missing,'pool_sha256':sha(self.root/'donor_pool.npy')}
        self.log('Quilting complete',self.report['quilting'])
    def render_bands(self,source,prefix):
        previous=source;temporary=False;result=[]
        for i,sigma in enumerate([2,8,24,64]):
            blurred=self.gaussian_rows(source,sigma,prefix+'_blur_'+str(i))
            band=self.create(prefix+'_band_'+str(i))
            for start,end,*_ in self.stripes(block=1024):band[start:end]=previous[start:end]-blurred[start:end]
            band.flush();result.append(band)
            if temporary:self.discard_array(previous)
            previous=blurred;temporary=True
        self.discard_array(previous)
        return result
    def render(self):
        mode=self.config.get('render_mode','auto')
        if mode=='auto':
            big=self.load('big_core');small=self.load('small_core');available=self.load('availability');tiny=self.load('tiny_gap');tone=self.load('tone_alpha');volume=0
            for start,end,lo,hi,sl in self.stripes(halo=512,block=1024):
                seed=np.s_[max(0,start-192):min(self.n,end+192),:self.n]
                columns=np.any(big[seed]|small[seed]|~available[seed]|tiny[seed],axis=0)|np.any(tone[start:end]>0,axis=0)
                indices=np.flatnonzero(columns)
                if len(indices):volume+=(hi-lo)*(min(self.n,int(indices[-1])+705)-max(0,int(indices[0])-704))
            mode='dense' if volume>1.25*self.n*self.n else 'local'
        if mode=='dense':
            from lunar_render import render
            render(self)
        else:self.render_legacy()
    def render_legacy(self):
        current=self.load('current');reference=self.load('reference');texture=self.load('texture');quilt=self.load('quilting')
        big=self.load('big_core');small=self.load('small_core');all_core=self.load('all_core');tiny=self.load('tiny_gap');available=self.load('availability')
        tone_alpha=self.load('tone_alpha');shade=self.load('shading_delta_small');target_rms=self.load('target_rms')
        aligned={name:self.load('aligned_'+name) for name in self.source_names}
        validity={name:self.load('valid_'+name) for name in aligned}
        resume=int(self.config.get('render_start_row',0))
        if resume:
            output=self.load('output','r+');changed_map=self.load('changed','r+');allowed_map=self.load('allowed','r+')
        else:
            output=self.create('output','uint16');changed_map=self.create('changed','bool');allowed_map=self.create('allowed','bool')
        protected=changed=covered=measured_count=0
        for start in range(0,resume,384):
            end=min(start+384,resume);sl=slice(start,end)
            protected+=int(np.count_nonzero((output[sl]!=current[sl])&~allowed_map[sl]))
            changed+=int(np.count_nonzero(output[sl]!=current[sl]))
            covered+=int(np.count_nonzero(~available[sl]))
            measured_count+=int(np.count_nonzero(small[sl]&available[sl]))
        render_block=int(self.config.get('render_block_rows',384))
        workers=max(1,min(3,int(self.config.get('filter_workers',1))))
        pool=ThreadPoolExecutor(max_workers=workers)
        for start,end,lo,hi,sl in self.stripes(halo=512,block=render_block):
            if start<resume:continue
            # The allowed footprint cannot extend more than 192px from these
            # masks. Copy protected columns/rows directly, retaining their uint16.
            seed=np.s_[max(0,start-192):min(self.n,end+192),:self.n]
            columns=np.any(big[seed]|small[seed]|~available[seed]|tiny[seed],axis=0)
            columns|=np.any(tone_alpha[start:end]>0,axis=0)
            output[start:end]=current[start:end]
            changed_map[start:end]=False;allowed_map[start:end]=False
            indices=np.flatnonzero(columns)
            if not len(indices):
                self.log('Copied protected rows',start,end)
                continue
            core_left=max(0,int(indices[0])-192);core_right=min(self.n,int(indices[-1])+193)
            # Keep the original row boundaries. 512px in x exceeds the finite
            # support of every composite Gaussian used by this renderer.
            left=max(0,core_left-512);right=min(self.n,core_right+512)
            v=current[lo:hi,left:right].astype('float32')/65535;oldtex=np.array(texture[lo:hi,left:right]);synthetic=np.array(quilt[lo:hi,left:right])
            yy,xx=np.mgrid[lo:hi,left:right].astype(np.float32)/8
            desired=ndi.map_coordinates(target_rms,[yy,xx],order=1,mode='nearest')
            fused=np.zeros(v.shape,np.float32);sum_weight=np.zeros_like(fused)
            def source_fields(name):
                raw=aligned[name][lo:hi,left:right].astype('float32')/65535;valid=validity[name][lo:hi,left:right].astype('float32')
                den=gaussian_filter(valid,16);mean=gaussian_filter(raw*valid,16)/np.maximum(den,1e-6)
                detail=np.clip((raw-mean)/(mean+.15),-.6,.6)
                if name=='Evening':detail=-detail
                source_std=np.sqrt(gaussian_filter(detail*detail*valid,32)/np.maximum(gaussian_filter(valid,32),1e-6))
                detail*=np.clip(desired/np.maximum(source_std,1e-6),.025,1)
                fade=np.clip(ndi.distance_transform_edt(valid>0)/32,0,1)
                return valid,detail,fade
            for valid,detail,fade in pool.map(source_fields,aligned):
                weight=valid*fade*(1-sum_weight);fused+=weight*detail;sum_weight+=weight
            del valid,detail,fade,weight
            fused/=np.maximum(sum_weight,1e-6)
            measured_region=small[lo:hi,left:right]&available[lo:hi,left:right]&(sum_weight>0)
            source_alpha=gaussian_filter(measured_region.astype('float32'),16)
            # Small supported defects receive measured source detail; only gaps
            # and the explicit large branch receive synthesized exemplar detail.
            smalltex=oldtex+source_alpha*(fused-oldtex)
            gap=(~available[lo:hi,left:right])&~big[lo:hi,left:right]&~tiny[lo:hi,left:right]
            smalltex[gap]=synthetic[gap]
            original_bands,big_bands,small_bands=pool.map(bands,[oldtex,synthetic,smalltex])
            delta=np.zeros(v.shape,np.float32)
            distance_big=outside_distance(big[lo:hi,left:right]);distance_small=outside_distance(small[lo:hi,left:right]|gap)
            for original,large_new,small_new,sigma in zip(original_bands,big_bands,small_bands,[8,16,32,48]):
                large_weight=gaussian_filter(big[lo:hi,left:right].astype('float32'),sigma)*(1-smoothstep((distance_big-128)/64))
                small_weight=gaussian_filter((small[lo:hi,left:right]|gap).astype('float32'),sigma)*(1-smoothstep((distance_small-128)/64))
                if sigma==48:small_weight*=gaussian_filter(gap.astype('float32'),32)
                delta+=large_weight*(large_new-original)+small_weight*(small_new-original)
            base=gaussian_filter(reference[lo:hi,left:right].astype('float32')/255,32)
            tone=np.clip(ndi.map_coordinates(shade,[yy,xx],order=1,mode='nearest'),-.1,.1)*(base+.15)*tone_alpha[lo:hi,left:right]
            coarse=gaussian_filter(v,64)
            proposed=v+tone+(coarse+.12)*delta
            local_tiny=np.array(tiny[lo:hi,left:right])
            if np.any(local_tiny):
                # Compact local solves keep one-pixel holes eligible without
                # allocating a full-resolution sparse matrix for the whole tile.
                labels,n=ndi.label(local_tiny);objects=ndi.find_objects(labels)
                for label,box in enumerate(objects,1):
                    if box is None:continue
                    a,b=box
                    roi=(slice(max(0,a.start-2),min(v.shape[0],a.stop+2)),slice(max(0,b.start-2),min(v.shape[1],b.stop+2)))
                    submask=labels[roi]==label
                    if np.any(submask[[0,-1]]) or np.any(submask[:,[0,-1]]):
                        indices=ndi.distance_transform_edt(submask,return_distances=False,return_indices=True)
                        replacement=proposed[roi][tuple(indices)]
                    else:replacement,_=harmonic(proposed[roi],submask,screen=0)
                    proposed[roi][submask]=replacement[submask]
            allowed=(tone_alpha[lo:hi,left:right]>0)|(distance_big<192)|(distance_small<192)|local_tiny
            view=np.s_[sl,core_left-left:core_right-left]
            core=np.s_[start:end,core_left:core_right]
            out=current[core].copy();active=allowed[view]
            out[active]=np.rint(np.clip(proposed[view][active],0,1)*65535).astype('uint16')
            protected+=int(np.count_nonzero(out[~active]!=current[core][~active]))
            output[core]=out;changed_map[core]=out!=current[core];allowed_map[core]=active
            changed+=int(np.count_nonzero(out!=current[core]));covered+=int(np.count_nonzero(~available[start:end]))
            measured_count+=int(np.count_nonzero(measured_region[view]))
            self.log('Rendered rows',start,end)
            del original_bands,big_bands,small_bands;gc.collect()
        pool.shutdown()
        output.flush();changed_map.flush();allowed_map.flush()
        if protected:raise RuntimeError('Protected pixels changed.')
        self.report.update(protected_pixels_changed=protected,changed_percent=100*changed/(self.n*self.n),
            no_usable_source_pixels_assigned=covered,no_usable_source_pixels_unassigned=0,
            measured_detail_core_percent=100*measured_count/(self.n*self.n),block_rows=render_block,render_halo=512,filter_workers=workers)
        self.save()
    def save(self):
        if getattr(self,'_defer_save',False):return
        output=self.load('output')
        outpath=Path(self.config['output']);outpath.parent.mkdir(parents=True,exist_ok=True)
        import tempfile
        fd,name=tempfile.mkstemp(prefix='final_',suffix='.png',dir=self.root);os.close(fd)
        temporary=Path(name)
        with open(temporary,'wb') as f:Image.fromarray(output).save(f,format='PNG',compress_level=2);f.flush();os.fsync(f.fileno())
        with Image.open(temporary) as im:im.verify()
        try:os.link(temporary,outpath)
        finally:temporary.unlink(missing_ok=True)
        with Image.open(outpath) as im:
            saved=np.asarray(im)
            if not np.array_equal(saved,output):raise RuntimeError('PNG roundtrip changed pixels.')
        self.report['output_sha256']=sha(outpath);self.report['png_roundtrip_exact']=True
        self.report['versions']={'numpy':np.__version__,'scipy':__import__('scipy').__version__,'Pillow':Image.__version__}
        outpath.with_suffix('.diagnostics.json').write_text(json.dumps(self.report,indent=2))
        self.log('Saved',outpath,self.report['changed_percent'],'percent changed')
    def restrain_shadow_texture(self):
        from lunar_shadow_texture import adjustment
        current=self.load('current');texture=self.load('texture');quilt=self.load('quilting')
        big=self.load('big_core');expected=self.load('weak_expected');allowed=self.load('allowed')
        output=self.load('output','r+');changed=self.load('changed','r+')
        altered=self.create('shadow_adjusted','bool');altered[:]=False
        touched=blocks=0;block=1024;halo=512
        for y in range(0,self.n,block):
            ey=min(y+block,self.n)
            for x in range(0,self.n,block):
                ex=min(x+block,self.n)
                seed=np.s_[max(0,y-192):min(self.n,ey+192),max(0,x-192):min(self.n,ex+192)]
                if not np.any(big[seed]):continue
                v=current[seed].astype('float32')/65535;t=texture[seed]
                coarse_estimate=(v-.12*t)/(1+t)
                if not np.any(big[seed]&(coarse_estimate<.36)):continue
                lo=max(0,y-halo);hi=min(self.n,ey+halo);left=max(0,x-halo);right=min(self.n,ex+halo)
                roi=np.s_[lo:hi,left:right];view=np.s_[y-lo:ey-lo,x-left:ex-left];core=np.s_[y:ey,x:ex]
                correction=adjustment(current[roi],np.array(texture[roi]),np.array(quilt[roi]),big[roi],expected,(lo,left),workers=int(self.config.get('filter_workers',1)))[view]
                active=allowed[core]&(np.abs(correction)>1e-10)
                before=np.array(output[core]);after=before.copy()
                after[active]=np.rint(np.clip(before[active].astype('float32')/65535+correction[active],0,1)*65535).astype('uint16')
                output[core]=after;changed[core]=after!=current[core];altered[core]=after!=before
                touched+=int(np.count_nonzero(after!=before));blocks+=1
                self.log('Shadow refinement block',y,x)
                del correction,before,after;gc.collect()
        output.flush();changed.flush();altered.flush()
        protected=count=0
        for start,end,*_ in self.stripes():
            different=output[start:end]!=current[start:end]
            count+=int(np.count_nonzero(different));protected+=int(np.count_nonzero(different&~allowed[start:end]))
        if protected:raise RuntimeError('Shadow refinement changed protected pixels.')
        self.report.update(changed_percent=100*count/(self.n*self.n),protected_pixels_changed=protected)
        self.report['shadow_refinement']={'blocks':blocks,'changed_from_first_render_percent':100*touched/(self.n*self.n),
            'policy':'In low-information quilted shadows, preserve original broad structure and calibrate 0–8px donor texture to nearby trusted fine contrast.',
            'full_shadow_below':.22,'fade_to_original_quilting_above':.35,'minimum_fine_gain':.025}
        self.report['no_usable_source_percent']=100*self.report['no_usable_source_pixels']/(self.n*self.n)
        self.save()
    def final_refinement(self):
        from lunar_final_refinement import run_stage
        final,report=run_stage(self.root/'output.npy',self.config['old_support'],self.config['support'],self.config['repair_strength'],self.root/'final_stage',reference_path=self.root/'reference.npy',source_folder=self.root)
        stage=load_array(final,mmap_mode='r');out=self.load('output','r+');allowed=self.load('allowed','r+')
        new_allowed=load_array(self.root/'final_stage/final_allowed.npy',mmap_mode='r');current=self.load('current')
        changed=self.load('changed','r+');count=0
        for start,end,*_ in self.stripes():
            out[start:end]=stage[start:end];allowed[start:end]|=new_allowed[start:end]
            changed[start:end]=out[start:end]!=current[start:end];count+=int(np.count_nonzero(changed[start:end]))
        out.flush();allowed.flush();changed.flush()
        self.report['final_refinement']=report;self.report['changed_percent']=100*count/(self.n*self.n)
        self.save()
    def run(self):
        started=time.time()
        self._defer_save=True
        self.report['stage_seconds']={}
        for stage in [self.prepare,self.masks,self.quilt,self.render,self.restrain_shadow_texture,self.final_refinement]:
            before=time.perf_counter();stage();elapsed=time.perf_counter()-before
            self.report['stage_seconds'][stage.__name__]=elapsed;self.log(stage.__name__,'seconds',round(elapsed,2))
            if stage.__name__=='quilt':
                from lunar_io import release_source_cache
                release_source_cache();gc.collect()
                for p in self.root.iterdir():
                    if p.is_file() and ('_analysis_' in p.name or p.name in ['Seamless.npy','Morning.npy','Evening.npy','fine_std.npy','weak_raw.npy']):unlink_array(p)
        self._defer_save=False
        before=time.perf_counter();self.save();self.report['stage_seconds']['png_save']=time.perf_counter()-before
        self.log('Elapsed seconds',time.time()-started)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    Trial(json.loads(Path(args.config).read_text())).run()
