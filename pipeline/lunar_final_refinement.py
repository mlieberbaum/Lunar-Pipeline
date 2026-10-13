from lunar_workspace import load_array, open_array, unlink_array
"""Final residual/left-boundary stage. Fixed full-map geometry; no tile IDs.

Preserves existing output unless an old repair component contains a long,
nearly straight left edge. Processing windows and donors are established from
the complete diagnostic map, independent of which crop is later displayed.
"""
import gc,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as nd
from lunar_residual_texture import residual
from lunar_boundary_refinement import refine
from lunar_texture_contrast import calibrate
Image.MAX_IMAGE_PIXELS=None

def windows(old):
    stride=8;small=old[::stride,::stride]<.85*255
    labels,n=nd.label(small);counts=np.bincount(labels.ravel());objects=nd.find_objects(labels)
    selected=[];h,w=old.shape;minimum=max(1536,int(.16*h));padding=1024
    for i,box in enumerate(objects,1):
        if not box or counts[i]*stride**2<262144:continue
        sy,sx=box;region=labels[box]==i;rows=region.any(axis=1)
        yy=np.flatnonzero(rows);left=np.argmax(region[rows],axis=1)+sx.start
        mode=int(np.argmax(np.bincount(left)));straight=np.abs(left-mode)<=2
        runs=nd.find_objects(nd.label(straight)[0])
        longest=max((s[0].stop-s[0].start for s in runs if s),default=0)*stride
        if longest<minimum:continue
        y0=max(0,sy.start*stride-padding);y1=min(h,sy.stop*stride+padding)
        x0=max(0,sx.start*stride-padding);x1=min(w,sx.stop*stride+padding)
        selected.append({'xyxy':[x0,y0,x1,y1],'straight_run_pixels':longest,'left_mode_x':mode*stride})
    return sorted(selected,key=lambda v:(v['xyxy'][1],v['xyxy'][0]))

def run_stage(input_path,old_path,support_path,repair_path,work_folder,reference_path=None,source_folder=None):
    root=Path(work_folder);root.mkdir(parents=True,exist_ok=True)
    baseline=load_array(input_path,mmap_mode='r')
    if baseline.dtype!=np.uint16 or baseline.ndim!=2:raise ValueError('Expected a uint16 tile.')
    def cache(name,path):
        if Path(path).suffix.lower()=='.npy':
            a=load_array(path,mmap_mode='r',allow_pickle=False)
            if a.shape!=baseline.shape:raise ValueError('Diagnostic map dimensions differ.')
            return a
        target=root/(name+'.npy')
        with Image.open(path) as im:a=np.asarray(im)
        if a.shape!=baseline.shape:raise ValueError('Diagnostic map dimensions differ.')
        dest=open_array(target,mode='w+',dtype=a.dtype,shape=a.shape);dest[:]=a;dest.flush()
        del a,dest;gc.collect();return load_array(target,mmap_mode='r')
    old=cache('old_support',old_path);support=cache('support',support_path);repair=cache('repair',repair_path)
    regions=windows(old)
    reference=load_array(reference_path,mmap_mode='r') if reference_path else None
    result=open_array(root/'final.npy',mode='w+',dtype='uint16',shape=baseline.shape)
    result[:]=baseline
    allowed=open_array(root/'final_allowed.npy',mode='w+',dtype='bool',shape=baseline.shape);allowed[:]=False
    report={'algorithm':'final-residual-and-left-boundary-2','candidate_windows':regions,'regions':[],
        'coordinates_policy':'Full diagnostic geometry; no tile ID or manual repair coordinates.',
        'donor_policy':'Fixed map-derived component window and raster order. Donors are local to that canonical window.',
        'limitations':['Only two tiles validated; this stage detects long straight repair boundaries.',
                      'Exemplar placement is synthetic, not recovered geography.',
                      'Canonical window changes can change texture; crop/full-tile byte equivalence is not claimed.']}
    for j,region in enumerate(regions):
        x0,y0,x1,y1=region['xyxy'];roi=np.s_[y0:y1,x0:x1]
        before=np.array(baseline[roi]);old_local=np.asarray(old[roi],dtype='float32')/255
        print('Final refinement window',j,region,flush=True)
        after,active,mask,stats=residual(before,old_local,np.asarray(support[roi],dtype='float32')/65535,np.asarray(repair[roi],dtype='float32')/255)
        print('Residual quilting',stats,flush=True)
        after,edge_active,edge=refine(after,old_local);active|=edge_active
        print('Old left-boundary correction',edge,flush=True)
        fill={}
        if mask.any():
            # Fragmented residuals may have no long edge; keep their texture repair.
            try:
                after,fill_active,fill=refine(after,(~mask).astype('float32'),correct_tone=False)
                active|=fill_active
            except ValueError as e:fill={'skipped':str(e)}
        if mask.any():
            after,contrast_active,contrast=calibrate(before,after,old_local,np.asarray(support[roi],dtype='float32')/65535,np.asarray(repair[roi],dtype='float32')/255,mask)
            active|=contrast_active
        else:contrast=[]
        from lunar_perimeter import blend
        after,perimeter_active,perimeter=blend(after,old_local<.85,reference[roi] if reference is not None else None)
        active|=perimeter_active
        assert np.array_equal(after[~active],before[~active])
        result[roi]=after;allowed[roi]|=active
        report['regions'].append(dict(window=region,residual=stats,old_boundary=edge,fill_boundary=fill,local_contrast=contrast,perimeter=perimeter))
        np.save(root/f'region_{j}_residual_mask.npy',mask)
        del before,after,active,mask,edge_active,old_local;gc.collect()
    result.flush();allowed.flush()
    from lunar_global_repair import run
    repaired,repair_allowed,global_report=run(root/'final.npy',old_path,support_path,repair_path,root/'global',reference_path=reference_path,source_folder=source_folder)
    merged=load_array(repaired,mmap_mode='r');new_allowed=load_array(repair_allowed,mmap_mode='r')
    for y in range(0,baseline.shape[0],1024):
        result[y:y+1024]=merged[y:y+1024];allowed[y:y+1024]|=new_allowed[y:y+1024]
    report['global_repair']=global_report;report['whole_footprint_texture']=[]
    for region in regions:
        x0,y0,x1,y1=region['xyxy'];roi=np.s_[y0:y1,x0:x1]
        after,active,stats=blend(np.array(result[roi]),old[roi]<.85*255,full_texture=True)
        result[roi]=after;allowed[roi]|=active;report['whole_footprint_texture'].append(stats)
    report['algorithm']='whole-tile-residual-and-all-edge-refinement-4'
    report['limitations']=['Exemplar texture is synthetic placement, not recovered geography.','Appearance on additional tiles requires review.']
    result.flush();allowed.flush();changed=protected=0
    for y in range(0,baseline.shape[0],384):
        different=result[y:y+384]!=baseline[y:y+384]
        changed+=int(different.sum());protected+=int(np.count_nonzero(different&~allowed[y:y+384]))
    if protected:raise RuntimeError('Final stage changed protected pixels.')
    report.update(protected_pixels_changed=protected,changed_pixels=changed,changed_percent=100*changed/baseline.size,
                  versions={'numpy':np.__version__,'scipy':__import__('scipy').__version__,'Pillow':Image.__version__})
    (root/'final_refinement.diagnostics.json').write_text(json.dumps(report,indent=2))
    print('Final refinement complete',report['changed_percent'],'percent changed',flush=True)
    return root/'final.npy',report
