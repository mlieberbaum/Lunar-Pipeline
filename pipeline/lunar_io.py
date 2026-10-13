from lunar_workspace import load_array, open_array, unlink_array
"""Exact array I/O and per-run reuse of normalized source pixels."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import numpy as np
from PIL import Image

_sources = ContextVar('lunar_normalized_sources', default=None)


def read_array(path):
    if Path(path).suffix.lower() == '.npy':
        return load_array(path, mmap_mode='r', allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def write_array(path, array):
    if Path(path).suffix.lower() == '.npy':
        np.save(path, array, allow_pickle=False)
    else:
        Image.fromarray(array).save(path)


@contextmanager
def reuse_sources(folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    token = _sources.set((folder, {}, {}))
    try:
        yield
    finally:
        _sources.reset(token)


def cached_source(path, name, loader):
    active = _sources.get()
    if active is None:
        return loader(path, name)
    folder, entries, _ = active
    key = (str(Path(path).resolve()), name)
    if key not in entries:
        array = loader(path, name)
        target = folder / f'source_{len(entries)}.npy'
        np.save(target, array, allow_pickle=False)
        entries[key] = target
    return load_array(entries[key], mmap_mode='r', allow_pickle=False)


def sources_reused():
    return _sources.get() is not None

def resize_source(array,size):
    def compute():
        return np.clip(np.rint(np.asarray(Image.fromarray(array).convert('F').resize((size,size),Image.Resampling.BICUBIC))),0,65535).astype('uint16')
    active=_sources.get()
    if active is None or not getattr(array,'filename',None):return compute()
    folder,entries,_=active
    key=('resize',str(array.filename),size)
    if key not in entries:
        target=folder/f'resized_{len(entries)}.npy';np.save(target,compute());entries[key]=target
    return load_array(entries[key],mmap_mode='r')

def release_source_cache():
    active=_sources.get()
    if active is None:return
    import gc
    paths=sorted(set(active[1].values()));active[1].clear();gc.collect()
    for path in paths:unlink_array(path)


def blend_fields(size=None):
    """Core pixels only: reuse base fields; repair's halo pixels are discarded."""
    active = _sources.get()
    if active is None:
        return None
    folder, _, fields = active
    if not fields and size is not None:
        for name in ['value', 'total', 'base', 'morning_detail', 'morning_weight']:
            fields[name] = open_array(folder/(name+'.npy'),
                mode='w+', dtype='float32', shape=(size,size))
    return fields or None


def release_blend_fields():
    active = _sources.get()
    if active is None:
        return
    fields = active[2]
    paths = [Path(field.filename) for field in fields.values()]
    for field in fields.values():field._mmap.close()
    fields.clear()
    for path in paths:unlink_array(path)
