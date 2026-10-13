"""Per-call mapping ownership; close handles before Windows cleanup."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import weakref
import numpy as np

_mappings=ContextVar('lunar_work_mappings',default=None)

def _track(array):
    registry=_mappings.get()
    if registry is not None and isinstance(array,np.memmap):
        registry.append((Path(array.filename).resolve(),weakref.ref(array._mmap)))
    return array

def load_array(*args,**kwargs):
    return _track(np.load(*args,**kwargs))

def open_array(*args,**kwargs):
    return _track(np.lib.format.open_memmap(*args,**kwargs))

def close_mappings(path=None):
    registry=_mappings.get()
    if registry is None:return
    target=Path(path).resolve() if path is not None else None
    retained=[]
    for filename,reference in registry:
        mapping=reference()
        if mapping is None or mapping.closed:continue
        if target is None or filename==target:mapping.close()
        else:retained.append((filename,reference))
    registry[:]=retained

def unlink_array(path):
    close_mappings(path)
    Path(path).unlink(missing_ok=True)

@contextmanager
def mapping_scope():
    token=_mappings.set([])
    try:yield
    finally:
        try:close_mappings()
        finally:_mappings.reset(token)

@contextmanager
def claim_output(output):
    lock=Path(output).with_name(Path(output).name+'.lunar.lock')
    try:
        with lock.open('x') as stream:stream.write('Output is being processed.\n')
    except FileExistsError:
        raise FileExistsError(f'Another call already owns this output: {output}') from None
    try:yield
    finally:lock.unlink(missing_ok=True)
