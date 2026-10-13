"""One worker executes synchronously, including nested processing stages."""
from concurrent.futures import ThreadPoolExecutor as _Executor,Future

class SerialExecutor:
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def shutdown(self,*args,**kwargs):pass
    def map(self,function,*iterables):return map(function,*iterables)
    def submit(self,function,*args,**kwargs):
        result=Future()
        try:result.set_result(function(*args,**kwargs))
        except BaseException as error:result.set_exception(error)
        return result

def ThreadPoolExecutor(max_workers):
    return SerialExecutor() if max_workers==1 else _Executor(max_workers=max_workers)
