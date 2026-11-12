"""Process-local allocator policy for low-memory FP16 profiles. No global environment changes."""
import os,sys

ALLOCATOR='backend:native,max_split_size_mb:128'

def configure_allocator():
    if all(os.environ.get(k)==ALLOCATOR for k in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']):return
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError('Crop allocator policy must be configured before CUDA initialization')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']=ALLOCATOR
    os.environ['PYTORCH_ALLOC_CONF']=ALLOCATOR

