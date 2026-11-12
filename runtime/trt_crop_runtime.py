"""Full-size FP16 crop execution with one resident engine and shape-sized scratch memory.

The original static engine still handles ordinary windows. An additional dynamic
engine handles larger crops without resizing/tiling the CT or switching precision.
TensorRT activation memory is owned by PyTorch and managed by the default CUDA allocator without a per-process quota.
"""
import gc
from pathlib import Path
import int8_setup
import tensorrt as trt
import query_radar as q
from query_radar import ROOT,torch
from stream_vision import mask_via_cpu_workspace,streamed_flags
LOGGER=trt.Logger(trt.Logger.WARNING)

STATIC_SHAPE=(1,1,96,256,384)

def validate_shape(shape,manifest):
    shape=tuple(shape)
    lo=manifest['min_shape'];hi=manifest['max_shape']
    if len(shape)!=5 or shape[:2]!=(1,1) or any(x%32 for x in shape[2:]):
        raise ValueError('FP16 crop engine requires batch/channel 1 and spatial sizes divisible by 32')
    if any(x<a or x>b for x,a,b in zip(shape,lo,hi)):
        raise ValueError(f'Crop {shape} exceeds configured dynamic range {tuple(lo)} .. {tuple(hi)}; no silent resizing or fallback')
    return shape

class ShapeAwareTRTBranch(torch.nn.Module):
    def __init__(self,folder):
        torch.nn.Module.__init__(self)
        self.folder=folder;self.organs=[]
        self.specs={
            'static':ROOT/'models/int8_experiment/vision_fp16',
            'dynamic':ROOT/'models/trt_dynamic/vision_dynamic_fp16',
        }
        self.manifests={k:q.read_json(str(p)+'.manifest.json') for k,p in self.specs.items()}
        for k,path in self.specs.items():
            if q.sha256(str(path)+'.plan')!=self.manifests[k]['sha256']:raise ValueError(f'{k} engine checksum mismatch')
        self.manifest=dict(self.manifests['static'],revision='dynamic_uncapped_v3',dynamic_crop_engine=self.manifests['dynamic'])
        self.runtime=trt.Runtime(LOGGER)
        self.engine=None;self.context=None;self.scratch=None;self.active_kind=None;self.active_shape=None
        self.audit={'revision':'dynamic_uncapped_v3','precision':'FP16','whole_crop_preserved':True,
            'allocation':'USER_MANAGED, per-shape scratch managed by the uncapped PyTorch allocator',
            'cache_trim_points':['after_engine_output_cast','after_mask_interpolation'],
            'resident_context_limit':1,'calls':0,'switches':[],'shape_allocations':[],
            'engine_sha256':{k:m['sha256'] for k,m in self.manifests.items()}}

    def _select(self,kind):
        if self.active_kind==kind:return
        torch.cuda.synchronize()
        self.context=None;self.scratch=None;self.engine=None
        gc.collect();torch.cuda.empty_cache()
        self.engine=self.runtime.deserialize_cuda_engine((self.specs[kind].parent/(self.specs[kind].name+'.plan')).read_bytes())
        if self.engine is None:raise RuntimeError('TensorRT engine incompatible with current GPU/runtime')
        self.context=self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        if self.context is None:raise RuntimeError('Cannot create TensorRT context')
        self.active_kind=kind;self.active_shape=None
        self.audit['switches'].append(kind)

    def core(self,x):
        shape=tuple(x.shape)
        kind='static' if shape==STATIC_SHAPE else 'dynamic'
        if kind=='dynamic':validate_shape(shape,self.manifests['dynamic'])
        self._select(kind)
        ctx=self.context
        if self.active_shape!=shape:
            torch.cuda.synchronize()
            if not ctx.set_input_shape('image',shape):raise RuntimeError('TensorRT rejected input shape')
            missing=ctx.infer_shapes()
            if missing:raise RuntimeError(f'TensorRT shape inference incomplete: {missing}')
            required=ctx.update_device_memory_size_for_shapes()
            if required<=0:raise RuntimeError('Invalid TensorRT scratch requirement')
            # No execution is in flight. Release the previous buffer before growing.
            self.scratch=None;torch.cuda.empty_cache()
            self.scratch=torch.empty(required,dtype=torch.uint8,device='cuda')
            if self.scratch.data_ptr()%256:raise RuntimeError('Unaligned TensorRT scratch buffer')
            ctx.set_device_memory(self.scratch.data_ptr(),required)
            self.active_shape=shape
            self.audit['shape_allocations'].append({'engine':kind,'shape':list(shape),'scratch_bytes':required})
        x=x.float().contiguous()
        ctx.set_tensor_address('image',x.data_ptr())
        outputs=[]
        for name in ['logits','scale1','scale2','scale3']:
            shape_out=tuple(ctx.get_tensor_shape(name))
            if any(n<=0 for n in shape_out):raise RuntimeError('Unresolved output dimensions')
            dtype={trt.float32:torch.float32,trt.float16:torch.float16}.get(self.engine.get_tensor_dtype(name))
            if dtype is None:raise RuntimeError('Unexpected TensorRT output dtype')
            output=torch.empty(shape_out,dtype=dtype,device='cuda')
            ctx.set_tensor_address(name,output.data_ptr());outputs.append(output)
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError('TensorRT execution failed')
        self.audit['calls']+=1
        q.dump(self.folder/'crop_runtime.json',self.audit)
        return outputs

    def forward(self,x,y):
        logits,a,b,c=self.core(x)
        logits=logits.half();a=a.half();b=b.half();c=c.half()
        if not all(torch.isfinite(z).all() for z in [logits,a,b,c]):raise RuntimeError('Nonfinite TensorRT output')
        torch.cuda.empty_cache()
        probs=torch.softmax(logits,1)
        mask=mask_via_cpu_workspace(logits,[logits.shape[-3],logits.shape[-2]*2,logits.shape[-1]*2])
        del logits
        torch.cuda.empty_cache()
        res=[z.flatten(2).transpose(1,2) for z in [a,b,c]]
        flags=streamed_flags(mask,[z.shape[1] for z in res],len(self.organs))
        return probs,mask,*res,*flags

