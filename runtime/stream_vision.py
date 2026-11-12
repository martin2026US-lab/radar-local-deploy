"""RADAR vision adaptation: bound mask workspace to one organ at a time.

Derived from upstream vision_branch.py (Apache-2.0 header, MONAI Consortium).
Original project: https://github.com/alibaba-damo-academy/damo-radar
Changes: release consumed tensors; stream organ mask pooling. Same arithmetic,
ROI, softmax, convolution parameters and output semantics as the original.
"""
import torch
import torch.nn.functional as F
from dynamic_network_architectures.vision_branch import VisionBranch

def softmax_argmax_slabs(logits,depth=8):
    # Retain the original softmax (including AMP FP32 behavior and rounding),
    # but materialize it for only a depth slab. Each voxel still sees all classes.
    shape=(logits.shape[0],)+tuple(logits.shape[2:])
    result=torch.empty(shape,dtype=torch.long,device=logits.device)
    for d in range(0,logits.shape[2],depth):
        result[:,d:d+depth]=torch.softmax(logits[:,:,d:d+depth],1).argmax(1)
    return result

def upsample_to_cpu_chunks(prob,size,channels=4):
    # Trilinear interpolation is independent across channels. Keep its original
    # FP32 CUDA arithmetic, transferring four channels at a time to CPU storage.
    result=torch.empty((prob.shape[0],prob.shape[1],*size),dtype=torch.float32,device='cpu')
    for c in range(0,prob.shape[1],channels):
        block=F.interpolate(prob[:,c:c+channels].float(),size=size,mode='trilinear')
        result[:,c:c+channels].copy_(block.cpu())
        del block
    return result

def mask_via_cpu_workspace(logits,size,depth=8):
    # Keep the large interpolated class volume on CPU. Interpolation and softmax
    # still run on CUDA with their original FP32 arithmetic, in independent blocks.
    expanded=upsample_to_cpu_chunks(logits,size)
    result=torch.empty((logits.shape[0],*size),device=logits.device,dtype=torch.long)
    for d in range(0,size[0],depth):
        block=expanded[:,:,d:d+depth].to(logits.device)
        result[:,d:d+depth]=torch.softmax(block,1).argmax(1)
        del block
    return result

def streamed_flags(y,lengths,organ_count=36):
    flags=[torch.zeros(y.shape[0],organ_count,n,dtype=torch.bool,device=y.device) for n in lengths]
    kernels=((8,32,32),(4,16,16),(2,8,8))
    with torch.no_grad():
        for batch in range(y.shape[0]):
            for label in torch.unique(y[batch]).tolist():
                if label==0:continue
                if not 1<=label<=organ_count:raise ValueError('Out-of-range anatomical label')
                mask=(y[batch]==label).float()[None,None]
                for dest,kernel in zip(flags,kernels):
                    dest[batch,label-1]=F.max_pool3d(mask,kernel_size=kernel,stride=kernel).flatten()>0
                del mask
    return flags

class StreamVisionBranch(VisionBranch):
    def forward(self,x,y):
        skips,segs=self.UNet(x)
        scale1=self.proj1(skips[-1]);scale2=self.proj2(skips[-2]);scale3=self.proj3(skips[-3])
        pred_logit=segs[0]
        seg_probs=torch.softmax(pred_logit,1)
        # Only projections and segs[0] are needed now. Release high-resolution
        # encoder skips and unused deep-supervision outputs before interpolation.
        del skips,segs
        target_size=[pred_logit.shape[-3],pred_logit.shape[-2]*2,pred_logit.shape[-1]*2]
        pred_mask=mask_via_cpu_workspace(pred_logit,target_size)
        del pred_logit
        res1=scale1.flatten(2).transpose(1,2)
        res2=scale2.flatten(2).transpose(1,2)
        res3=scale3.flatten(2).transpose(1,2)
        del scale1,scale2,scale3
        f1,f2,f3=streamed_flags(pred_mask,(res1.shape[1],res2.shape[1],res3.shape[1]),len(self.organs))
        return seg_probs,pred_mask,res1,res2,res3,f1,f2,f3
