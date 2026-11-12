"""Unvalidated FP32 workspace-only repair prototype; no inference on import.

Uses the existing streamed vision execution with the unchanged original model
constructor/parameters and checkpoint. This does not change convolution layers,
precision, crop geometry, fixed text scores or CPU stitching. Validation against
an existing same-precision successful control is required before batch adoption.
"""

def install(module):
    import torch
    from stream_vision import StreamVisionBranch,upsample_to_cpu_chunks
    original_load=module.load_module
    def load_module():
        mod=original_load()
        original_class=mod.VisionBranch
        class FP32WorkspaceVisionBranch(original_class):
            # Inherit the exact original constructor and trained parameter names.
            # Only allocation lifetime and postprocessing workspaces are changed.
            forward=StreamVisionBranch.forward
        mod.VisionBranch=FP32WorkspaceVisionBranch
        return mod
    module.load_module=load_module
    def forward(model,patch,logits,test_organs,text,features,skip=None):
        if model.local_amp_enabled:
            raise RuntimeError('FP32 workspace prototype must not enable autocast')
        if any(p.dtype!=torch.float32 for p in model.visual_encoder.parameters()):
            raise RuntimeError('Visual weights must remain FP32')
        x=patch.to(device='cuda',dtype=torch.float32)
        logits,prob=model.forward_test_win(x,None,logits,test_organs,text,features,None,skip_organ=skip)
        if prob.dtype!=torch.float32:raise RuntimeError('Visual outputs must remain FP32')
        if not torch.isfinite(prob).all():raise RuntimeError('Nonfinite FP32 segmentation probabilities')
        return logits,upsample_to_cpu_chunks(prob,x.shape[2:])
    module.forward=forward
