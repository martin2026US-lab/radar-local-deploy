"""Unchanged tested FP16 builder/scoring path, extracted from run_trt_inference.py."""
import types,torch
import query_radar as q
from query_radar import ROOT
MODE="fp16"
OUT=ROOT/"models/int8_experiment"

class UnusedText(torch.nn.Module):
    config=types.SimpleNamespace(hidden_size=768)
    def resize_token_embeddings(self,n):pass
    def forward(self,*args,**kwargs):raise RuntimeError('Use the separate CPU INT8 text query encoder')

def build_model(mod,precision,folder):
    if MODE=='fp16':
        from trt_crop_runtime import ShapeAwareTRTBranch
        branch=ShapeAwareTRTBranch(folder)
    else:
        branch=TRTBranch()
    model=mod.RADAR(image_encoder=branch,text_encoder=UnusedText())
    model.text_proj=torch.nn.Identity()
    state=torch.load(OUT/'scoring_fp32.pth',map_location='cpu',weights_only=False)['model']
    model.load_state_dict(state,strict=True);branch.organs=model.organs
    model.eval().requires_grad_(False)
    for child in (model.attention,model.vision_projs):child.cuda()
    model.query_tokens=torch.nn.Parameter(model.query_tokens.detach().cuda(),requires_grad=False)
    model.temp=torch.nn.Parameter(model.temp.detach().cuda(),requires_grad=False)
    model.local_amp_enabled=True;model.local_amp_dtype=torch.float16
    q.dump(folder/'engine_manifest.json',branch.manifest)
    return model
