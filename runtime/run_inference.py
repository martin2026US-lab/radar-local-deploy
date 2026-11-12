"""Isolated RADAR trial: original model, CPU stitching, optional vision autocast.

The upstream directory is immutable. A generated copy changes only the vision
call precision and casts its three feature outputs back to FP32 for scoring.
All runs use the original ROI, preprocessing, attention and fixed text features.
"""
import argparse, contextlib, gc, hashlib, importlib.util, json, os, pathlib
import sys, threading, time, traceback

ROOT = pathlib.Path(os.environ['RADAR_ASSET_ROOT']).resolve()
UPSTREAM = pathlib.Path(__file__).resolve().parents[1] / 'vendor' / 'RADAR_inference'
EXPERIMENT = pathlib.Path(os.environ['RADAR_OUTPUT_ROOT']).resolve()
os.environ['MODEL_ROOT'] = str(ROOT / 'models')
os.environ['CONFIGS_ROOT'] = str(ROOT / 'models')
sys.path.insert(0, str(UPSTREAM))

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data.utils import dense_patch_slices

GiB = 1024 ** 3
ROI = (96, 256, 384)

def dump(path, data):
    pathlib.Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_module():
    source = (UPSTREAM / 'inference_demo.py').read_text(encoding='utf-8')
    old = '        seg_probs, seg, image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3 = self.visual_encoder(images, None)'
    new = '''        with torch.autocast(device_type="cuda", dtype=self.local_amp_dtype, enabled=self.local_amp_enabled):
            seg_probs, seg, image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3 = self.visual_encoder(images, None)
        image_embeds1 = image_embeds1.float()
        image_embeds2 = image_embeds2.float()
        image_embeds3 = image_embeds3.float()'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    folder = EXPERIMENT / 'implementation'; folder.mkdir(parents=True,exist_ok=True)
    patched = folder / 'inference_precision.py'
    patched.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('inference_precision', patched)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class Monitor:
    def __init__(self, folder):
        self.folder = folder; self.phase = 'startup'; self.stop_event = threading.Event()
        self.proc = psutil.Process(); self.rows = []; self.unsafe = None
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit(); self.handle = pynvml.nvmlDeviceGetHandleByIndex(0); self.nvml = pynvml
        except Exception as e: print('NVML unavailable:',repr(e),flush=True)
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def sample(self):
        mem = self.proc.memory_info(); vm = psutil.virtual_memory()
        row = {'t':time.time(), 'phase':self.phase, 'rss':mem.rss, 'private':getattr(mem,'private',mem.vms), 'ram_available':vm.available}
        if self.nvml:
            g = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            row.update(gpu_global_used=g.used, gpu_global_free=g.free)
        self.rows.append(row)
        if vm.available < 1.5 * GiB: self.unsafe = 'Available system RAM fell below 1.5 GiB'
    def loop(self):
        while not self.stop_event.is_set():
            try: self.sample()
            except Exception: pass
            self.stop_event.wait(0.2)
    def start(self): self.thread.start()
    def close(self):
        self.stop_event.set(); self.thread.join(timeout=2)
        dump(self.folder/'resource_trace.json', self.rows)
        if not self.rows: return {}
        return { 'rss_peak_GiB':max(r['rss'] for r in self.rows)/GiB,
            'private_peak_GiB':max(r['private'] for r in self.rows)/GiB,
            'system_available_min_GiB':min(r['ram_available'] for r in self.rows)/GiB,
            'gpu_global_peak_GiB':max(r.get('gpu_global_used',0) for r in self.rows)/GiB,
            'samples':len(self.rows), 'interval_seconds':0.2 }

def guard(monitor, extra_cpu=0):
    if monitor.unsafe: raise RuntimeError(monitor.unsafe)
    available = psutil.virtual_memory().available
    if available - extra_cpu < 2 * GiB:
        raise RuntimeError(f'RAM preflight: available={available/GiB:.2f} GiB, new={extra_cpu/GiB:.2f} GiB; keeping 2 GiB free')

def stage_summary(start, name, shape=None):
    torch.cuda.synchronize()
    result = {'stage':name,'seconds':time.perf_counter()-start,
              'gpu_allocated_peak_GiB':torch.cuda.max_memory_allocated()/GiB,
              'gpu_reserved_peak_GiB':torch.cuda.max_memory_reserved()/GiB}
    if shape is not None: result['shape']=list(shape)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def stitch_selfcheck():
    # Independent CPU numerical regression for shared counts and chunked division.
    rng = torch.Generator().manual_seed(137)
    acc = torch.rand((1,37,7,9,11),generator=rng)
    count = torch.randint(1,5,(1,1,7,9,11),generator=rng).float()
    original = (acc / count.expand_as(acc)).argmax(1)
    chunked = torch.empty((1,7,9,11),dtype=torch.long)
    for d in range(0,7,3):
        acc[:,:,d:d+3].div_(count[:,:,d:d+3])
        chunked[:,d:d+3]=acc[:,:,d:d+3].argmax(1)
    assert torch.equal(original,chunked)

def smoke(folder):
    result={'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'executable':sys.executable,
            'base_prefix':sys.base_prefix,'gpu':torch.cuda.get_device_name(),'arch_list':torch.cuda.get_arch_list(),
            'bf16_supported':torch.cuda.is_bf16_supported(),'include_system_site_packages':False}
    conv=torch.nn.Conv3d(1,8,3,padding=1).cuda().eval()
    x=torch.randn(1,1,8,32,32,device='cuda')
    result['kernels']={}
    for name,dtype in [('fp32',torch.float32),('fp16',torch.float16),('bf16',torch.bfloat16)]:
        with torch.inference_mode(),torch.autocast('cuda',dtype=dtype,enabled=name!='fp32'):
            y=conv(x)
        torch.cuda.synchronize()
        assert torch.isfinite(y).all()
        result['kernels'][name]={'finite':True,'output_dtype':str(y.dtype)}
    stitch_selfcheck();result['cpu_stitch_check']='pass'
    dump(folder/'smoke.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def build_model(mod, precision, folder, checkpoint_path=None, pretrained_text=True):
    vision=mod.VisionBranch()
    if pretrained_text:
        text=mod.XBertEncoder.from_config({},from_pretrained=True)
    else:
        from compact_runtime import text_encoder_from_config
        text=text_encoder_from_config(mod.XBertEncoder)
    model=mod.RADAR(image_encoder=vision,text_encoder=text)
    cp=torch.load(checkpoint_path or ROOT/'models/checkpoint_radar_pretrain.pth',map_location='cpu',weights_only=False,mmap=True)
    msg=model.load_state_dict(cp['model'],strict=False)
    loading={'missing_keys':list(msg.missing_keys),'unexpected_keys':list(msg.unexpected_keys),
             'checkpoint_top_keys':list(cp.keys()),'parameters':sum(p.numel() for p in model.parameters())}
    allowed_missing={
        'text_encoder.cls.predictions.bias',
        'text_encoder.cls.predictions.transform.dense.weight',
        'text_encoder.cls.predictions.transform.dense.bias',
        'text_encoder.cls.predictions.transform.LayerNorm.weight',
        'text_encoder.cls.predictions.transform.LayerNorm.bias',
    }
    loading['allowed_unused_mlm_keys']=sorted(allowed_missing)
    loading['active_path_missing_keys']=[k for k in msg.missing_keys if k not in allowed_missing]
    loading['missing_key_explanation']='Only unused masked-language-model prediction head keys may be absent. Fixed text embeddings bypass the entire text encoder; a hook rejects any text-encoder invocation.'
    dump(folder/'checkpoint_loading.json',loading)
    if loading['active_path_missing_keys'] or msg.unexpected_keys:
        raise RuntimeError('Unexplained checkpoint keys; see checkpoint_loading.json')
    def reject_text_encoder(*unused):
        raise RuntimeError('Text encoder unexpectedly invoked in fixed-text-feature inference')
    model.text_encoder.register_forward_pre_hook(reject_text_encoder)
    del cp;gc.collect()
    model.eval()
    for child in (model.visual_encoder,model.attention,model.vision_projs): child.cuda()
    model.query_tokens=torch.nn.Parameter(model.query_tokens.detach().cuda(),requires_grad=False)
    model.temp=torch.nn.Parameter(model.temp.detach().cuda(),requires_grad=False)
    model.local_amp_enabled=precision!='fp32'
    model.local_amp_dtype=torch.bfloat16 if precision=='bf16' else torch.float16
    # Keep text branch on CPU: the fixed finding path never calls it.
    model.requires_grad_(False)
    return model

def forward(model, patch, logits, test_organs, text, features, skip=None):
    gpu_patch=patch.to(device='cuda',dtype=torch.float32)
    logits,prob=model.forward_test_win(gpu_patch,None,logits,test_organs,text,features,None,skip_organ=skip)
    if not torch.isfinite(prob).all(): raise RuntimeError('Non-finite segmentation probabilities')
    # Perform the original interpolation in FP32 on CUDA, only for this window.
    up=F.interpolate(prob.float(),size=gpu_patch.shape[2:],mode='trilinear')
    cpu=up.cpu()
    del gpu_patch,prob,up
    return logits,cpu

@torch.inference_mode()
def run(args, folder, mon, metrics):
    mod=load_module();stitch_selfcheck()
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.manual_seed(0);np.random.seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    free,total=torch.cuda.mem_get_info()
    # Revised test mode: no per-process GPU allocator quota; real device limits still apply.
    if getattr(args,'gpu_budget_gib',None) is not None:
        raise ValueError('Unrestricted test mode does not accept an explicit GPU allocator budget')
    metrics.update(gpu_allocator_policy='CUDA default; no explicit per-process quota',
                   gpu_free_at_start_GiB=free/GiB,gpu_total_GiB=total/GiB)
    metrics.update(gpu_allocator_budget_GiB=None,precision=args.precision,roi=list(ROI),overlap=0.25,
                   preprocessing='upstream DataFolder unchanged',text_encoder_device='cpu',score_dtype='float32',
                   source_commit='0dbf0ece209b77d722588e5e7943014d3db48f7f')
    mon.phase='load';start=time.perf_counter();model=build_model(mod,args.precision,folder)
    metrics['stages'].append(stage_summary(start,'load'))
    guard(mon);mon.phase='preprocess';start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    dataset=mod.DataFolder(str(ROOT/'upstream/data/demo_cases'))
    image,items,meta=dataset[0];image=image[None].contiguous();fid=pathlib.Path(meta['img_path']).name
    assert image.device.type=='cpu' and image.shape[0:2]==(1,1)
    metrics['case_id']=fid;metrics['preprocessed_shape']=list(image.shape)
    metrics['stages'].append(stage_summary(start,'preprocess',image.shape))
    # Save once for consistent cross-precision input and exact preprocessing evidence.
    digest=hashlib.sha256(image.numpy().tobytes()).hexdigest();metrics['preprocessed_sha256']=digest
    text=torch.load(ROOT/'upstream/ckpt/infer_text_embedding_radar.pt',map_location='cpu',weights_only=False)
    text={k:v.float().cuda() for k,v in text.items()}
    logits={item:[] for item in items};features={};test_organs=meta['test_organ_names']
    shape=tuple(image.shape[2:]);interval=mod._get_scan_interval(shape,ROI,3,0.25)
    slices=dense_patch_slices(shape,ROI,interval)
    metrics['window_count']=len(slices);metrics['completed_windows']=0
    if args.single_window:
        mon.phase='single_window';guard(mon);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        patch=image[(slice(None),slice(None))+slices[0]]
        logits,p=forward(model,patch,logits,test_organs,text,features)
        metrics['stages'].append(stage_summary(start,'single_window',patch.shape))
        np.save(folder/'single_window_seg.npy',p.argmax(1).numpy().astype(np.uint8))
        dump(folder/'single_window_scores.json',{k:float(np.concatenate(v).mean(0)[1]) for k,v in logits.items() if v})
        dump(folder/'single_window_features.json',features)
        metrics['completed_windows']=1;metrics['status']='single_window_pass';return
    voxels=int(np.prod(shape));guard(mon,extra_cpu=voxels*(37*4+4+1)+int(np.prod(ROI))*37*4)
    accum=torch.zeros((1,37)+shape,dtype=torch.float32,device='cpu')
    count=torch.zeros((1,1)+shape,dtype=torch.float32,device='cpu')
    for wi,sl in enumerate(slices):
        guard(mon);mon.phase=f'window_{wi+1}';torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        ix=(slice(None),slice(None))+sl;patch=image[ix]
        logits,p=forward(model,patch,logits,test_organs,text,features)
        accum[ix].add_(p);count[ix].add_(1)
        del p
        metrics['stages'].append(stage_summary(start,mon.phase,patch.shape));metrics['completed_windows']=wi+1
        dump(folder/'progress.json',metrics)
    guard(mon);mon.phase='cpu_stitch';start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if count.min().item()<1: raise RuntimeError('Uncovered voxels in window count')
    seg=torch.empty((1,1)+shape,dtype=torch.uint8)
    for d in range(0,shape[0],8):
        block=accum[:,:,d:d+8];block.div_(count[:,:,d:d+8].clamp_min_(1))
        seg[:,:,d:d+8]=block.argmax(1,keepdim=True).to(torch.uint8)
    del accum,count,block;gc.collect()
    # Release inactive window allocations before possibly larger organ crops.
    torch.cuda.empty_cache()
    metrics['stages'].append(stage_summary(start,'cpu_stitch',seg.shape))
    np.savez_compressed(folder/'segmentation_preprocessed.npz',segmentation=seg.numpy())
    labels,counts=torch.unique(seg,return_counts=True)
    metrics['segmentation_voxels']={str(int(k)):int(v) for k,v in zip(labels,counts)}
    pad=transforms.DivisiblePadd(keys=['image','label'],k=32,mode='constant',constant_values=0,method='end')
    metrics['fallback_crops']=[];metrics['organ_status']={}
    # Preserve upstream per-finding fallback iteration (even repeat attempts).
    for item,values in logits.items():
        organ=item.split('_')[0];oid=dataset.organs.index(organ)
        if not len(values):
            mask=seg.eq(oid+1)
            if not mask.any():
                metrics['organ_status'][organ]='not_detected_in_stitched_segmentation';continue
            guard(mon);mon.phase='fallback_'+organ;torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            crop,mask=mod.center_crop(image,mask,crop_size=ROI)
            mask=mask.float();mask[mask==1]=oid+1
            padded=pad({'image':crop[0],'label':mask[0]});patch=padded['image'][None]
            del mask,padded
            print('FALLBACK',organ,list(patch.shape),flush=True)
            logits,p=forward(model,patch,logits,test_organs,text,features,skip=oid);del p
            stage=stage_summary(start,mon.phase,patch.shape);metrics['stages'].append(stage);metrics['fallback_crops'].append(stage)
            metrics['organ_status'][organ]='scored' if organ in features else 'detected_but_unscored'
            dump(folder/'progress.json',metrics)
    row={'file_name':fid};scored=0
    for item,values in logits.items():
        value=float(np.concatenate(values).mean(0)[1]) if values else np.nan
        if values:
            if not np.isfinite(value): raise RuntimeError('Non-finite finding score')
            scored+=1;metrics['organ_status'][item.split('_')[0]]='scored'
        row[f'{item} ({dataset.english_mapping[item]})']=value
    pd.DataFrame([row]).to_csv(folder/'findings.csv',index=False,encoding='utf-8-sig')
    dump(folder/'organ_features.json',features)
    metrics['scored_findings']=scored;metrics['total_findings']=len(items);metrics['status']='complete'

def main():
    p=argparse.ArgumentParser();p.add_argument('--precision',choices=['smoke','fp32','fp16','bf16'],required=True)
    p.add_argument('--gpu-budget-gib',type=float,help='Optional allocator cap; not a simulation of all GPU memory overhead')
    p.add_argument('--single-window',action='store_true');p.add_argument('--tag',default='');args=p.parse_args()
    name=args.tag or (args.precision+('_window' if args.single_window else '_full'))
    if pathlib.Path(name).name!=name or ':' in name:raise ValueError('Invalid tag')
    folder=EXPERIMENT/'runs'/name
    if folder.exists():raise FileExistsError('Use a new tag; previous measurements are preserved: '+str(folder))
    folder.mkdir(parents=True);mon=Monitor(folder);mon.start();start=time.perf_counter();metrics={'status':'started','stages':[]}
    try:
        if args.precision=='smoke':smoke(folder);metrics['status']='smoke_pass'
        else:run(args,folder,mon,metrics)
    except Exception as e:
        metrics['status']='failed';metrics['error']=repr(e);metrics['traceback']=traceback.format_exc()
        print(metrics['traceback'],flush=True)
    finally:
        metrics['wall_seconds']=time.perf_counter()-start
        metrics['resources']=mon.close();dump(folder/'metrics.json',metrics)
        print('FINAL',json.dumps(metrics,ensure_ascii=False),flush=True)
    return 0 if metrics['status'] in ('complete','single_window_pass','smoke_pass') else 1

if __name__=='__main__':sys.exit(main())
