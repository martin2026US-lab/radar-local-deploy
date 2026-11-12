"""Portable fixed146 NIfTI entry point. --validate-only does not import torch or run inference."""
import argparse,csv,hashlib,json,os,pathlib,re,sys,tempfile,time,types
PACKAGE=pathlib.Path(__file__).resolve().parent
PROFILES={'fp32':'gpu_fp32_cpu_stitch','fp16':'gpu_fp16_cpu_stitch','compact-fp16':'compact_fp16_crop_v2','trt-fp16':'trt_fp16_crop_v2'}
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def required(profile):
    common=['upstream/ckpt/infer_text_embedding_radar.pt']
    if profile in ['fp32','fp16']:
        return common+['models/checkpoint_radar_pretrain.pth','models/bert-base-chinese/config.json','models/bert-base-chinese/pytorch_model.bin']
    if profile=='compact-fp16':return common+['models/radar_inference_only_v1.pth','models/radar_inference_only_v1.manifest.json','models/bert-base-chinese/config.json']
    return common+['models/int8_experiment/scoring_fp32.pth','models/int8_experiment/vision_fp16.plan','models/int8_experiment/vision_fp16.manifest.json','models/trt_dynamic/vision_dynamic_fp16.plan','models/trt_dynamic/vision_dynamic_fp16.manifest.json']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--asset-root',type=pathlib.Path,required=True)
    p.add_argument('--ct',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path)
    p.add_argument('--tag',default=time.strftime('%Y%m%d_%H%M%S'));p.add_argument('--assume-mm',action='store_true')
    p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    assets=args.asset_root.resolve();manifest=dict(read(PACKAGE/'assets_manifest.json'),**read(PACKAGE/'external_assets.json'))
    errors=[]
    for key in required(args.profile):
        target=assets/key
        if not target.is_file():errors.append('Missing asset: '+key)
        elif sha(target)!=manifest[key]['sha256']:errors.append('SHA256 mismatch: '+key)
    if errors:raise SystemExit('\n'.join(errors)+'\nUse tools/fetch_assets.py for this profile; modified engines need updated trusted manifests.')
    if args.validate_only:
        print(json.dumps(dict(profile=args.profile,assets_verified=True,model_execution=False)));return 0
    if not args.ct or not args.output:p.error('--ct and --output are required for inference')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.tag):p.error('--tag must be an ASCII identifier')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for key in ['TEMP','TMP','HF_HOME','TRANSFORMERS_CACHE','TORCH_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR']:
        target=output/'cache'/args.tag/key;target.mkdir(parents=True,exist_ok=True);os.environ[key]=str(target)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
        RADAR_ASSET_ROOT=str(assets),RADAR_OUTPUT_ROOT=str(output),MODEL_ROOT=str(assets/'models'),CONFIGS_ROOT=str(assets/'models'))
    allocator='backend:native,max_split_size_mb:128' if args.profile=='trt-fp16' else 'backend:native'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']=allocator;os.environ['PYTORCH_ALLOC_CONF']=allocator
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(PACKAGE/'runtime'))
    import torch,numpy as np,nibabel as nib
    if not torch.cuda.is_available():raise RuntimeError('These four configurations require CUDA; CPU stitching is not CPU-only inference')
    import run_inference as base
    # run_inference imports the vendored RADAR module using its pinned local path.
    sys.path.insert(0,str(PACKAGE/'runtime'))
    q=types.ModuleType('query_radar');q.ROOT=assets;q.torch=torch;q.read_json=read;q.sha256=sha;q.dump=base.dump;sys.modules['query_radar']=q
    if args.profile=='fp32':
        from fp32_workspace_runtime import install
        install(base)
    elif args.profile=='compact-fp16':
        loader=base.load_module
        def compact_loader():
            mod=loader()
            from stream_vision import StreamVisionBranch
            mod.VisionBranch=StreamVisionBranch
            return mod
        base.load_module=compact_loader;builder=base.build_model
        def compact_builder(mod,precision,folder):
            from compact_runtime import verified_checkpoint
            checkpoint,_=verified_checkpoint()
            return builder(mod,precision,folder,checkpoint_path=checkpoint,pretrained_text=False)
        base.build_model=compact_builder
    elif args.profile=='trt-fp16':
        from trt_builder import build_model
        base.build_model=build_model
    if args.profile in ['compact-fp16','trt-fp16']:
        from stream_vision import upsample_to_cpu_chunks
        def streamed_forward(model,patch,logits,test_organs,text,features,skip=None):
            x=patch.to(device='cuda',dtype=torch.float32)
            logits,prob=model.forward_test_win(x,None,logits,test_organs,text,features,None,skip_organ=skip)
            if not torch.isfinite(prob).all():raise RuntimeError('Nonfinite segmentation probabilities')
            return logits,upsample_to_cpu_chunks(prob,x.shape[2:])
        base.forward=streamed_forward
    loader=base.load_module;input_info={}
    scratch=output/'input_work';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=args.tag+'_',dir=scratch) as tmp:
        raw=nib.load(args.ct.resolve())
        if len(raw.shape)!=3:raise ValueError('Input must be a single3D NIfTI CT')
        unit=raw.header.get_xyzt_units()[0]
        if unit!='mm' and not(unit=='unknown' and args.assume_mm):raise ValueError('Input spacing must be declared mm; --assume-mm explicitly permits an unknown unit')
        if not np.isfinite(raw.affine).all() or not np.isfinite(np.asanyarray(raw.dataobj)).all():raise ValueError('Nonfinite CT input')
        orientation=nib.orientations.ornt_transform(nib.orientations.io_orientation(raw.affine),nib.orientations.axcodes2ornt(('L','A','S')))
        raw=raw.as_reoriented(orientation)
        if not np.allclose(raw.affine[:3,:3],np.diag(np.diag(raw.affine[:3,:3])),atol=1e-4):raise ValueError('Oblique CT is unsupported by frozen preprocessing')
        canonical=pathlib.Path(tmp)/'input_las.nii.gz';nib.save(raw,canonical)
        input_info=dict(original_sha256=sha(args.ct),orientation='LAS after lossless axis permutation/flip',original_unit=unit,assume_mm=args.assume_mm)
        def input_loader():
            mod=loader();parent=mod.DataFolder
            class InputFolder(parent):
                def __init__(self,*unused):
                    super().__init__(str(canonical.parent));self.img_paths=[str(canonical)]
            mod.DataFolder=InputFolder;return mod
        base.load_module=input_loader
        original_run=base.run
        def run(call_args,folder,monitor,metrics):
            try:return original_run(call_args,folder,monitor,metrics)
            finally:metrics.update(input=input_info,profile=PROFILES[args.profile],release='v0.1.0',CPU_stitch=True,release_wrapper_cohort_retested=False)
        base.run=run
        sys.argv=['run.py','--precision','fp32' if args.profile=='fp32' else 'fp16','--tag',args.tag]
        result=base.main()
        folder=output/'runs'/args.tag
        if (folder/'findings.csv').exists():
            with (folder/'findings.csv').open(encoding='utf-8-sig',newline='') as f:row=next(csv.DictReader(f))
            values=[]
            for r in read(PACKAGE/'finding_dictionary.json'):
                value=row[r['display']];values.append(None if not value or value.lower()=='nan' else float(value))
            assert len(values)==146
            base.dump(folder/'scores_146.json',dict(keys=[r['key'] for r in read(PACKAGE/'finding_dictionary.json')],scores=values,profile=args.profile))
        return result
if __name__=='__main__':raise SystemExit(main())
