"""Explicit hardware-specific FP16 engine rebuild; no calibration or inference."""
import argparse,hashlib,json,pathlib,time
def main():
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=pathlib.Path,required=True);p.add_argument('--output-prefix',type=pathlib.Path,required=True)
    p.add_argument('--static',action='store_true');p.add_argument('--min',nargs=3,type=int,default=[96,256,384]);p.add_argument('--opt',nargs=3,type=int,default=[128,288,416]);p.add_argument('--max',nargs=3,type=int,default=[192,384,512]);a=p.parse_args()
    import tensorrt as trt,torch
    logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(net,logger)
    if not parser.parse_from_file(str(a.onnx)):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_flag(trt.BuilderFlag.FP16);config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256<<20);config.builder_optimization_level=3
    shapes={'min':[1,1,*a.min],'opt':[1,1,*a.opt],'max':[1,1,*a.max]}
    if a.static:
        net.get_input(0).shape=[1,1,96,256,384];shapes={k:[1,1,96,256,384] for k in shapes}
    else:
        profile=builder.create_optimization_profile();profile.set_shape('image',shapes['min'],shapes['opt'],shapes['max']);config.add_optimization_profile(profile)
    started=time.perf_counter();serialized=builder.build_serialized_network(net,config)
    if serialized is None:raise RuntimeError('TensorRT build failed')
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True);target=pathlib.Path(str(a.output_prefix)+'.plan')
    if target.exists():raise FileExistsError('Use a new prefix; previous engine must remain preserved')
    target.write_bytes(serialized)
    manifest=dict(sha256=hashlib.sha256(serialized).hexdigest(),bytes=target.stat().st_size,mode='fp16',calibration=None,
        min_shape=shapes['min'],opt_shape=shapes['opt'],max_shape=shapes['max'],tensorrt=trt.__version__,GPU=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()),build_seconds=time.perf_counter()-started,onnx_sha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest())
    pathlib.Path(str(a.output_prefix)+'.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('New engine saved. Update a trusted local assets_manifest.json before validation/inference; rebuilding does not reproduce the published benchmark automatically.')
if __name__=='__main__':main()
