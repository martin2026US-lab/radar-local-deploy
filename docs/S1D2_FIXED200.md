# S1D2 ordinary windows and FP16 fallback · fixed 200 CTs

With the original trained RADAR weights, **200/200 whole-CT requests completed** and all 200 passed the structural checks. The reference was the historical same-backend GPU FP32 run with CPU stitching on the identical cases. On the same 28,559 finite score pairs, selectively retaining FP32 reduced mean absolute score difference by **65.6%** relative to ordinary FP16. **Nine scores across five CTs still differed by more than 0.1**.

| Fixed-200 comparison against FP32 | Ordinary FP16 | S1D2 + FP16 fallback |
|---|---:|---:|
| Mean absolute score difference | 0.0009162493 | 0.0003151614 |
| Scores with absolute difference > 0.1 | 38 / 28,559 | 9 / 28,559 |
| Largest absolute score difference | 0.379382 | 0.331178 |

## Where the scores diverged

Of the 38 positions above 0.1 under ordinary FP16, 33 moved to 0.1 or below and five stayed above it. Four additional positions crossed the threshold under S1D2, all involving the portal vein. Letters group findings from the same CT. The criterion uses full-precision scores before rounding.

| CT | Finding / 征象 | FP32 | S1D2 | Absolute difference | Also > 0.1 under ordinary FP16 |
|---|---|---:|---:|---:|---|
| A | GIST · 胃_间质瘤（gist） | 0.447678 | 0.185901 | **0.261777** | Yes |
| B | Sacrum / Osteitis · 骶骨_骨炎 | 0.886219 | 0.774988 | **0.111231** | Yes |
| C | Portal vein / Thrombosis · 门静脉_栓塞 | 0.029728 | 0.153636 | **0.123908** | No |
| D | Portal vein / Dilatation · 门静脉_增宽 | 0.190203 | 0.521382 | **0.331178** | No |
| D | Portal vein / Hypertension · 门静脉_高压 | 0.096400 | 0.419188 | **0.322789** | No |
| D | Portal vein / Thrombosis · 门静脉_栓塞 | 0.012076 | 0.239981 | **0.227905** | No |
| E | Large bowel / Appendicolith · 大肠_阑尾粪石 | 0.358499 | 0.189634 | **0.168865** | Yes |
| E | Large bowel / Appendicitis · 大肠_阑尾炎 | 0.387839 | 0.275127 | **0.112712** | Yes |
| E | Large bowel / Perforation · 大肠_肠穿孔 | 0.472825 | 0.368851 | **0.103974** | Yes |

Each CT has 146 output positions. Of the 29,200 positions, 28,559 had two finite scores and entered the MAE. Both runs returned N/A at 635 positions. Six stomach findings from one CT were N/A in FP32 and finite in S1D2. Zero positions had a finite FP32 score and an S1D2 N/A. Missing values were excluded from arithmetic and retained in the coverage counts.

## What ran

- On ordinary windows, the first encoder block, the final two decoder blocks and the final 37-class segmentation convolution used FP32. The remaining ordinary-window visual computation used FP16. Every fallback visual forward used FP16. The finding-scoring head used FP32 throughout.
- The runner preserved the original network, probability averaging order, first eligible organ features, fallback crop selection, internal masks and tokens. After fallback scoring it skipped the outer probability upsampling and CPU transfer whose result the caller discarded. All **1,018** ordinary-window probability outputs were retained and **753** fallback outer copies were skipped.
- The local CUDA runner completed six gated cases, then the remaining 194 with resource guards active. The mean whole-CT request was **19.07 s**, median **15.83 s**. The timer excluded worker imports, the bridge check and final weight verification. Model residency differed from historical runs, so these timings do not establish a speedup. The lowest sampled free memory on the whole GPU was **0.245 GiB**. In earlier separate stress tests, the guard stopped a full-FP32 fallback variant before two large crops; those attempts produced no complete score vector.

## How to read this result

This fixed set of AbdomenAtlas 3.0 Mini CTs was selected for engineering comparisons (100 random and 100 coverage-enriched). FP32 provides a numerical reference. Clinical labels, diagnostic accuracy and sensitivity were outside this experiment. The primary four-profile comparison in the v0.1.0 benchmark continues to use its frozen 138 jointly successful scans, with its own failure ledger. The S1D2 measurements were obtained with the local research runner. The v0.1.0 CLI lists its four packaged profiles separately.

The machine-readable results include full-precision outliers and the hashes of the local comparison tables. The following excerpts show the tested decoder precision loop and the caller-side copy decision. In the ordinary-window branch the runner invokes `decode(dec, skips, (2, 3), trace)`; fallback takes the original decoder path under FP16 autocast.

```python
def decode(decoder, skips, fp32_indices, trace):
    # Match the frozen decoder loop, including the auxiliary supervision outputs.
    x = skips[-1]
    outputs = []
    last = len(decoder.stages) - 1
    for s, (upsample, fusion) in enumerate(zip(decoder.transpconvs, decoder.stages)):
        residual = skips[-(s+2)]
        fp32 = s in fp32_indices
        with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
            up = upsample(x.float() if fp32 else x)
            up_dtype = str(up.dtype)
            joined = torch.cat((up, residual.float() if fp32 else residual), 1)
            del up
            join_dtype = str(joined.dtype)
            x = fusion(joined)
            del joined
        expected = "torch.float32" if fp32 else "torch.float16"
        assert up_dtype == join_dtype == str(x.dtype) == expected
        output_dtype = None
        if decoder.deep_supervision or s == last:
            with torch.autocast("cuda", dtype=torch.float16, enabled=s != last):
                score = decoder.seg_layers[s](x.float() if s == last else x)
            output_dtype = str(score.dtype)
            assert output_dtype == ("torch.float32" if s == last else "torch.float16")
            outputs.append(score)
            del score
        trace.append({
            "stage_index": s, "skip_index": len(skips) - (s+2),
            "skip_source_dtype": str(residual.dtype), "upsample_dtype": up_dtype,
            "concat_dtype": join_dtype, "fusion_dtype": str(x.dtype),
            "segmentation_output_dtype": output_dtype,
        })
    outputs.reverse()
    return outputs if decoder.deep_supervision else outputs[0]
```

```python
logits,prob=model.forward_test_win(x,None,logits,test_organs,text,features,None,skip_organ=skip)
if not torch.isfinite(prob).all():raise RuntimeError('Nonfinite visual probabilities')
cpu = upsample_to_cpu_chunks(prob,x.shape[2:]) if skip is None else None
row['output_probability_upsampled'] = cpu is not None
```
