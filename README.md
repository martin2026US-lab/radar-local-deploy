# RADAR Local Deploy — v0.1.0 research preview

Four community deployment configurations of [RADAR](https://github.com/alibaba-damo-academy/damo-radar), preserving CPU stitching and fixed 146-finding scoring. No retraining, structural pruning, added LLM, or new clinical-accuracy claim.

On **138 jointly successful CT scans under the final runtime configuration**, mean absolute score differences from same-case GPU FP32 + CPU stitching were **0.000747 for FP16 and Compact FP16**, and **0.001018 for TensorRT FP16**. The corresponding 95th-percentile absolute differences were 0.002099 and 0.002927. These are engineering agreement metrics, not training loss, AUROC, diagnostic accuracy, or proof that every finding is preserved.

| Configuration | Mean absolute score difference | Whole-CT median | Median per-CT dedicated GPU peak | Median per-CT RSS peak |
|---|---:|---:|---:|---:|
| GPU FP32, CPU stitch, streamed workspace | reference | 38.17 s | 5.70 GiB | 4.90 GiB |
| GPU FP16, CPU stitch | 0.000747 | 30.60 s | 6.08 GiB | 5.05 GiB |
| Compact FP16, CPU stitch | 0.000747 | 37.40 s | 4.34 GiB | 5.23 GiB |
| TensorRT FP16, CPU stitch | 0.001018 | 34.36 s | 3.13 GiB | 4.54 GiB |

Compact and ordinary FP16 produced exactly identical score vectors, including missing-value positions, on all **197 jointly successful cases** in the effective 200-case record. Compact retains the same 693 model tensors; it removes training state, reducing the checkpoint from 1,566,049,482 to 522,203,623 bytes. It is not a pruned network or a separately trained model. CPU stitching still requires a CUDA GPU for visual inference.

## Scope and known limitations

- Full effective record: **200 CTs / 800 configuration slots, 790 successes and 10 failures**, 197 CTs completing all four configurations. Successes: FP32 198/200, ordinary FP16 197/200, Compact 197/200, TensorRT 198/200. Prior failed attempts remain in the local audit, with separate successful retest provenance.
- The ten current failures affect three CTs: **8 CPU RAM preflight rejections, 1 system-RAM protection failure, 1 TensorRT shape-range rejection**. They were not recorded as CUDA OOMs. The full 37-channel FP32 CPU stitching accumulator can exceed 10–12 GiB. RAM safeguards remain enabled; no per-process GPU memory quota is set.
- Rare large score disagreements remain. In the 138-case cohort, maximum absolute differences were **0.178357 for ordinary/Compact FP16** and **0.491174 for TensorRT**. Differences above 0.1 occurred in 14/19,718 valid score pairs across 6/138 cases for ordinary/Compact FP16, and 31/19,718 pairs across 8/138 cases for TensorRT. The largest TensorRT example was adrenal adenoma: FP32 0.654870 versus TensorRT 0.163695. Its cause requires further tracing; no clinical interpretation is established. Historical inherited FP16 results also contain a maximum difference of 0.379382.
- Fixed146 positions do not mean 146 valid values per CT. Successful vectors contain 102–146 finite scores; unavailable organ scores remain null/N/A. Failed attempts are not converted to negative labels or zero-filled.
- Data: AbdomenAtlas 3.0 Mini snapshot f94cd3f4, 100 random + 100 coverage-enriched cases. Source independence and absence of upstream exposure are not asserted. AMOS20 had already been exposed during development. This does not reproduce the original 146-condition clinical study.
- Historical measurements used different runtime revisions/quotas. The resource table uses only the final-configuration common 138 subset. The benchmark comes from frozen Windows adapters. This portable CLI adds asset/input/output routing and has static/import/hash checks; its new entry point was **not rerun on the 200-case cohort**.
- Tested hardware: RTX 5070 12 GB / SM 12.0, approximately 23.20 GiB OS-visible RAM, Python 3.11, PyTorch 2.7.1+cu128, TensorRT 10.13.3.9. Times include fresh process, imports/loading, preprocessing, inference, CPU stitching, scoring, saved files and diagnostics. FP32/Compact had more allocator logging than the other configurations.

## Install and run

[Source on GitHub](https://github.com/martin2026US-lab/radar-local-deploy) · [Converted artifacts on Hugging Face](https://huggingface.co/Martin123456qwe/radar-local-deploy)

Use an isolated environment and explicit model/output directories. This preview does not modify the original GUI or default backend.

~~~powershell
git clone https://github.com/martin2026US-lab/radar-local-deploy.git
cd radar-local-deploy
python -m venv D:/RADAR-env
& D:/RADAR-env/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
& D:/RADAR-env/Scripts/python.exe -m pip install -r requirements.txt
& D:/RADAR-env/Scripts/python.exe tools/fetch_assets.py --profile compact-fp16 --destination D:/RADAR-assets
& D:/RADAR-env/Scripts/python.exe run.py --profile compact-fp16 --asset-root D:/RADAR-assets --validate-only
& D:/RADAR-env/Scripts/python.exe run.py --profile compact-fp16 --asset-root D:/RADAR-assets --ct D:/data/ct.nii.gz --output D:/RADAR-output --tag case001
~~~

Profiles: fp32, fp16, compact-fp16, trt-fp16. Fetch assets for your selected profile first. FP32/ordinary FP16 fetch the full source checkpoint and BERT initialization weights directly from the official model repository; these are not duplicated here. Compact uses the inference-only checkpoint with all trained tensors preserved. The fixed-finding path rejects unexpected BERT execution. The existing local advanced custom-query interface and GUI are outside this preview; no small LLM is introduced.

Input is one 3D NIfTI CT. Lossless axis permutation/flip to LAS precedes frozen preprocessing. Oblique input is rejected. Millimetre spacing is required; --assume-mm explicitly permits an unknown unit after the user has verified it. Existing output tags cannot be overwritten. Results include findings.csv, scores_146.json, segmentation and resource metrics.

For TensorRT, install the separately licensed official runtime from requirements-tensorrt.txt and fetch the trt-fp16 assets. Plans are hardware/runtime-specific. The static plan handles 96×256×384 windows; the dynamic plan uses MIN 96×256×384, OPT 128×288×416, MAX 192×384×512, spatial multiples of 32. Depth 224 failed this range. Other hardware or larger crops require explicit rebuilding, not silent resizing or precision fallback:

~~~powershell
python tools/build_trt_fp16.py --onnx D:/RADAR-assets/models/vision_fp32.onnx --output-prefix D:/rebuilt/vision_dynamic_fp16
~~~

The fetch utility only downloads assets needed for the selected inference profile. Separately download models/vision_fp32.onnx from the Hub for rebuilding. Rebuilt plan hashes/manifests must be trusted and added to a local assets_manifest.json: the strict validator intentionally rejects unrecognized artifacts. Different tactics can change numerical/resource results; a rebuilt plan does not automatically inherit this benchmark.

## Results and attribution

See [benchmark details](docs/BENCHMARK.md), [aggregate JSON](benchmark/aggregate_results.json), [all 146 finding comparisons](benchmark/per_finding146.json), and [source provenance](SOURCE_PROVENANCE.json). No CTs, patient tables, original clinical text, per-patient score vectors, credentials, complete environments, CUDA/TensorRT DLLs or drivers are included.

This is an unaffiliated, non-commercial community research release. The RADAR-derived model and code retain **CC BY-NC-SA 4.0**. Third-party BSD/MIT/Apache notices are preserved; NVIDIA runtimes retain their own terms. The NC restriction means this is not unrestricted commercial/OSI software. No endorsement by the original authors, Alibaba/DAMO or NVIDIA is claimed. Research use only; not a certified clinical product.

Upstream: Qi Zhang et al., *An expert-level generalist AI for abdominal CT diagnosis*, Science (2026), [doi:10.1126/science.aec6129](https://doi.org/10.1126/science.aec6129). See [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and [CITATION.bib](CITATION.bib).
