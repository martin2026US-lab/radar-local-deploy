# Benchmark protocol and interpretation

The README leads with mean absolute score difference. Maxima, frequencies and failed attempts remain part of this release.

Primary comparison: 138 CTs with all four configurations successfully measured under the final runtime. Each candidate has 19,718 finite score pairs against same-case GPU FP32/CPU stitching. N/A pairs are excluded and separately counted. MAE=sum of absolute differences /finite-pair count. This is not cross-entropy, training loss, calibrated clinical probability error or AUROC.

| Candidate | Score-pair MAE | Case-average MAE | 95th percentile | Maximum | >0.1 pairs / cases |
|---|---:|---:|---:|---:|---:|
| Ordinary FP16 | 0.0007467135 | 0.0007489715 | 0.0020994633 | 0.1783565283 | 14 / 6 |
| Compact FP16 | 0.0007467135 | 0.0007489715 | 0.0020994633 | 0.1783565283 | 14 / 6 |
| TensorRT FP16 | 0.0010181720 | 0.0010212862 | 0.0029265868 | 0.4911744446 | 31 / 8 |

The largest TensorRT adrenal disagreement had no adrenal fallback call in either saved run. A cause based solely on fallback routing is not established. Segmentation, boundary coverage, token selection and the first window contributing an organ feature need a dedicated trace. No threshold was tuned to reduce these differences.

Full 200-case effective records retain multiple runtime revisions. Historical automatic 76% quota measurements are not pooled into the main speed/VRAM table. Prior failed attempts were not erased; successful retests have explicit provenance. 197 CTs completed all four variants; the performance headline uses final-configuration common 138. The first partial case across the last checkpoint is excluded from that paired-performance subset.

Ten current failures: 8 preflight rejections before any sliding window because planned new CPU memory 12.58–14.04 GiB plus 2 GiB reserve exceeded available RAM; 1 Compact run below 1.5 GiB available RAM after 10/20 windows;1 TensorRT run completed 20 windows and CPU stitch, then rejected 224×288×384. No CUDA OOM was recorded for these 10 failures. CPU stitching still materializes a 37-channel FP32 full-volume accumulator. Future work includes bounded CPU/file-backed storage while preserving summation order.

Effective success counts: FP32 198, ordinary FP16 197, Compact 197, TRT198. Finite values per successful vector range 102–146. Successful vectors contain 641 N/A positions for FP32, 629 each for FP16/Compact, 632 for TRT, distinct from 292/438/438/292 absent positions due to failed tasks. Missing values are never converted to negative labels or zeros.

Memory: recursive actual worker PIDs, Windows dedicated/shared GPU counters sampled about 1 s, process-tree RSS/private commit, separate device-wide NVML. Table medians summarize per-CT sampled peaks, not the absolute run maximum. In common 138, maximum shared usage was 9.37 GiB for FP32, 9.26 for ordinary FP16, 0.205 for Compact, 0.080 for TRT. Shared memory is not universally eliminated. RSS/shared accounting may overlap; do not sum them.

FP32 uses streamed workspaces, ordinary FP16 retains its original visual path. Precision alone does not determine memory. Compact reduced checkpoint size and typical VRAM but was not consistently faster than ordinary FP16. Fresh-process initialization, CPU work, engine switches and diagnostics are included. FP32/Compact have more allocator instrumentation than the other two variants.

Data: pinned AbdomenAtlas 3.0 Mini snapshot f94cd3f4, 100 random + 100 coverage-enriched CTs. Source leakage was not excluded; this is engineering comparison, not independent clinical validation. CTs, patient identifiers, original reports, patient tables, segmentations and per-patient predictions are not republished. Aggregate 146-finding results and known-limitation counts are included.

No new clinical label study, model training or FP16 calibration was performed. The earlier export experiment used an exposed development CT and also contained INT8 calibration artifacts. Those calibration arrays are excluded, and no independent validation is claimed from that development data.

The retained FP32 workspace source still carries its original pre-test “prototype” header. It is kept byte-identical for provenance; the frozen diagnostic and continuation records provide its subsequent validation. This does not imply that the new portable CLI was tested on the cohort.
