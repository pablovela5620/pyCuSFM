# Why the RaCo-ALIKED + LightGlue+ backend does not look 3x faster

**Question.** `colsfm`'s `raco` backend runs fabio-sim's RaCo-ALIKED and
LightGlue+ through TensorRT, and the blog post that motivated it advertises "up
to 3x". On Galileo the stage times barely move: extraction 7.52 s against the
stock TensorRT backend's 7.01 s, matching 2.31 s against 2.91 s. Did we build it
wrong, or did we read the claim wrong?

**Answer, in one line.** Both, in different places. The device compute *is*
2.5x faster than the stock blob pipeline (24.7 ms/pair against 61.9 ms/pair on
this 5090), which is a bigger gap than the blog ever claims for anything; the
stage wall time hides it because 69% of the serial per-image budget is host-side
buffer shuffling that has nothing to do with the model. Separately, we deployed
at 1920x1216, a resolution 1.4x larger than the biggest configuration the blog
measured and which the blog explicitly calls dominated.

Measured on an RTX 5090, TensorRT 10.13.3.9, FP16, K=2048, in the repo's
`colsfm` Pixi environment. Scratch scripts: `tools/audit/raco_speed/`.

---

## 1. The claim as stated, and as we read it

### What the post says

Every speedup number in the post is against **FP16 `torch.compile()` of the same
RaCo-ALIKED-LightGlue+ pipeline**, on an RTX 4080 Laptop:

> "The half-precision `torch.compile()` implementation, used as the PyTorch
> performance baseline."

> "To emphasize, TensorRT is already faster than the baseline (FP16
> `torch.compile`) by about 1.3x. Sol doubled that."

> "Even without any further optimization, TensorRT already outperforms the
> compiled PyTorch baseline across most tested configurations, with a median
> speedup of 1.38x."

> "Across all 28 configurations, the optimized pipeline substantially
> outperforms the half-precision `torch.compile()` baseline, with a median
> inference speedup of 2.67x."

> "July 27, 2026 - Update: ... boosting the median speedup to 3x the baseline."

### What that implies for a TensorRT deployment

We do not deploy `torch.compile`. We deploy TensorRT. The right comparison for
us is Sol-optimized TensorRT against **naive** TensorRT, which is the ratio of
the two medians:

| comparison | median factor |
| --- | --- |
| naive TRT FP16 vs FP16 `torch.compile` | 1.38x |
| Sol-optimized TRT FP16 vs FP16 `torch.compile` | 2.67x (3x after the July 27 update) |
| **Sol-optimized TRT vs naive TRT (what an already-TensorRT deployment gains)** | **1.93x (2.2x after the update)** |

So the ceiling on "what the optimisations buy a TensorRT user" is about 2x, not
3x, and the post says so in as many words ("Sol doubled that").

### Is "3x against the stock ALIKED engine" a claim the post makes?

**No.** The post never benchmarks RaCo-ALIKED against stock ALIKED, SuperPoint,
DISK, or any other model. Its three compared implementations are all the *same*
pipeline: FP16 `torch.compile`, naive ONNX+TRT, Sol-optimized ONNX+TRT. Reading
the headline as "3x faster than the ALIKED + LightGlue engines in the cuSFM
blob" is our extrapolation, not the author's claim.

### The resolutions the claim covers

> "Image resolutions: 512^2, 768^2, 1024^2, and 1280^2 ... Keypoints per image:
> 512, 1024, 1536, 2048, 2560, 3072, and 3584"

and, on the Pareto frontier:

> "the low-latency 512^2, K=1024 configuration at 11.9 ms" ... "768^2, K=3584
> ... at 31.1 ms" ... "1024^2, K=3584 adds only 13 additional correct
> correspondences while increasing latency by more than 17 ms (diminishing
> returns), and **every 1280^2 configuration is dominated** by a faster
> configuration with equal or better correspondence yield."

Those latencies are **per image pair for the whole pipeline** (two images
extracted plus one match), median GPU time, excluding preprocessing.

`colsfm` runs at 1920x1216 = 2.33 Mpix. That is 1.42x the pixels of 1280^2
(1.64 Mpix) and 2.22x the pixels of 1024^2 (1.05 Mpix). We are operating one
step past the end of the measured range, in the region the post tells readers
not to use.

---

## 2. Our export against upstream's

`tools/export_cusfm_raco.py` builds `RaCoALIKED(num_keypoints=2048,
portable_deform_conv=True, topk_chunk_size=65536, ranker_mode=RankerMode.boundary)`
and calls `fuse_batch_norm()`. Checked against
`lightglue_dynamo` at the pinned rev `d12b4ba`
(`.pixi/envs/raco/lib/python3.12/site-packages/lightglue_dynamo/`):

### Optimisations we DO have (no divergence)

| Sol optimisation | upstream knob | our value | verdict |
| --- | --- | --- | --- |
| Hierarchical chunked TopK | `topk_chunk_size=65536` | 65536 | same as upstream default |
| BatchNorm folding | `pipeline.fuse_batch_norm()` | called on the extractor | same |
| Portable deformable conv (GridSample rewrite) | `portable_deform_conv=True` | `True` | same as upstream default |
| Integer-space FP16 overflow fix | `custom_translation_table={aten.div.Tensor_mode: onnx_op.Div}` | identical function, copied | same |
| Boundary reranking, R=256 | `ranker_mode` | `RankerMode.boundary` | **same as `auto` resolves to** |
| Candidate factor P | `candidate_multiplier` | default 2, capped at 3840 | same |
| `ranker_scale` | `ranker_scale` | default 1.0 (required by boundary mode) | same |
| Native SELU / logit-space ordering | baked into the model code at `d12b4ba` | inherited | same |

`RankerMode.auto.resolve(2048, 1200, 1920)` returns `boundary` — the branch
`1024 <= num_keypoints < 2560 and min(height, width) >= 768`. Our hard-coded
`boundary` is exactly upstream's own choice for this operating point. **Item 5's
"try boundary reranking" is already done; there is nothing to switch on.**

### Divergences

1. **Two graphs, not one pipeline graph.** Upstream exports
   `Pipeline(extractor, matcher)` — one graph, `images (2B,3,H,W)` in, `matches
   (S,3)` out, no host round-trip. We export an extractor and a matcher
   separately and route between them through a COLMAP SQLite database:
   `descriptors -> D2H -> numpy -> pycolmap.FeatureDescriptorsFloat.to_bytes ->
   SQLite -> read back -> H2D`, 1 MB per image each way. This is not
   removable: COLMAP's mapper and `verify_matches` need persisted features. It
   does mean the blog's per-pair latency is not the thing our stage measures.
2. **Fixed spatial shape.** Only the batch axis is dynamic
   (`torch.export.Dim("batch", 1..16)`); H and W are frozen at 1200x1920.
   Upstream's `-h 0 -w 0` marks `divisor * height_factor` and `divisor *
   width_factor` dynamic. Consequence on KITTI: 1226x370 frames (0.45 Mpix) are
   stretched to 2.33 Mpix, 5.1x the pixels, for nothing.
3. **An extra in-graph resize.** 1200 is not a multiple of 32, so
   `CusfmExtractor.extract_batch` runs `F.interpolate` to 1216 inside the graph,
   on top of the CPU `cv2.resize` that already produced 1200x1920.
4. **Matcher batch is one pair, statically.** `CusfmLightGlue.forward` takes
   four `[1, n, c]` tensors and `torch.cat`s them to `[2, n, c]`; dim 0 is a
   constant. Upstream's `build_dynamic_shapes` marks `image_shapes[0] = 2 *
   pair_count`. Note the blob itself ships `lightglue_aliked_b16.onnx`, so a
   batched matcher is a shape upstream and the blob both already support.
5. **Extra matcher work.** Our adapter pads both sides to `sym_max(n0, n1)` and
   scatters the sparse `(S,3)` matches into a dense `[n1, 2]` buffer with zero
   scores as the sentinel, because the frozen cuSFM buffer manager cannot take a
   data-dependent output shape. Upstream returns the sparse tensor directly.
6. **Scores are synthetic.** Boundary reranking materialises no dense score map,
   so we emit `arange(2048, 0, -1)/2048`. Correct, and documented, but it means
   the `min_score` gate and the `argsort` in `extract_raco`'s post-loop are
   pure overhead (measured below: 0.03 ms/image, so harmless).
7. **Builder settings.** `colsfm.tensorrt_runtime.build_fp16_engine` uses
   `builder_optimization_level = 3` and an 8 GiB workspace. ORT's TensorRT EP
   uses its own defaults. Not a measured difference; listed for completeness.

None of 1-7 disables a Sol optimisation. The extractor graph we run is the
optimised one.

---

## 3. Device benchmark on this 5090

`tools/audit/raco_speed/bench_device.py --mode engines`, 30 timed calls after 5
warm-ups, median. **Trust the compute-only column**: the "call" column includes
pageable host copies and Python, and was taken while another benchmark held ~18
cores, so it is noisy (a quieter earlier run gave `raco B=8` at 126.1 ms/call;
the compute column agreed to within 1%).

All at 1200x1920 input, K=2048, FP16.

| configuration | ms / call | ms / call, compute only | ms / image, compute only | ms / pair, compute only |
| --- | --- | --- | --- | --- |
| raco extractor, B=1 | 22.08 | 16.18 | 16.18 | 32.4 |
| raco extractor, B=4 | 90.70 | 47.68 | 11.92 | 23.8 |
| **raco extractor, B=8** | 170.79 | **80.66** | **10.08** | **20.2** |
| raco extractor, B=16 | — | — | — | out of memory (8.0 GB tactic request) |
| stock blob ALIKED, B=1 | 31.09 | 27.00 | 27.00 | 54.0 |
| **raco LightGlue+, 1 pair** | 4.69 | **4.53** | — | **4.53** |
| stock blob LightGlue, 1 pair | 17.23 | 7.88 | — | 7.88 |

Whole-pipeline device compute per image pair, at 1920x1200 and K=2048:

| pipeline | ms / pair |
| --- | --- |
| **RaCo-ALIKED + LightGlue+ (ours, extractor at B=8)** | **24.7** |
| stock blob ALIKED + LightGlue | **61.9** |
| ratio | **2.51x** |

For scale, the blog's Pareto points on an RTX 4080 Laptop: 11.9 ms/pair at
512^2/K=1024, 31.1 at 768^2/K=3584, 48.4 at 1024^2/K=3584. We get 24.7 ms/pair
at 2.22x the pixels of the 1024^2 point, on a much larger GPU. Nothing here says
our engines are slow.

### The release end-to-end pipeline graph: not measured, and the reason is a bug

`raco_aliked_lightglue_pipeline_k2048.onnx` (release v3.0, 65.8 MB) through ONNX
Runtime 1.29.0 in the `bench` environment
(`tools/audit/raco_speed/bench_release_ort.py`). Three attempts, none of which
produced a TensorRT number:

1. **Four configurations in one process.** Reached **68.8 GB RSS** across ~18
   cores after 25 minutes without emitting a result. Killed under the 60 GB
   budget rule.
2. **Bounded: one configuration per process** (1024^2, 1 pair),
   `trt_max_workspace_size = 8 GiB`, engine and timing cache enabled. Held
   **8.05 GB RSS**, completed in ~16 minutes, and reported **4317 ms per pair**.
   That is not a GPU number. The log explains it:

   ```
   [E:onnxruntime:Default, provider_bridge_ort.cc:2367 TryGetProviderInfo_TensorRT]
   ... RegisterTensorRTPluginsAsCustomOps ... Please install TensorRT libraries
   EP Error ... when using [('TensorrtExecutionProvider', {...})]
   ```

   **`ort.get_available_providers()` in the `bench` environment lists
   `TensorrtExecutionProvider`, but the provider cannot load.** The environment
   *does* ship `site-packages/tensorrt_libs`, but `bench` has no
   `[feature.bench.activation.env]` block, so — unlike `colsfm`, `colsfm-caspar`
   and the default feature, all of which append
   `$CONDA_PREFIX/lib/python3.12/site-packages/tensorrt_libs` to
   `LD_LIBRARY_PATH` — the `.so` files are on no search path. ONNX Runtime then
   fell all the way through to the CPU provider, and 4317 ms/pair is the CPU
   time for this graph.
3. **Retry with `LD_LIBRARY_PATH` patched by hand** in the `pixi run` shell:
   the TensorRT provider still failed to register **and the CUDA provider then
   failed too** (`TryGetProviderInfo_CUDA`, "Please install all dependencies as
   mentioned in the GPU requirements page"), so prepending the wheel's
   `tensorrt_libs` is not the whole fix. Killed at the time box.

**So the head-to-head — release pipeline graph against our two engines at the
same shape — is unfinished, and the blocker is an environment defect in
`pixi.toml`, not the graph.** `[feature.bench]` needs the same
`activation.env.LD_LIBRARY_PATH` the other GPU features carry, and the CUDA-EP
failure needs a look after that. Fixing it is a prerequisite for any future
MegaDepth-1500 benchmarking in that environment too, since that is what `bench`
exists for.

The FP16 `torch.compile` baseline (item 3d) was not attempted: the `raco`
environment's torch is CPU-only by design (`pixi.toml`,
`[feature.raco.pypi-dependencies]`, `index = ".../whl/cpu"`), so it needs a new
CUDA torch environment, well over the 10-minute setup allowance.

### Host transfer cost (`--mode transfer`)

One extractor batch of 8 at 1200x1920:

| strategy | bytes | ms / call | ms / image | effective GB/s |
| --- | --- | --- | --- | --- |
| pageable float32 (what we do) | 210.9 MB | 17.17 | 2.15 | 12.3 |
| pinned float32 | 210.9 MB | 6.14 | 0.77 | 34.4 |
| pinned uint8 (upload raw, normalise in-graph) | 52.7 MB | 1.42 | 0.18 | 37.1 |

---

## 4. Where the Galileo extraction stage time actually goes

`--mode stage`, 226 Galileo JPEGs at 1920x1200, batch 8, single-threaded so each
phase is attributable (the real stage overlaps the CPU phases across 8 worker
threads, so these are serial costs, not the 7.52 s wall time).

| phase | total (s) | ms / image | share |
| --- | --- | --- | --- |
| **`np.concatenate` of the batch** | 5.567 | **24.63** | **32.6%** |
| engine call (H2D + execute + D2H + output alloc) | 4.593 | 20.32 | 26.9% |
| **JPEG decode (`cv2.imread`)** | 2.648 | **11.72** | 15.5% |
| BGR->RGB planar float32 (`ascontiguousarray(...transpose)`) | 2.518 | 11.14 | 14.7% |
| divide by 255 | 1.207 | 5.34 | 7.1% |
| **`cv2.resize`** | 0.543 | **2.40** | 3.2% |
| post-processing (score gate, argsort, keypoint mapping) | 0.007 | 0.03 | 0.04% |
| **serial total** | **17.083** | **75.58** | |

**Decode + resize share, for the GPU-decode worker's baseline: 3.191 s over 226
frames = 14.12 ms/image serial, 18.7% of the serial budget.** Decode alone is
11.72 ms/image, resize alone 2.40 ms/image.

Of the 20.32 ms/image engine call, 10.08 ms is device compute and 2.15 ms is the
pageable H2D; the remaining ~8 ms is D2H into freshly-`np.empty`'d host arrays
(page faults on a 1 MB descriptor block per image), `set_input_shape`, and
Python.

### The top three costs, and whether each is a mistake

1. **`np.concatenate` into a fresh 210 MB buffer, 24.63 ms/image — a mistake.**
   `extract_raco` builds every batch with
   `np.ascontiguousarray(np.concatenate([item.network_bchw for item in batch]))`.
   Each worker thread already allocated a `[1,3,1200,1920]` float32 array
   (27.6 MB); the concatenate allocates a *third* 210 MB buffer and copies into
   it, on the main thread, serialised with the engine call. Writing each
   prepared image straight into a slot of one preallocated pinned batch buffer
   removes the allocation, the copy, and the pageable-transfer penalty at once.
2. **The engine call's 10.2 ms/image of non-compute, 20.32 ms/image total — half
   a mistake.** Pageable host memory (2.15 ms/image against 0.18 ms/image for
   pinned uint8) and a fresh `np.empty` per output per call are both avoidable.
   `TensorRTSession.run` also synchronises on every call, so no batch's copies
   overlap the next batch's compute.
3. **CPU float32 conversion, 11.14 + 5.34 = 16.48 ms/image — a mistake.** The
   BGR->RGB planar transpose to float32 and the `/255` produce the 27.6 MB
   tensor that then costs 2.15 ms to push over PCIe. Uploading the decoded
   uint8 HWC frame (7 MB) and doing the permute/scale as the graph's first ops
   removes all three costs.

And one small one: **`cv2.resize` is an identity on Galileo**, whose frames are
already 1920x1200, yet it still copies 6.9 MB per image (2.40 ms). Skipping it
when the source shape already matches is free.

None of the top costs is engine-side. Nothing here is "per-call context
creation" or "verification dominating"; the post-processing loop (score gate,
`argsort`, keypoint mapping) is 0.04% of the budget, so the synthetic-score gate
noted in divergence 6 costs nothing.

The matching stage was not profiled phase-by-phase inside the box. Its budget is
bounded by the engine numbers: 331 pairs x 4.53 ms device = 1.50 s of the
measured 2.31 s stage, so at most 0.8 s is SSC, `pycolmap.verify_matches`, the
database round-trip and Python — matching is not where the money is on Galileo.

---

## 5. Verdict

### Our error (real, measurable, fixable)

| item | cost measured | note |
| --- | --- | --- |
| Batch built with `np.concatenate` into a new 210 MB buffer | 24.63 ms/image serial, on the main thread | 32.6% of the serial per-image budget |
| CPU float32 CHW conversion + `/255` before upload | 16.48 ms/image serial | plus 4x the PCIe bytes |
| Pageable rather than pinned host staging | 2.15 ms/image vs 0.77 (f32) / 0.18 (uint8) | 12.3 GB/s vs 37 GB/s |
| Identity `cv2.resize` on already-1920x1200 frames | 2.40 ms/image | pure copy |
| Fixed 1200x1920 engine profile on 1226x370 KITTI frames | 5.1x the pixels of the source | measured 19.6 ms/image against pycolmap ALIKED's 7.8 ms native |
| Matcher graph frozen at one pair | ~0.16 ms/pair of call overhead on Galileo | matters at RoboCap's pair count, not here |

### Expectation error (we mis-read the claim)

| item | numbers |
| --- | --- |
| We read "3x" as against a TensorRT deployment of a *different* model. The post's baseline is FP16 `torch.compile` of the *same* pipeline. | 2.67x (3x updated) vs `torch.compile`; the TensorRT-vs-TensorRT gain the post actually reports is 2.67/1.38 = **1.93x** (2.2x updated), and only against a naive export of the same model. |
| We compared *stage wall time* against a claim about *median GPU latency excluding preprocessing*. On Galileo the stage is host-bound: 55.2 of 75.6 ms/image serial is decode, layout and buffer copies. | device compute per pair **24.7 ms (ours) vs 61.9 ms (stock blob)** = **2.51x** — larger than the post claims for anything, and invisible in the stage number. |
| We deployed at 1920x1216 = 2.33 Mpix. The post's range tops out at 1280^2 = 1.64 Mpix and calls every 1280^2 configuration dominated. | 1.42x the pixels of the largest measured, dominated point; 2.22x the pixels of 1024^2. |

### Also found, not part of the question

The `bench` environment's TensorRT execution provider does not load
(section 3). `[feature.bench]` is missing the `activation.env.LD_LIBRARY_PATH`
entry that `colsfm`, `colsfm-caspar` and the default feature all carry, so
`onnxruntime-gpu` advertises `TensorrtExecutionProvider` and then silently falls
back to CPU — 4317 ms/pair where a TensorRT number was expected. Any benchmark
run in `bench` up to now should be re-checked for the same silent fallback.

### Blog overstates

Nothing found. The post's numbers are internally consistent, its baseline is
stated three times in plain words, and the 1.38x naive-TensorRT median is
published alongside the 2.67x headline so the TensorRT-to-TensorRT ratio is
derivable from the post itself. The one thing a reader can be led astray by is
the abstract's bare "up to 3x" without the baseline attached — and the body
attaches it immediately.

---

## 6. Ranked changes, with expected gain

Gains are on the measured serial per-image budget (75.58 ms/image) unless
stated; the stage overlaps CPU work across 8 threads, so wall-time gains are
smaller than serial-cost gains except for phases on the main thread.

| # | change | Galileo (1920x1200) | KITTI (1226x370) | confidence |
| --- | --- | --- | --- | --- |
| 1 | **Preallocated pinned batch buffer**: workers write into slots of one pinned `[8,3,H,W]` staging array; drop `np.concatenate`. | removes 24.63 ms/image of *main-thread* serial cost and 1.38 ms/image of transfer. Stage 7.52 s -> ~5.0 s expected. | same shape of gain | high — both costs measured directly |
| 2 | **Upload uint8 HWC, normalise in-graph**: move the BGR->RGB permute, float cast and `/255` into the ONNX graph. | removes 16.48 ms/image of worker CPU and drops H2D from 2.15 to 0.18 ms/image | same | high — transfer measured; graph-side cost is a permute+scale on 7 MB |
| 3 | **Right-sized engine profile** (dynamic H/W, or a second static profile per camera geometry). | none: 1920x1200 is already native. Removes only the in-graph 1200->1216 `interpolate`. | 2.33 Mpix -> 0.45 Mpix. Detector convs, the chunked TopK and the candidate pool all scale with pixels; **19.6 ms/image -> ~5 ms/image, ~4x**, is the expectation. | medium on the KITTI number — extrapolated from the pixel ratio, not measured; the fixed K=2048 head does not shrink |
| 4 | **Skip `cv2.resize` when the source already matches the network shape.** | 2.40 ms/image of worker CPU | not applicable (KITTI always resizes) | high |
| 5 | **Overlap copies with compute**: per-call `cudaStreamSynchronize` in `TensorRTSession.run` serialises every batch's D2H against the next batch's H2D; reuse output host buffers instead of `np.empty` per call. | part of the ~8 ms/image of engine-call overhead that is neither compute nor H2D | same | medium — the 8 ms is measured, its split is not |
| 6 | **Re-export the matcher with a dynamic pair axis** (upstream's `-b 0`; the blob already ships `lightglue_aliked_b16.onnx` as precedent). | 331 pairs x ~0.16 ms/pair of call overhead = ~0.05 s. Negligible here. | matters at RoboCap's pair count, not at Galileo's | medium |
| 7 | **Ranker mode.** No change available: our export already uses `RankerMode.boundary`, which is what upstream's `auto` resolves to at K=2048 and min(H,W) >= 768. | 0 | 0 | high — read from `lightglue_dynamo/config.py` at `d12b4ba` |

Items 1, 2 and 4 together remove 43.5 ms of the 75.6 ms/image serial budget and
1.97 ms/image of PCIe time, without touching a single model weight.

GPU JPEG decode (11.72 ms/image, the largest remaining worker-thread cost) is
deliberately out of scope here; a separate worker is measuring torchcodec's
nvJPEG path against the 14.12 ms/image decode+resize baseline recorded above.

---

## Reproducing

```
pixi run -e colsfm python -m tools.audit.raco_speed.bench_device --mode engines
pixi run -e colsfm python -m tools.audit.raco_speed.bench_device --mode transfer
pixi run -e colsfm python -m tools.audit.raco_speed.bench_device --mode stage
pixi run -e bench   python -m tools.audit.raco_speed.bench_release_ort   # one config; currently falls back to CPU, see section 3
```

The release graph lands in `data/cusfm_models/raco_release/` (gitignored) via
`gh release download v3.0 --repo fabio-sim/LightGlue-ONNX --pattern 'raco_aliked_lightglue_pipeline_k2048.onnx'`.
