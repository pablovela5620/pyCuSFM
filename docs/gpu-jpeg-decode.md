# GPU JPEG decode (nvJPEG via torchcodec) against the CPU path colsfm runs today

Measured on this host: RTX 5090, CUDA 13 driver 580, torch 2.13.0 (cuda130),
torchcodec 0.16.0, OpenCV 5.0.0, over the 226 Galileo frames
(`data/r2b_galileo/**/*.jpeg`, 1920x1200) and 226 KITTI frames
(`data/kitti/06/image_0/*.png`, 1226x370). Median of 3 passes after a warm-up
pass. Reproduce with `pixi run -e decode-bench decode-bench`
(`tools/audit/decode_bench.py`).

## What the baseline actually is

`colsfm.features_trt.preprocess_image` decodes with **OpenCV**, not PIL:

```python
bgr_hwc = cv2.imread(str(path), cv2.IMREAD_COLOR)
resized = cv2.resize(bgr_hwc, (1920, 1200), interpolation=cv2.INTER_LINEAR)
network_chw = np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1), dtype=np.float32)
network_chw /= np.float32(255.0)
```

A plain stretch, no aspect-ratio preservation. Eight threads
(`DEFAULT_PREPROCESSING_WORKERS`) run it ahead of the GPU. The benchmark copies
this code rather than importing it, so the bench environment needs no pycolmap
or TensorRT.

Galileo's frames are **already 1920x1200**, so `cv2.resize` there is a copy and
the pixel-equivalence numbers below isolate the two decoders with no resampling
mixed in. KITTI's frames are genuinely upscaled.

## Results

| dataset | method | variant | ms/image |
| --- | --- | --- | --- |
| galileo | (a) cv2 decode only | single-thread | 12.94 |
| galileo | (a) cv2 full path + HtoD | 8 threads | **18.24** |
| galileo | (b) torchcodec CPU decode | single-thread | 6.49 |
| galileo | (c) nvJPEG CUDA, one at a time | batch 1 | 1.42 |
| galileo | (d) nvJPEG CUDA batched + GPU resize/convert | batch 8 | **1.29** |
| galileo | (d) nvJPEG CUDA batched + GPU resize/convert | batch 16 | 1.68 |
| galileo | (d) nvJPEG CUDA batched + GPU resize/convert | batch 32 | 1.96 |
| galileo | (e) as (d), threaded file readers | batch 8, 4 readers | 2.52 |
| galileo | (e) as (d), threaded file readers | batch 16, 4 readers | 1.50 |
| galileo | (e) as (d), threaded file readers | batch 32, 4 readers | 2.16 |
| galileo | (e) as (d), threaded file readers | batch 8, 8 readers | 2.31 |
| galileo | (e) as (d), threaded file readers | batch 16, 8 readers | 1.78 |
| galileo | (e) as (d), threaded file readers | batch 32, 8 readers | 1.51 |
| kitti | (a) cv2 decode only | single-thread | 6.10 |
| kitti | (a) cv2 full path + HtoD | 8 threads | 7.79 |
| kitti | (b) torchcodec CPU decode | single-thread | 3.21 |

**KITTI has no GPU row, and that is a property of the format.** nvJPEG is
JPEG-only, and torchcodec 0.16's `decode_png` takes no `device` argument at all
(only `decode_jpeg` does; `decode_image` takes none either — the release note's
"CUDA through nvJPEG" is reachable *only* through `decode_jpeg`). KITTI is
therefore CPU-only here. torchcodec's CPU PNG decoder is still 1.9x faster than
`cv2.imread`, which is a free win independent of any GPU.

**Threaded readers (e) do not help, and sometimes hurt.** The files are in the
page cache after the warm-up pass, so reading bytes is nearly free and the extra
threads only add contention and GIL churn. On a cold cache or a network volume
the ordering would likely invert; on this host, serial reads with batch 8 (d)
are the fastest configuration.

**Batch 8 beats batch 16 and 32.** Larger nvJPEG batches raise peak decode
throughput but also raise the per-batch allocation and the `torch.stack` /
`interpolate` working set; at 1920x1200x3 float32 a batch of 32 is 884 MB of
intermediate. Batch 8 also matches `colsfm.features_raco.DEFAULT_BATCH_SIZE`, so
no new batching parameter is needed.

### Where the 18.24 ms goes

Single-threaded sub-steps on Galileo, ms/image:

| sub-step | ms/image |
| --- | --- |
| `cv2.imread` decode | 12.94 |
| `cv2.resize` (identity at 1920x1200) | 1.67 |
| BGR to planar RGB float32, `/255` | 13.96 |
| host-to-device copy (synchronous) | 7.06 |
| **total, single-threaded** | **35.63** |

The thread pool folds 35.63 ms down to 18.24 ms of wall time. Note that the
*format conversion* (13.96 ms) is as expensive as the decode itself — a
transpose plus a `uint8`-to-`float32` widening over 6.9 M pixels — and that the
HtoD copy of a 27.6 MB float32 tensor costs another 7.06 ms.

Method (d) removes all four rows at once. nvJPEG writes planar CHW uint8
straight into device memory, and the widen, the `/255` and the (Galileo-identity)
resize all happen on the 5090 in the same pass. **There is no host-to-device copy
left**: the tensor is already where the engine wants it.

## Projected Galileo extraction-stage time

The stage is currently 7.5 s (raco backend) / 7.0 s (stock TensorRT backend) for
226 images, of which the engine is ~2.5 s. Measured preprocessing is
226 x 18.24 ms = **4.12 s**, which reconciles with those totals (4.12 + 2.5 =
6.62 s, the remainder being engine load, database writes and keypoint
post-processing). Method (d) at batch 8 costs 226 x 1.29 ms = **0.29 s**.

| backend | stage now | projected with (d) | saving |
| --- | --- | --- | --- |
| raco (batched RaCo-ALIKED) | 7.5 s | **~3.7 s** | 3.8 s (51%) |
| stock (blob's ALIKED) | 7.0 s | **~3.2 s** | 3.8 s (54%) |

Treat these as a floor, not a promise. nvJPEG and the TensorRT engine contend
for the same GPU: today decode overlaps engine execution because it runs on
otherwise-idle CPU threads, and after the move both sit on the 5090. Recovering
the overlap needs the decode on a separate CUDA stream from the engine, which
the current `colsfm.tensorrt_runtime` single-stream design does not do. Without
that, expect the two to serialise and the real figure to land above 3.7 s.

## Pixel equivalence

Five Galileo images decoded both ways, compared after resize in 0-255 units:

| statistic | value |
| --- | --- |
| max abs difference | 37.00 |
| mean abs difference | 0.1313 |
| median abs difference | 0.000 |
| p99 / p99.9 / p99.99 | 2.0 / 7.0 / 14.0 |
| fraction of pixels > 1 level | 4.28% |
| fraction of pixels > 5 levels | 0.19% |
| **luma (0.299R+0.587G+0.114B) max** | **7.73** |
| **luma mean / fraction > 1 level** | **0.029 / 0.02%** |

Control: comparing with the channel order reversed gives mean 12.19 (93x worse),
which confirms the RGB ordering is right and the residual is a real decoder
difference, not a bug.

**The difference is chroma, not luma.** Per channel the mean error is R 0.127,
G 0.083, B 0.185 — the green channel, which carries most of the luma, is the
cleanest, and the red/blue extremes are where 4:2:0 chroma upsampling differs
(libjpeg-turbo's "fancy" upsampling against nvJPEG's). Over 90% of pixels are
bit-identical. On the luma projection the worst pixel in the sample is off by
7.7/255 and only 0.02% of pixels move by more than one level.

**Would it perturb keypoints?** Probably very little, but this is not proof.
ALIKED consumes all three channels, so a 4% of pixels differing by one or two
levels can nudge sub-pixel refinement, and the differences concentrate exactly
at high-contrast edges where keypoints live. What is measured here is the input
tensor, not the output keypoints. Before adopting, run the engine over the same
226 frames both ways and compare keypoint repeatability, descriptor cosine
distance, and downstream match counts. That A/B is the gating experiment and it
has not been run.

## Dependency footprint

`torchcodec 0.16.0` is on conda-forge with a `cuda130_py312*` build that depends
on `pytorch * cuda130*` (>=2.13.0,<2.14) and `libnvjpeg`. **No PyPI wheels and no
`download.pytorch.org` index are needed** — the whole stack resolves from
conda-forge, which is why the `decode-bench` feature uses plain conda
dependencies. As with `pycolmap` in `[feature.colsfm.dependencies]`, the build
string is what selects the CUDA variant; a bare `pytorch` spec resolves to the
`cpu_*` build even with `cuda-version = "13.0.*"` present.

Adding torch + torchcodec on top of what `colsfm` already has costs **1.66 GiB
of downloads across 48 new packages**:

| package | download size |
| --- | --- |
| libtorch | 457.4 MiB |
| libmagma | 259.9 MiB |
| nccl | 228.3 MiB |
| libcufft | 172.1 MiB |
| mkl | 135.7 MiB |
| libnpp | 115.9 MiB |
| libcudss | 74.6 MiB |
| cuda-nvrtc | 65.2 MiB |
| triton, qt6-main, libopencv, ... (38 more) | ~190 MiB |
| pytorch (python bindings) | 26.2 MiB |
| **torchcodec itself** | **0.9 MiB** |

On disk the `decode-bench` environment is 7.0 GB against `colsfm`'s 6.6 GB. Note
that `nccl`, `libmagma`, `libcudss` and `libcufft` — over 700 MiB — are pulled in
by libtorch for training and linear algebra that a JPEG decoder never touches.

### The blocker for merging this into `colsfm`

`torchcodec 0.16.0 cuda130*` depends on **`ffmpeg >=9.0.1,<10`**.
`[feature.colsfm.dependencies]` pins **`ffmpeg = "8.*"`**, deliberately — "Keep
the tested 8.x major until the JPEG extraction path is validated on 9.x", the
same hold the default feature carries. Adding torchcodec to `colsfm` forces that
pin to 9.x and re-solves the environment that the NOTES.md decisions exist to
keep frozen. This is not a version-range nicety; it is the reason `decode-bench`
is a separate `no-default-feature` environment with its own solve group.

## Zero-copy handoff to TensorRT

Verified in `tools/audit/decode_bench.demonstrate_zero_copy_handoff`: a torch
CUDA tensor's `data_ptr()` is a plain integer device address, and the benchmark
reads from it with `libcudart`'s `cudaMemcpy` reached through `ctypes` — a
consumer that knows nothing about torch — and gets bytes identical to torch's
own view. TensorRT's `IExecutionContext.set_tensor_address(name, address)` takes
exactly this integer, which is what `cuda.bindings` already hands it in
`colsfm.tensorrt_runtime`. So there is **no marshalling cost** between a
torchcodec-decoded batch and the engine: method (d)'s output tensor can be bound
directly, provided the tensor is kept alive for the duration of the execution
and is `.contiguous()`.

Two ways to get there, both with real costs:

1. **Add torch + torchcodec to the `colsfm` environment.** Simplest data path
   (one process, one CUDA context, bind `data_ptr()` and go), but it costs the
   1.66 GiB above, forces the ffmpeg 8 -> 9 bump described above, and puts a
   second full CUDA library set (libtorch's bundled cuBLAS/cuFFT/cuDNN
   expectations) alongside the ones the frozen cuSFM binaries and TensorRT
   already resolve. NOTES.md already records CUDA library duplication as the
   hazard that keeps the solve groups apart, and `[feature.raco]` keeps PyTorch
   CPU-only for exactly this reason.
2. **A separate decode process or service.** Keeps `colsfm`'s solve untouched,
   but a device pointer is not valid across processes without CUDA IPC
   (`cudaIpcGetMemHandle`), which adds a handle exchange, a shared-memory
   control channel, and lifetime management of the exporting allocation. The
   engineering cost is well above the 3.8 s it would save on Galileo.

## Recommendation

**Do not adopt yet — but the measurement says the ceiling is real.** Specifically:

1. **The speedup is genuine and large.** 18.24 -> 1.29 ms/image is 14x on the
   preprocessing, and it halves the Galileo extraction stage on paper (7.5 s ->
   ~3.7 s). The mechanism is sound: nvJPEG removes the decode, the GPU removes
   the format conversion, and the batch lands in device memory so the 7.06 ms
   HtoD copy disappears entirely.

2. **The blocker is the environment, not the technique.** The ffmpeg 8 -> 9 bump
   plus 1.66 GiB and a second CUDA stack inside the environment that exists to
   stay frozen is a bad trade for 3.8 s on a 226-frame sample. Revisit when
   `colsfm` validates ffmpeg 9.x on its own schedule; the constraint disappears
   at that point and option 1 becomes cheap.

3. **Run the keypoint A/B before anything else.** The pixel difference is small
   and chroma-dominated (luma max 7.7/255, 0.02% of pixels above one level), but
   "the tensors are close" is not "the keypoints are the same". Decode the 226
   Galileo frames both ways, run the engine, and compare keypoint repeatability
   and match counts. If that comes back clean, the case for adoption is strong;
   if it does not, none of the above matters.

4. **Take the free CPU win now if a win is wanted today.** torchcodec's CPU JPEG
   decoder is 2.0x faster than `cv2.imread` on Galileo (6.49 vs 12.94 ms) and
   1.9x on KITTI PNG (3.21 vs 6.10 ms), with no GPU involved. But note that
   torchcodec's CPU decoder still arrives with the full torch dependency, so on
   footprint grounds this is only attractive if torch lands in the environment
   for other reasons.

5. **If the stage is re-optimised without torch, attack the conversion step.**
   13.96 ms/image of the 35.63 ms is the BGR-to-planar-float32 transpose, and
   7.06 ms is a synchronous HtoD copy of a float32 tensor. Uploading `uint8` HWC
   and doing the transpose, widen and `/255` in a small CUDA kernel (or as the
   first layers of the TensorRT graph) would recover most of those 21 ms for
   zero new dependencies. That is the higher-value change, and it is independent
   of nvJPEG.

## Scope note

The `decode-bench` pixi environment is audit-only: `no-default-feature`, its own
solve group, and referenced by nothing in `colsfm/`. Adding it left `pixi.lock`
purely additive (`git diff -w --stat pixi.lock` shows insertions only).
