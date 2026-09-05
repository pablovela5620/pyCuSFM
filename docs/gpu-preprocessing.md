# Moving the image preprocessing onto the GPU

The `raco` and `tensorrt` feature backends used to spend more host time turning a
decoded frame into a tensor than the GPU spent detecting keypoints in it. This
records what was moved, why it was moved that way, what it is worth, and what is
left.

Measured on this host: RTX 5090, CUDA 13 driver 580, TensorRT 10.13.3.9, OpenCV
5.0.0, over the 226 Galileo frames (`data/r2b_galileo/**/*.jpeg`, 1920x1200) and
50 KITTI 06 frames (1226x370). Reproduce with
`PYTHONPATH=. pixi run -e colsfm python -m tools.audit.gpu_preprocess_bench`.

**Read the numbers with the contention caveat below.** Every stage figure here
was taken while another process held the same GPU at 50-95 % utilisation, and the
spread between repeats of one configuration reached 1.7x. The equivalence results
are exact and unaffected; the timings are indicative.

## What the host used to do, per image

```python
bgr_hwc = cv2.imread(str(path), cv2.IMREAD_COLOR)          # 4.2 ms
resized = cv2.resize(bgr_hwc, (1920, 1200), cv2.INTER_LINEAR)  # 2.4 ms, an identity on Galileo
network_chw = np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1), np.float32)
network_chw /= np.float32(255.0)                            # 16.5 ms for the two lines
```

then `np.concatenate` of eight of those into a fresh 210 MB buffer (24.6 ms per
image, on the main thread) and a pageable 27.6 MB host-to-device copy (2.15 ms).
The serial per-image profile is `docs/raco-speed-investigation.md` §4.

## The option chosen, and why

The task offered two: prepend the preprocessing to the ONNX graph and rebuild the
detector engine (A), or write a CUDA kernel through NVRTC (B). What is
implemented is neither exactly — it is **A's idea without A's rebuild**:
`colsfm.tensorrt_runtime.build_preprocess_engine` authors a four-layer network
with TensorRT's own graph API and builds it into a separate engine.

```
image_u8 [batch, H, W, 3] uint8   ->  Cast(float32)
                                  ->  Transpose(0, 3, 1, 2)
                                  ->  Gather(axis=1, [2, 1, 0])   # BGR to RGB
                                  ->  Div(255)                    -> image [batch, 3, H, W] float32
```

Four reasons to prefer it over graph surgery on `aliked.onnx` and
`raco-aliked-b1-16.onnx`:

* **No rebuild of either detector engine.** Those cost about 205 s each and the
  blob's `aliked_fp16_10_13_3_9_sm_12_0.engine` ships in the repo; prepending
  nodes would have replaced it with a locally-built one. The preprocessing engine
  builds in **1.6 s** and is cached under `data/cusfm_models/` by input size,
  batch ceiling, TensorRT version and compute capability.
* **No `onnx` dependency, and no cross-environment step.** The `colsfm`
  environment has TensorRT but not `onnx` (that is in `raco`), so option A as
  written needed a script run in another environment to produce a graph that the
  first environment then consumes. The TensorRT graph API is already imported
  here.
* **It is written once and used by both backends**, at whatever batch each one
  runs, because the profile is `1..batch_size` rather than baked into a graph.
* **It is bit-exact.** FP16 is deliberately *not* enabled on this engine. Widening
  a uint8 is exact and `x / 255.0f` is one correctly-rounded IEEE division, so the
  tensor the detector receives is the same tensor `numpy` used to build — asserted
  with `np.array_equal`, not a tolerance
  (`tests/colsfm/test_features_trt_gpu_preprocess.py`).

Option B was not needed. It would have been the fallback had TensorRT refused a
`uint8` network input; it did not — TensorRT 10.13 accepts one as long as a
`Cast` is the first consumer, which is how the network is written.

### The three other changes that came with it

1. **The batch is staged in one page-locked buffer, not concatenated.**
   `GpuPreprocessor` owns a pinned `[max_batch, H, W, 3]` uint8 buffer; the main
   thread copies each decoded frame into its slot (0.5 ms) instead of allocating
   and filling a 210 MB float32 array (24.6 ms per image). The buffer is safe to
   refill after the detector's `run` returns, because that call synchronises the
   shared stream.
2. **The identity `cv2.resize` is skipped.** Galileo's frames are already
   1920x1200 and OpenCV does not shortcut that case; it copies 6.9 MB per image.
   `INTER_LINEAR` at 1:1 is an identity, so nothing changes.
3. **Host output buffers are reused.** `TensorRTSession(..., reuse_output_buffers=True)`
   keeps one host array per output binding instead of a fresh `np.empty` per call,
   whose first-touch page faults cost about 1 ms per megabyte of descriptors. The
   flag is off by default because it invalidates the previous call's arrays; the
   feature backends turn it on because they copy what they keep before the next
   call. `colsfm.matching_trt` is untouched.

## Before and after

`gpu_preprocessing=False` restores the host conversion and the `np.concatenate`,
so both columns come from the same binary in the same session. `HEAD` is the
unmodified code loaded from `git show HEAD:` into the same process, which is the
only way to charge changes 1-3 above to the right column.

| backend | HEAD | host preprocessing (this branch) | GPU preprocessing | speed-up vs HEAD |
| --- | --- | --- | --- | --- |
| raco, batch 8 | 4.66 s | 4.20 s | **2.90 s** | **1.61x** |
| tensorrt, batch 1 | 9.38 s | 7.00 s | 7.21 s | 1.30x |

226 images; best of two runs per configuration. Spread across repeats under
contention: raco host 4.20-7.31 s, raco GPU 2.74-2.90 s, stock host 6.77-7.73 s,
stock GPU 6.91-9.67 s.

**The batched backend is where the win is, and that is not an accident.** At
batch 8 the staged upload is one 55 MB pinned transfer replacing eight pageable
27.6 MB ones, and one preprocessing launch covers eight images. At batch 1 the
stock backend pays a second engine launch and a second `set_input_shape` per
image to save a single 21 MB upload, and the measured result is a wash inside the
noise. The stock backend's improvement over HEAD is real but comes from the
staging and the buffer reuse, not from the preprocessing move.

### Per-image breakdown, Galileo, single-threaded

| step | HEAD | now | where it runs |
| --- | --- | --- | --- |
| `cv2.imread` | 4.2 ms | 4.2 ms | CPU, in the 8-thread pool |
| `cv2.resize` | 2.4 ms | 0.0 ms | skipped (identity at 1920x1200) |
| BGR to planar RGB float32, `/255` | 16.5 ms | 0.0 ms | moved into the engine |
| `np.concatenate` into the batch | 24.6 ms | 0.5 ms | pinned slot copy, main thread |
| host-to-device bytes | 26.4 MiB | **6.6 MiB** | pinned instead of pageable |

Device side, batch of eight, milliseconds per image including the transfer:

| path | ms/image |
| --- | --- |
| upload float32 + detector engine | 14.8 |
| upload uint8 + preprocessing engine + detector engine | **12.3** |

The 2.5 ms difference is the transfer saving net of the preprocessing engine's own
cost, on a GPU that another process was using at the same time.

## End to end, against the blob

```
pixi run -e colsfm python -m colsfm run --input-dir data/r2b_galileo \
    --output-dir data/cusfm_runs/galileo_gpuprep/cusfm \
    --features-backend raco --matching-backend raco
pixi run -e colsfm python -m colsfm.bench_cli --dataset galileo \
    --run-a data/cusfm_runs/galileo_blobref/cusfm \
    --run-b data/cusfm_runs/galileo_gpuprep/cusfm --input-dir data/r2b_galileo
```

| metric | raco baseline (NOTES.md) | with GPU preprocessing |
| --- | --- | --- |
| feature extraction stage | 7.52 s | **4.77 s** |
| total | 15.9 s | 15.98 s |
| registered images | 224/226 | 224/226 |
| mean reprojection error | 1.500 px | 1.500 px |
| ATE vs ground truth | 5.19 mm | **5.17 mm** |
| acceptance bounds | 4/4 | **4/4** |

The reconstruction is the same reconstruction: rig poses agree with the blob to
0.39 mm RMSE and the ATE moves by 0.02 mm, which is the reordering of two
floating-point sums, not a different answer.

**The total did not move, and that is contention, not the change.** Matching took
5.05 s in this run against the baseline's ~2.3 s, on the same GPU another process
was using; extraction gave back 2.75 s and matching took 2.75 s back. Extraction
is the stage this change touches and it is the stage that moved.

Note that the pipeline's extraction stage (4.77 s) is larger than `extract_raco`'s
own reported time (2.90 s): the stage also creates the database, resolves the
image list and reads the keypoint counts back.

## What the stage is made of afterwards

At 2.90 s for 226 raco images — 12.8 ms per image of wall time — against roughly
10 ms per image of detector compute, **the extraction stage is now engine-bound**.
The CPU half is 4.2 ms of decode per image spread over eight threads (0.5 ms of
wall time per image), the staging copy is 0.5 ms on the main thread, and the
transfer has fallen from 2.15 ms to about 0.2 ms. There is no large host cost left
to remove; the next real saving would have to come from the detector itself or
from a decoder that is not `libjpeg-turbo`.

That is the same conclusion `docs/gpu-jpeg-decode.md` reaches from the other
direction, and it changes the case for nvJPEG: with the conversion and the
transfer already gone, torchcodec's remaining prize is the 4.2 ms decode, spread
across eight threads, in exchange for 1.66 GiB of PyTorch and an `ffmpeg` 9 pin.
It is worth less now than it was when that document was written.

## KITTI, the small-image case

50 frames of `data/kitti/06/image_0`, 1226x370 PNGs upscaled to the engine's
1920x1200. No full KITTI run.

| step | ms/image |
| --- | --- |
| `cv2.imread` (PNG) | 2.6 |
| `cv2.resize` 1226x370 to 1920x1200 | 0.5 |
| host float32 conversion (removed) | 3.7 |
| staging copy into pinned | 0.6 |
| upload float32 + engine, batch 8 | 18.5 / 11.5 |
| upload uint8 + preprocess + engine, batch 8 | **16.4 / 9.1** |

The two device figures are the same measurement under heavy and light contention
from the neighbouring process; the *difference* is stable at 2.1-2.4 ms per image,
which is the number the change owns.

The saving is the same 2.1 ms per image, because it is a property of the *engine's*
input size, not the file's: KITTI is upscaled to 1920x1200 before anything is
transferred either way. The conversion removed is smaller (3.7 ms against
Galileo's 16.5 ms) only because it runs on a warm, small source that the resize
has just written. KITTI's real problem is unchanged and is not this one: a fixed
1920x1200 profile over a 1226x370 source is 5.1x the pixels of the input
(`docs/raco-speed-investigation.md` §5).

## Equivalence

| check | result |
| --- | --- |
| `GpuPreprocessor.run` vs `to_network_bchw`, 3 Galileo frames | **bit-identical** (`np.array_equal`) |
| raco keypoints, GPU vs host preprocessing, 3 frames | **identical**, 2048 per image |
| raco descriptors, minimum cosine over all keypoints | **1.000** (gate: > 0.999) |
| stock keypoints, GPU vs host preprocessing, 3 frames | **identical** |
| stock descriptors, minimum cosine | **1.000** |

The control that makes the first row meaningful is that it runs on real JPEGs
rather than random bytes: a channel order reversed once too often is invisible in
a symmetric test and obvious in a real one.

## Contention caveat

Another process held this GPU at 50-95 % utilisation throughout. Two effects are
visible and neither is a property of the change:

* Stage times vary by up to 1.7x between repeats of the same configuration, which
  is why the table reports the best of two and prints the spread.
* TensorRT intermittently failed a run-time allocation
  (`Myelin ... run-time allocation failed (47)`) with 26 GB of the 32 GB free,
  and the same failure reproduces on unmodified `HEAD` code. It is contention for
  the stream-ordered allocator, not a regression, and it is why the benchmark and
  pipeline invocations here are wrapped in a retry.

Re-measure on an idle GPU before quoting these stage numbers anywhere that
matters. The equivalence results need no such caveat.

## Several input sizes at once

`colsfm.features_raco` gained a native-resolution path: with
`FeatureOptions.native_resolution` (the default) each image runs at its own size
rounded up to 32 rather than stretched to 1920x1200, so one extraction pass can
touch more than one input shape. Nothing in `GpuPreprocessor` had to change for
that — the engine was already keyed by `(height, width, max_batch)` in
`preprocess_engine_path` and built on demand — but two consequences are worth
recording.

**Each new size costs one build, once.** A four-layer network builds in a second
or two, and the file is cached beside the detector engines under
`preprocess_bgr_u8_<h>x<w>_b<batch>.engine`. `data/cusfm_models` now holds a
`1216x1920` (Galileo at the network grid) and a `384x1248` (KITTI) next to the
older `1200x1920`. The build is invisible on a full dataset and very visible on
a short one: KITTI's first 50 frames measured 21.97 ms per image with the build
inside the stage and 4.87 ms per image over 200 frames without it. Warm-cache
numbers are the only ones worth quoting.

**`extract_raco` keeps one preprocessor per size for the length of the run.**
Images are grouped by rounded size before batching, and each group's
`GpuPreprocessor` is entered on the run's `ExitStack`, so a mixed-camera set
pays one build and one page-locked staging buffer per distinct size rather than
one per batch. The staging buffer is `max_batch * height * width * 3` bytes of
pinned host memory, which is 45 MB at batch 8 and 1216x1920 — worth knowing
before pointing this at a set with many distinct camera sizes.

## Native resolution picks an engine, not just a size

Running every image through the shape-dynamic engine cost Galileo 1.8x. Galileo
is 1920x1200, which *is* the dynamic profile's maximum, so it gains nothing from
a smaller input and loses two things: one optimisation profile has to pick
tactics for everything from 256x256 to 1216x1920, and the host pays a
`cv2.resize` from 1200 to 1216 that the fixed-shape path skips (the fixed graph
resamples to 1216 inside itself, from the 1920x1200 it declares).

`select_raco_engine` therefore decides per size group, and
`FeatureOptions.raco_engine="auto"` is the default. The rule is one comparison:
put the fixed engine's declared input through `network_size_for`, the same
rounding the dynamic path uses, and a group that equals it goes to the fixed
engine — today only 1216x1920. The comparison has to be on the rounded group
rather than on 1200x1920, because no image ever produces a 1200 group; 32 does
not divide it. `"fixed"` and `"dynamic"` force one engine, which is what a
measurement of the two wants.

| Galileo, 226 keyframes, warm engines | stage | per image |
| --- | --- | --- |
| `raco_engine="auto"` (the default) | **2.64 s** | 11.7 ms |
| `raco_engine="fixed"` | 2.62 s | 11.6 ms |
| `raco_engine="dynamic"` | 4.96 s | 22.0 ms |

| KITTI 06, 200 frames, warm engines | stage | per image |
| --- | --- | --- |
| `raco_engine="auto"` (dynamic at 1248x384) | **1.09 s** | 5.4 ms |
| `raco_engine="dynamic"` | 1.09 s | 5.5 ms |
| `native_resolution=False` (the stretch) | 2.30 s | 11.5 ms |

So `auto` is the fixed number on Galileo and the dynamic number on KITTI, which
is the whole claim.

**The 1.8x was not build contention.** The dynamic engine had been built while
other workers held the GPU, and `docs/raco-speed-investigation.md` §3b records
that a contended build picks worse tactics permanently. Deleting both the
engine and the TensorRT timing cache and rebuilding on an idle GPU (251 s)
produced an 8.9 MiB engine against the old 9.2 MiB — different tactics — that
runs Galileo in 4.960 s against the contended engine's 4.919 s. The profile
width and the host resize are the whole explanation; nothing here is worth a
rebuild.
