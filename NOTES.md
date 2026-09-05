# NOTES — pixified pyCuSFM fork

Fork of [nvidia-isaac/pyCuSFM](https://github.com/nvidia-isaac/pyCuSFM), frozen at
upstream **`0c97a67`** ("Release 0.1.3: add github pages"). The `main` branch is byte-identical
to `upstream/main`; all pixi work lives on the `pixi` branch.

Goal: `pixi run demo` runs cuSFM on a real sample and shows it in Rerun, with **zero**
`pip`/`uv`/`conda` and **without** upstream's `setup.bash`.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **CUDA 13 binaries** (`pycusfm/x86_cuda13`) | Host is an RTX 5090 (Blackwell, sm_120), driver 580.173. TensorRT builds a native `sm_120` engine. No CUDA-12 fallback was needed. |
| 2 | **No `setup.bash`** | It only creates two symlinks. `command_runner.py:26` derives `lib_dir = dirname(binary_dir)/lib`, so passing `--binary_dir .../x86_cuda13/bin` finds `x86_cuda13/lib` for free. Two `ln -sfn` pixi tasks cover upstream's own CLI, which still defaults to `<package>/bin`. |
| 3 | **cuVSLAM skipped** (`skip_cuvslam=True`) | Both datasets already carry a trajectory. cuSFM's global BA *refines* it — that is the interesting operation, not re-deriving what we already have. |
| 4 | **Two rigs in one recording** | `rig_00` = input trajectory, `rig_01` = cuSFM-refined, rigidly aligned. Makes the BA correction visible instead of asserted. |
| 5 | **`ba_frame_type=vehicle_rig`, `optimize_extrinsics=False`** | exoego:v2 has a single `world_T_rig(t)` per rig, which is only meaningful if `rig_T_cam` stays fixed. Optimising extrinsics would silently invalidate the schema. |
| 6 | **`min_inter_frame_distance=0.0`** | The `isaac` default is 0.5 m. Galileo's whole trajectory is 0.66 m, so the default kept only 32 of 226 keyframes and produced 1275 points. At 0.0 we keep all 226 and get 4938 points. |
| 7 | **COLMAP model parsed by hand** | This env is a delicately balanced CUDA-13 solve with a TensorRT dependency override; pulling pycolmap's ceres/CUDA stack risks perturbing it for two file formats worth ~40 lines. cuSFM writes text by default. A deliberate exception to "do not hand-roll what exists" — revisit if binary models are ever needed. |
| 8 | **simplecv from the monorepo, not the standalone repo** | See gotcha 4. |
| 9 | **CPU video decode, not nvdec** | Measured; see gotcha 6. |

## Datasets

### `pixi run demo` — r2b_galileo (bundled)

Upstream's own sample: 8 PINHOLE cameras, 226 keyframes, 232 JPEGs, 69 MB, LFS-tracked in-repo.
Needs no network and no catalog, so it satisfies the fresh-clone gate unconditionally.
It also ships `ground_truth.txt`, which is the only independent accuracy reference here.

### `pixi run demo-robocap` — RoboCap segment `s00000021`

A head-worn 6-camera rig, frozen out of a Rerun catalog to `data/robocap/s00000021.rrd`
(477.5 MB, `base` + `slam` layers). 6 Kannala-Brandt fisheye cameras at 1920x1080, 4648 frames,
2 m 35 s, and a per-frame **basalt VIO** trajectory
(`source = "basalt VIT catalog driver (drive_catalog.py, non-deterministic, all threads, downscale 3)"`).

The rig reference is `imu_00`, so `world_T_rig` is `world_T_imu` and `rig_T_cam` is `imu_T_cam`.

**Rig geometry** (`rig_T_cam`, computed from the recording):

| Pair | Baseline | Angle between optical axes | Baseline vs camera X |
|---|---|---|---|
| **left_front ↔ right_front** | **86.2 mm** | **2.6°** | **1.3°** |
| left ↔ left_front | 126.1 mm | 87.9° | 40.4° |
| right_front ↔ right | 131.9 mm | 90.1° | 56.1° |
| left ↔ right | 239.7 mm | 168.0° | 86.2° |

Only `left_front↔right_front` is a real stereo pair. `left`/`right` point sideways (+X / −X)
while the front pair points forward (+Y); `left↔right` are back-to-back with no overlap.
**Exactly one `stereo_pair` is declared.** Declaring the others would hand cuVSLAM
plausible-looking baselines for non-rectifiable geometry.

`left_eye`/`right_eye` (`cam_02`/`cam_03`) point at the wearer's eyes. They are excluded from
the reconstruction and drawn as grey frusta, so the recording shows the real hardware without
implying those cameras contributed.

## Reproduction

| Run | Registered | Points | vs input estimate | ATE vs GT | Rig rigidity |
|---|---|---|---|---|---|
| galileo, upstream defaults (`demo-upstream`) | 28 / 226 | 1275 (uncoloured) | — | — | — |
| galileo, `demo` | **224 / 226** | **4938** | **1.7 mm** / 0.66 m | **3.9 mm** | **0.00 mm** |
| robocap, stride 20 | 694 / 932 | 5 101 | 1214 mm | n/a (no GT) | 0.00 mm |
| robocap, **stride 4** | **4623 / 4648** | **168 926** | **499 mm** | n/a (no GT) | 0.00 mm |

Galileo BA converged 3.18 → 1.54 px mean reprojection error.
The "would-be scale" (the scale a similarity fit *would* have chosen, with the actual fit held
at 1.0) was **0.99345** — cuSFM preserved metric scale to within 0.7 %.

`ATE vs GT` compares cuSFM against the shipped `ground_truth.txt`, matched to within 0.02 ms.
It is a **baseline on one short sample**, not a reproduction of a published number — upstream
ships no per-dataset accuracy figure for this sample.

## RoboCap reconstruction quality — diagnosis

cuSFM's RoboCap reconstruction disagrees substantially with the basalt VIO trajectory it was
initialised from. **Neither is ground truth.** RoboCap ships no GT, and the supplied basalt
trajectory is pure multi-camera VIO — no loop closure, no mapping — so over 124 m of walking it
accumulates drift of its own. The numbers below are therefore a *disagreement between two
estimates*, and an earlier draft of this file wrongly framed them as cuSFM error. What is solid
is the **localisation** of where the divergence enters the pipeline.

A third, independent trajectory settles which estimate moves: cuSFM bundles cuVSLAM, and
`--run.no-skip-cuvslam` makes it write `cuvslam_output/odom_poses.tum` (odometry, no loop
closure — directly comparable to basalt) and `slam_poses.tum` (with loop closure).

**Symptom.** The wearer walks a flat museum floor; basalt puts the whole trajectory in a
0.26 m vertical band, cuSFM spreads it over 6.02 m. A flat floor is the one strong prior here,
and a VIO with an IMU has observable gravity, so bounded vertical drift is expected of basalt —
which makes the 6 m excursion the more suspicious of the two. That is an argument, not a
measurement; the cuVSLAM comparison is the measurement.

| | X | Y | Z (vertical) | path length |
|---|---|---|---|---|
| basalt VIO (input) | 5.22 m | 9.72 m | **0.26 m** | 123.5 m |
| cuSFM (4 cameras) | 8.90 m | 11.64 m | **6.02 m** | 186.6 m |

**It is not scale.** Refitting with scale free only moves ATE 1214 mm → 1000 mm.
**It is not a few outliers.** 86.7 % of samples are >0.3 m off, and samples where all four
cameras registered are as bad (842 mm median) as samples with one (914 mm).

**Where it happens.** Each stage writes its own `frames_meta.json`, so the trajectory can be
traced through the pipeline:

| Stage | position delta vs basalt input | vertical range |
|---|---|---|
| input (basalt) | — | 0.29 m |
| `keyframes/` (after feature extraction) | **0.0 mm** | 0.29 m |
| `pose_graph/` (after pose graph) | **0.0 mm** | 0.29 m |
| `sparse/` (after `keypoints_mapper_main` BA) | **median 692 mm, max 6720 mm** | **6.04 m** |

The initialisation survives feature extraction and the pose graph **byte-exact**. All damage
occurs in the final bundle adjustment. In particular this **exonerates the BoW vocabulary
warning** (`occupied ratio 0.339924, less than 0.7`) — tempting as a false-loop-closure
explanation, but the pose graph provably moves nothing.

**Why BA drifts.** cuSFM treats the supplied poses as an *initialisation*, not a *prior*, so BA
is free to move them. And it is badly under-constrained here: it triangulates 17760 points and
keeps 5101 (71 % rejected), leaving ~7 points per registered image over 124 m.

| Run | pts / registered image |
|---|---|
| galileo | 4938 / 28 = **176** |
| robocap | 5101 / 694 = **7.4** |

A tempting explanation was match-pair *downsampling*: `pycusfm/configs/isaac/match_pair_select_config.pb.txt`
sets `max_keyframes_per_collection: 6` per `collection_time_interval_seconds: 10` with
`use_downsampling: true`, and the runner defaults `downsampling_matches=True` — sensible for a
vehicle driving past never-repeated scenery, plausibly starving a wearer who circles one room.

**Tested and falsified.** `--run.no-downsampling-matches` changed essentially nothing:
17764 triangulated vs 17760, 5124 points kept vs 5101, disagreement 1298 mm vs 1214 mm.
Pair selection was never the binding constraint. The 71 % point rejection is intrinsic to this
sequence — repetitive bookshelf texture through a fisheye at 0.5 m sample spacing — and the
real cause remains open.

**Experiments run.**

| Variant | Registered | Points | disagreement vs basalt | would-be scale |
|---|---|---|---|---|
| 4 cameras (default) | 694 | 5101 | 1214 mm | 0.826 |
| front stereo pair only (2 cameras) | 446 | 4502 | **1788 mm** | 0.740 |
| `skip_pose_graph=True` | **0** | — | — | — |
| `downsampling_matches=False` | 705 | 5124 | 1298 mm | 0.823 |
| `skip_cuvslam=False`, 1 stereo pair | — | — | cuVSLAM lost tracking at frame 231/912 | — |
| `skip_cuvslam=False`, 4-camera rig | cuVSLAM tracked cleanly (230/233 poses, 0 errors) | | see below | |

Two findings from these:

- **The sideways cameras help.** Dropping to the front stereo pair alone makes ATE *worse*
  (1214 → 1788 mm), despite `left`/`right` registering at only 59 % / 48 %. Their weak
  cross-camera overlap is still worth more than the 86 mm front baseline alone.
- **`skip_pose_graph=True` is not a usable knob**: it produces an empty reconstruction, because
  `match_features()` consumes `pose_graph/frames_meta.json`, which only the pose-graph step
  writes. It fails silently at the cuSFM level (exit 0, empty model) — the demo raises instead.

**cuVSLAM as a third opinion.** cuSFM bundles cuVSLAM, so `--run.no-skip-cuvslam` yields an
independent trajectory. Two caveats, both real: cuVSLAM here is fed no IMU (the `frames_meta.json`
contract carries cameras only), so it is pure *visual* odometry against basalt's visual-*inertial*
— an IMU makes gravity observable and bounds vertical drift, so cuVSLAM is expected to be the
weaker estimator, not the referee. And it needs the whole rig: with only the one true stereo pair
declared it builds a 2-camera rig and loses tracking at frame 231 of 912
(`Can't select enough new tracks`, `Failed to track on the PnP stage`). Feeding it all four
world-facing cameras requires declaring the ~90-degree neighbours as `stereo_pair` entries too —
rig *adjacency*, not rectifiable stereo — which is what `--dataset.rig-adjacency-pairs` does.
That flag is deliberately off by default: for a BA-only run, declaring a non-rectifiable pair is
a false claim about the geometry. With the four-camera rig cuVSLAM tracks the sequence instead
of dying at frame 231.

**The three-way comparison cannot cleanly adjudicate, and should not be read as if it could.**
The three estimators do not see the same data:

| | sensing | frame rate | loop closure |
|---|---|---|---|
| basalt (supplied) | visual-**inertial**, 4 cameras | full 30 fps (4648 frames) | no |
| cuVSLAM here | **visual only**, 4 cameras | 1.5 fps (stride 20) | odom: no / slam: yes |
| cuSFM | global BA over 233 samples | 1.5 fps | via pose graph |

cuVSLAM is fed frames 667 ms apart and warns about it
(`Delta between frames at frame 43 is 2001 ms ... Check camera fps and sync settings`); VIO
assumes small inter-frame motion, so this is well outside its envelope. Running it at full rate
would mean 4648 samples x 4 cameras through the whole cuSFM pipeline, which is not a demo.
Treat cuVSLAM here as a sanity check on whether cuSFM's *shape* of disagreement is plausible,
never as a referee between cuSFM and basalt.

**Result: cuVSLAM at 1.5 fps is unusable, and cannot arbitrate.**

| Track | Poses | Path | Vertical range |
|---|---|---|---|
| basalt VIO (visual-inertial, 30 fps) | 233 | 124.0 m | **0.28 m** |
| cuVSLAM odometry (visual-only, 1.5 fps) | 230 | 238.8 m | **88.46 m** |
| cuVSLAM SLAM (visual-only, +loop closure) | 230 | 238.8 m | **88.46 m** |

88 m of vertical excursion inside a single-storey library is nonsense, and the odometry and SLAM
tracks are **bit-identical (0 mm apart)** — loop closure never fired once. cuVSLAM disagrees with
basalt by 40 m RMSE. The 4-camera rig did fix *tracking* (230/233 poses, zero
`Tracking lost` errors, versus dying at frame 231 with one pair), so the failure is the frame
rate and the missing IMU, not the rig.

One inference does survive: cuSFM's disagreement with basalt (**1.2 m**) is ~33x *smaller* than
an independent visual-only VIO's (**40 m**). cuSFM's bundle adjustment stays in the
neighbourhood of its initialisation and drifts; it is not diverging wildly.

**Is this the stock algorithm?** Effectively yes. Every cuSFM *algorithm* setting the demo passes
matches upstream's default: `config_set=isaac`, `feature_type=aliked`, `optimize_extrinsics=False`,
and `ba_frame_type=vehicle_rig` — which merely restates what
`pycusfm/configs/isaac/vision_mapping_config.pb.txt:50` already sets (`ba_frame_type: VEHICLE_RIG`). The
only functional deviations are `min_inter_frame_distance` 0.5 -> 0.0 (which barely binds, since
samples already sit ~0.53 m apart) and `skip_cuvslam=True` (substituting basalt's initialisation
for cuVSLAM's, measured above at 88 m of vertical error — so that choice helps, not hurts).

**The substantial difference is the data, not the settings.** At `frame_stride=20` cuSFM is given
233 of 4648 frames — 5 % — while basalt consumed all of them at 30 fps. Keyframe *spacing* makes
the point better than the frame count:

| Run | Samples | Path | Spacing between keyframes | Result |
|---|---|---|---|---|
| galileo | 29 | 0.66 m | **23 mm** | 3.9 mm vs ground truth |
| robocap, stride 20 | 233 | 123.5 m | **530 mm** | 1.21 m disagreement |

cuSFM performs beautifully where keyframes are 23 mm apart and poorly where they are 23x further
apart, with unconstrained head rotation between consecutive fisheye views on top. That is a data
regime difference, not an algorithm-versus-algorithm comparison, and any statement of the form
"cuSFM is worse than basalt" has to carry that caveat. A denser sweep (`--dataset.frame-stride 4`,
~106 mm spacing) is the direct test.

**Tested, and it is decisive: density was the problem, not the algorithm.**

| Metric | stride 20 (530 mm) | **stride 4 (106 mm)** | basalt |
|---|---|---|---|
| Vertical range (floor is flat) | 6.02 m | **0.73 m** | 0.35 m |
| Path length | 186.6 m | **136.7 m** | 126.0 m |
| Disagreement vs basalt | 1214 mm | **499 mm** | — |
| would-be scale | 0.826 | **0.931** | 1.000 |
| Sparse points | 5 101 | **168 926** | — |
| Points per registered image | 7.4 | **36.5** | — |
| BA point rejection | 71 % | **0.5 %** | — |
| `left` / `right` registered | 59 % / 48 % | **99.9 % / 98.2 %** | — |
| Samples > 2 m off | 17 (7.5 %) | **0 (0.0 %)** | — |

The 6 m vertical excursion — the single clearest sign the stride-20 result was wrong — collapses to
0.73 m, which is plausible for a head-worn rig over 126 m. Residual disagreement is 499 mm over
126 m, i.e. **0.4 %**, unremarkable between a VIO and an SfM with no ground truth to separate them.

This also explains the earlier front-pair-only result. `left`/`right` were not bad cameras; they
were *under-sampled*. At stride 4 they register at 99.9 % / 98.2 % instead of 59 % / 48 %, and
removing them at stride 20 hurt precisely because the remaining pair was starved too.

Runtime cost: ~20 min versus ~6 min at stride 20, on an RTX 5090.

**Where this leaves the question.** Unresolved, honestly. There is no ground truth for this
segment. basalt is the most physically plausible of the three (a flat floor, and an IMU makes
gravity observable so vertical drift is bounded), but "most plausible" is not "correct", and
basalt has no loop closure so it accumulates its own drift over 124 m. What is *established* is
the localisation: the trajectory is untouched through feature extraction and the pose graph, and
diverges only in `keypoints_mapper_main`, where 71 % of triangulated points are rejected and BA
is left with ~7 points per image.

**Not yet tried**: a config profile with keyframe spacing suited to a walking wearer
(the docs mention a `backpack` profile at 6 cm / 1.5°, but its config directory is not shipped),
`--run.optimize-extrinsics` (rejected here because it would break the exoego static-extrinsic
contract), and feeding BA a pose prior, which cuSFM does not appear to expose.

**Galileo is unaffected** — 224/226 images, ATE 1.7 mm against its own input and 3.9 mm against
shipped ground truth. Whatever is happening is specific to this long, repetitive, fisheye
sequence, not to the fork's plumbing.

## Trajectory polylines

Both trajectories are also logged as polylines in the **same entity layout the RoboCap catalog's
slam layer uses**, so the recording reads the same way as its source:

    /world/runs/<name>              AnyValues{num_poses, source}
      /trajectory                   static LineStrips3D, n-1 two-point segments,
                                    per-segment colour ramp (direction of travel), radii 0.004
      /endpoints                    Points3D labelled start/end, green/red, radii 0.02

`runs/input` is blue, `runs/cusfm` orange. This turns out to be the clearest single view of the
reconstruction-quality issue: basalt draws a smooth planar loop around the room, while cuSFM
follows the same loop but visibly jagged and with one large excursion.

## Upstream edits

**None.** `git diff main -- '*.py'` shows only the added `demo_rerun.py`.
Change set: `pixi.toml`, `pixi.lock`, `.gitignore`, `demo_rerun.py`, `NOTES.md`, one README section.


### Why pixi deps cannot make the binaries portable

Tempting: pin `glog=0.6` (SONAME `libglog.so.1`) and `libopencv=4.6` (`.so.406`) so
`LD_LIBRARY_PATH` serves them from the env. Measured outcome:

- `libopencv 4.6` requires `ffmpeg >=4.4,<6`, which conflicts with this workspace's
  `ffmpeg 8` (and PyAV's `add_stream_from_template` needs a modern av, which needs
  modern ffmpeg). The solve fails outright.
- Even with libs served, the executables themselves demand `GLIBC_2.38` and
  `GLIBCXX_3.4.32` symbol versions (checked with `objdump -T`), so Ubuntu 22.04
  (glibc 2.35) fails in the dynamic loader regardless — observed verbatim on a
  22.04 fleet machine. Fixing that needs `patchelf --set-interpreter` onto copies
  of the binaries plus a `sysroot_linux-64=2.39` runtime tree: possible, invasive,
  out of scope.

Hence the README's apt line for the host libs and the hard Ubuntu 24.04 requirement.


### Video frame references (rerun 0.37.1) — what they buy here, and what they don't

0.37.1 lets `VideoFrameReference` point at a `VideoStream` on another entity. Measured on
this recording (306 MB): **302 MB is the four H.264 `VideoStream`s under `rig_00`, logged
once**; everything else is 4 MB. So references cannot shrink *this* file — there was never a
duplicate to remove, and `AssetVideo(path=...)` **embeds** the bytes (77.6 MB mp4 -> 78.0 MB
rrd; Rerun has no external-file video reference). They do give the cuSFM rig imagery for
free: `log_video_references` puts `VideoFrameReference` columns under
`rig_01/cam_NN/pinhole/video` pointing at `rig_00`'s streams — 72,436 references cost 0.3 MB
(306.74 -> 307.05 MB), pixel-verified identical frames on both rigs at the same cursor.

Two gotchas: (1) a single *static* reference (`nanoseconds=0`) rendered a **different**
frame than the stream at the same cursor on 0.37.1 — use one reference per source sample on
the stream's timeline, which tracks exactly; (2) the viewer opens on `log_time`, where the
video has no data, so validation screenshots need the time cursor moved onto `video_time`
(viewer MCP `set_time`) or they are black. Nothing here is rectified — raw fisheye under a
`Pinhole` with distortion coefficients — so the same frames are valid on both rigs.

## Gotchas found

1. **`tensorrt-cu13==10.13.3.9` is broken on PyPI** — it depends on the retired
   `nvidia-cuda-runtime-cu13==0.0.1`. Use **`10.13.3.9.post1`**.
2. **`pycusfm`'s own `pyproject.toml` hard-depends on `tensorrt-cu12`**, which conflicts with
   `cuda-toolkit>=13`. Resolved with a pixi `[pypi-options.dependency-overrides]`.
3. **Both TensorRT wheels install into the same `tensorrt_libs/` directory**, and the cu12 wheel
   overwrites the cu13 `.so` files despite a direct pin. The `select-tensorrt-cu13` task
   reinstalls the cu13 libs/bindings last. Ugly but load-bearing — do not delete it.
4. **`simplecv` on PyPI is an unrelated 2013 machine-vision package (v1.3).** A bare
   `simplecv = "*"` installs *that*, silently. Worse, the standalone `pablovela5620/simplecv`
   repo predates `rerun_rig_logger.py` / `rig.py`, so it has no `log_rig_static` and no
   exoego:v2 at all. The only usable copy is the monorepo's, pulled as a git subdirectory
   dependency (the monorepo is public, so this needs no auth).
5. **`typing-extensions` must be pinned `>=4.1,<4.16`.** simplecv → `pyserde<0.32` requires it,
   but the conda solve otherwise pins 4.16.0 and the pypi solve then has *no solution*.
   (Documented in the add-model skill; hit it anyway.)
6. **nvdec is available but slower than CPU here.** Once video is remuxed to MP4,
   `ffmpeg -c:v h264_cuvid` works — but extracting 233 JPEGs took **4.16 s** against **2.32 s**
   on CPU. The cost is JPEG *encoding* plus the device-to-host copy, not H.264 decode.
   PyAV's library-level `h264_cuvid` path fails outright (`avcodec_send_packet()` EOF) because
   PyAV never wires up the CUDA hw device context. CPU decode of 4 cameras is ~10 s total,
   which is noise next to matching 932 fisheye images.
7. **Reading video out of a recording: concatenate, do not hand-feed packets.** Rerun's
   [documented remux recipe](https://rerun.io/docs/howto/query-and-transform/query_videos#exporting-to-mp4-remuxing)
   — join the Annex-B samples into one buffer, `av.open(BytesIO(...), format="h264")`, rewrite
   `pts`/`dts` from the recording's nanosecond times (no B-frames, so `dts == pts`) — works
   first try. Feeding individual packets to a `CodecContext` is fragile because SPS/PPS live
   in-band at keyframes.
8. **`frames_meta.json` is proto3-serialised, so zero-valued fields are omitted.**
   29 of galileo's 226 keyframes have **no `camera_params_id`** — every one of them is camera
   `"0"` (`back_stereo_camera_left`). Likewise a `stereo_pair` entry is missing
   `left_camera_param_id` because its left camera is `"0"`. Any parser must default these to
   `"0"` rather than `KeyError`.
9. **Chunk record batches use bare component names.** Reading a recording via
   `RrdReader(...).store().stream()` gives columns named `name`, `kind`, `Transform3D:mat3x3`;
   the *catalog* dataframe API gives the same components prefixed with the entity path
   (`/world/rig_00/cam_00:name`). A helper written against one shape silently finds nothing
   against the other.
10. **`camera_to_world` in `frames_meta.json` is `world_T_cam`, not the vehicle pose.**
    Verified numerically: `camera_to_world @ inv(sensor_to_vehicle_transform)` reproduces
    galileo's `ground_truth.txt` vehicle position to **~1.6 mm**. `sensor_to_vehicle_transform`
    is `rig_T_cam`.
11. **`--output_rgb` is opt-in.** Without it every point in `points3D.txt` is `0 0 0` and the
    cloud renders solid black — which looks like a logging bug but is not.
12. **The catalog stores segments in memory** (`memory:///store/...`). A catalog restart loses
    all 40 RoboCap segments, so the demo freezes its segment to disk once and never reads the
    catalog again on subsequent runs.
13. **Pin `ffmpeg = "8.*"`.** The lock currently supplies 8.1.2, the major used to validate
    explicit `select` plus `-fps_mode passthrough` JPEG extraction. Keep that tested major until
    the same path has been exercised on ffmpeg 9; no removed `-vsync` option is used now.
14. **Trimming must move image selection *and* pose selection together.** Adding the head/tail
    trim initially changed only `sample_indices`, leaving frame extraction at
    `range(0, n, stride)`. Poses started ~60 frames in while images still started at 0, pairing
    every image with a pose ~15 samples away — silent, and it would have fed cuSFM systematically
    mismatched image/pose pairs. Both now derive from one trimmed index list.
15. **tyro argument order**: top-level options must precede the subcommand
    (`demo_rerun.py --rr-config.headless dataset:robocap`, not the reverse).

## Fisheye frusta — a known visual limitation

Rerun draws a camera frustum from **`K` alone**. A Kannala-Brandt fisheye with a 150°+ field of
view therefore renders as a much narrower cone than it really sees. The distortion coefficients
*are* logged (`PinholeWithDistortion` → `simplecv.components.DistortionModel` /
`DistortionCoefficients`) and all geometry is correct; only the drawn cone under-represents the
FOV. This is identical to what the RoboCap catalog's own `base` layer shows, so "correct" here
means "matches the catalog". Undistorting to a pinhole would make the cone honest but would
destroy the lossless KB4 mapping and change what cuSFM sees.

RoboCap's `Fisheye62` distortion has 8 slots but populates only `k1..k4`, so the mapping to
cuSFM's `OPENCV_FISHEYE` (and onward to COLMAP model_id 9) is **lossless**.

## Non-hermetic: Ubuntu 24.04 host only

The prebuilt binaries have an **absolute** RUNPATH
(`/usr/local/lib:/opt/nvidia/cvcuda0/lib:/usr/local/cuda/lib64`), not `$ORIGIN`-relative, and
link Ubuntu 24.04 system C++ libraries (opencv 4.6 `.so.406`, glog, gflags, protobuf `.so.32`,
abseil) that conda-forge cannot reproduce ABI-for-ABI. `LD_LIBRARY_PATH` therefore falls back
to `/usr/lib/x86_64-linux-gnu`. This is inherited from upstream's binary distribution and
cannot be fixed from the pixi side.

## Running the full pipeline (Figure 2 of the paper)

The paper's Figure 2 has eight modules. Two were silently off, because cuSFM's defaults
disable them and nothing warns:

| Figure 2 module | Binary / flag | Default |
|---|---|---|
| Feature Extraction (ALIKED) | `feature_extractor_main` | on |
| Camera Projection | fisheye projector | on |
| Dictionary Construction | `generate_bow_vocabulary_main` + `_index_main` | on |
| **Loop Closure Detection** | `generate_association_main` | **OFF** (`skip_data_association=True`) |
| Feature Matching (LightGlue) | `feature_matcher_main` | on |
| Pose Graph Optimization | `pose_graph_main` | on, but with **0 loop pairs** without the above |
| Triangulation & Mapping | `keypoints_mapper_main` | on |
| **Extrinsic Refinement** | `--optimize_extrinsics` | **OFF** |

`skip_data_association` defaults to `True` in `get_default_cusfm_params`, so
`generate_association_main` never runs. Without it the pose graph falls back to
`image_retrieval_config.pb.txt`, whose **`query_result_number: 1`** returns a single BoW
candidate per query — nearly always the temporally adjacent frame, never a loop. Every run
before this reported `Perform pose graph optimization with 0 loop constraints`, with
~316 s (98 % of the stage, 30 % of total runtime) spent searching and finding nothing.
`--optimize_extrinsics` additionally requires `ba_frame_type=vehicle_rig`.

**Bundle adjustment is not unconstrained** — an earlier draft of this file claimed it was.
`vision_mapping_config.pb.txt` sets `use_relative_pose_constraint: true` ("always use the
initial guess pose as relative pose constraint") with `relative_pose_translation_error_meters: 0.1`
and `absolute_pose_translation_error_meters: 1`. That 1 m absolute leash is the scale of the
disagreements measured here (1214 mm at stride 20, 499 mm at stride 4) — cuSFM is behaving as
configured, and that sigma is the knob if tighter adherence to the prior is wanted.

### Trimming the capture

The first ~2 s are a blown-out white frame while exposure settles (textureless, pollutes the
vocabulary), and the last ~2 s contain the wearer's hand stopping the capture — a large moving
occluder exactly where the loop should close. `--dataset.trim-start-seconds` /
`--dataset.trim-end-seconds` default to 2.0, verified at 2.000 s / 2.001 s.

### Two things measured, not assumed

- **`feature_extractor_batch_size` must stay 0 for ALIKED.** `--batch_size > 0` routes to a
  batched detector path that aborts: `Unsupported detector type: ALIKED_DETECTOR`.
- **`num_threads` changes nothing.** 6.0 s at `--num_thread 1` vs `--num_thread 32` on the same
  96-image sample; throughput matches the large run (16 vs 17 img/s). ALIKED runs on the GPU
  through TensorRT, so the flag only affects CPU-side loading. Both were briefly written up here
  as missed optimisations; they are not.

## Profiling

Two independent instruments, because the work splits cleanly in two: cuSFM ships its own
per-stage wall-clock in `<base>/runtime.csv`, and the Python side was profiled with
`py-spy record --rate 50`.

### cuSFM pipeline (RoboCap `s00000021`, RTX 5090), slowest to fastest

| Stage | stride 4 | share | stride 20 | scaling for 5x data |
|---|---:|---:|---:|---:|
| `pose_graph_main` | **321.6 s** | 30.9 % | 28.1 s | **11.4x — superlinear** |
| `feature_extractor_main` | 269.6 s | 25.9 % | 53.5 s | 5.0x linear |
| `generate_bow_vocabulary_main` | 163.9 s | 15.8 % | 35.2 s | 4.7x linear |
| `keypoints_mapper_main` (BA) | 130.2 s | 12.5 % | **235.4 s** | **0.55x — faster with more data** |
| `generate_bow_index_main` | 77.6 s | 7.5 % | 13.0 s | 6.0x |
| `feature_matcher_main` | 66.2 s | 6.4 % | 13.2 s | 5.0x linear |
| `kpmap_to_colmap` | 5.9 s | 0.6 % | 5.0 s | 1.2x |
| `feature_matcher_task_builder_main` | 5.0 s | 0.5 % | 3.5 s | 1.4x |
| `extract_pose_from_map_main` | 0.1 s | 0.0 % | 0.0 s | — |
| **total** | **1040 s (17.3 min)** | | **387 s (6.4 min)** | 2.7x |

Two results worth keeping:

- **`pose_graph_main` is the scaling bottleneck**, the only superlinear stage. Anything denser
  than stride 4 will be limited by it, not by matching or BA.
- **Bundle adjustment is *faster* with 5x more data** (235 s -> 130 s). At stride 20 it thrashes,
  rejecting 71 % of points, and takes 1.8x longer to produce a far worse answer. Starving it cost
  both quality and time.

### Python side — a 164x fix found by profiling

`py-spy` put **94.6 % of 73.6 s on a single line** of `read_robocap_rrd`:

```python
for sample, timestamp in zip(samples.to_pylist(), times.to_pylist()):
```

`to_pylist()` on the `VideoStream:sample` column boxes every byte of every H.264 blob into Python
objects — and the recording stores video in 2395 small chunks, so the cost is paid over and over.
Reading the Arrow buffer directly (`flatten()` -> `offsets`/`values` -> `tobytes()`) is byte-for-byte
identical (SHA-256 over all 27 906 samples matches) and:

| | before | after |
|---|---:|---:|
| video read | 65.7 s | **0.4 s** |
| `read_robocap_rrd` | 65.4 s | **0.5 s** |
| full re-log run | ~74 s | **2.96 s** |

## Timing (this host: Ubuntu 24.04, RTX 5090, driver 580.173)

| Step | Time |
|---|---|
| `pixi install` (warm package cache, empty env) | 27 s |
| pixi solve (warm repodata) | 0.35 s |
| Freeze `s00000021` from catalog → 477.5 MB `.rrd` | 1.1 s |
| RoboCap prepare (4 cams: read rrd, remux, extract 932 JPEGs) | 76 s |
| First TensorRT engine build (`sm_120`, fp16) | ~4 min (cached afterwards) |
| galileo `demo-upstream` (upstream defaults, incl. engine build) | 267 s |
