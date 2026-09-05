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

Three gotchas, isolated with a five-pane A/B (stream; static ts=0; static ts=MID;
temporal-at-MID ts=0; per-sample columns) at two cursor positions:
(1) a stream reference shows the frame at the **reference's own timestamp**, not the viewer
cursor — `static` is irrelevant; ts=MID stays pinned on MID at every cursor, and ts=0 clamps
to the first sample. Only per-sample references with the stream's timestamps follow the
cursor, which is what `log_video_references` logs. (2) Even those showed a late frame after a
*backward* seek (correct after forward seeks) — plausibly a decoder-seek issue in 0.37.1 for
reference-driven decoding; worth reporting upstream. (3) The viewer opens on `log_time`,
where the video has no data, so validation screenshots need the cursor moved onto
`video_time` (viewer MCP `set_time`) or they are black. Nothing here is rectified — raw fisheye under a
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

## colsfm: the open pipeline on pycolmap

`colsfm/` is a Python and pycolmap pipeline that replaces the 21 NVIDIA cuSFM binaries in
`pycusfm/x86_cuda13/bin`. It reads the same `frames_meta.json`, writes the same outputs
(`sparse/` COLMAP model, `kpmap/keyframes/frames_meta.json` with optimised poses, TUM pose
files, `runtime.csv`), and runs the same eight stages.

**Why.** The blob hides every algorithm setting. A protobuf text config exposes a few
numbers, but the schedule, the gates, the loss and the solver options live inside an ELF
file. None of that could be read, measured, or changed. Two things were needed: a
pipeline that can be benchmarked against the binary it replaces, and a pipeline whose
settings can be edited. `docs/spec/*.md` is the contract between the two. Those eight specs
are reverse-engineered from the unstripped binaries, the embedded protobuf descriptors, the
shipped configs, and A/B re-runs of the real binaries. `colsfm/` implements the specs;
`colsfm/benchmark.py` measures the difference.

### Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **Opus 5 for every worker** (reverse engineering, specs, implementation, review) | User decision, recorded in `docs/open-pipeline-plan.md` "Engineering rules". The work is one long chain of inference over disassembly; a cheaper tier drops the chain. |
| 2 | **No clean room** | The repo is Apache 2.0 and the binaries carry no separate EULA. Reading the disassembly directly is allowed, so one worker both reads the blob and writes the port. A clean-room split would double the cost and buy nothing. |
| 3 | **pycolmap-native `ALIKED_N16ROT` + LightGlue, not the TensorRT engines** | COLMAP 4.2 ships `FeatureExtractorType.ALIKED_N16ROT` and `FeatureMatcherType.ALIKED_LIGHTGLUE` as ONNX and downloads the graphs itself into `~/.cache/colmap/`. The blob's `aliked.onnx` carries a 32-channel SDDH offset convolution, so `M = 16`, the ALIKED-n16 architecture; `ALIKED_N16ROT` is the closest COLMAP variant. A thin typed driver replaces a TensorRT runner and its engine cache. The weights are trained with rotation augmentation, so descriptors are not the blob's bit for bit. The plan never asked for that. |
| 4 | **Pose graph on pyceres with a Python cost** | pycolmap has no pose-graph optimiser: `pycolmap.PoseGraph` is a container with no `optimize`, and `PoseGraphEdge` carries no information matrix. `pycolmap.cost_functions` advertises `RelativePosePriorCost` and friends, but every factory raises `TypeError: Unregistered type` against the standalone `pyceres`, because conda-forge's pycolmap embeds a cut-down pyceres in its own pybind11 registry. The cost functions cannot cross into pyceres, so the residual is written in Python. Measured on 1000 nodes: pyceres converges in 24.6 s (cost 9.563 to 4.704e-06), scipy `least_squares` does not converge and was killed after 40 minutes. The sparse Cholesky solve is 40 ms of that 24.6 s, so all the headroom sits in the residual. |
| 5 | **One pose per rig frame, vehicle body as reference sensor** | cuSFM's `VEHICLE_RIG` mode carries one pose per `synced_sample_id` in the vehicle FLU frame. COLMAP forces a rig's reference sensor to identity (`rig.h:200 Check failed: sensor_id != ref_sensor_id_`), so no camera can be the reference without moving the rig origin onto it. The vehicle body is registered as `sensor_t(SensorType.IMU, 0)` and owns no images. Triangulation, `ObservationManager` and `create_default_bundle_adjuster` all accept that. The two models then compare pose to pose with no change of basis. |
| 6 | **`loss_function_scale = 4.0`** | cuSFM divides the reprojection residual by `reprojection_error_standard_deviation = 4.0 px` and then applies `CauchyLoss(1.0)`. COLMAP does not whiten. `CauchyLoss(a)` is `a^2 log(1 + s/a^2)`, so evaluating it at `s/sigma^2` with `a = 1` equals evaluating it at `s` with `a = sigma`, up to a constant that cannot move the minimum. Setting the scale to sigma reproduces the blob's robustifier without touching the residual. |
| 7 | **A 500-match spatial cap in place of the blob's SSC NMS** | The blob thins each pair to a spatially uniform top 500 using the LightGlue score as the keypoint response. pycolmap exposes no per-match score: `Database.read_matches` and `FeatureMatcher.match` both return bare `uint32[m, 2]` index pairs. `subsample_matches_by_coverage` lays a grid of about `match_top_k` cells over image 0 and keeps one match per occupied cell, after verification, so every survivor is an inlier of the same RANSAC. Measured over the 331 Galileo pairs: uncapped 1300 matches per pair, 1.80 px, 5.39 mm ATE; capped 156 matches per pair, 1.33 px, 4.32 mm; the blob 430 matches, 1.55 px, 5.00 mm. The cap also takes bundle adjustment from 27.7 s to 2.8 s. |
| 8 | **Relative acceptance bounds, not absolute ones** | The first bounds were absolute: registered >= 220, ATE <= 5 mm, reprojection <= 1.7 px. The ATE and reprojection figures came from the older NOTES table above (3.9 mm, 1.54 px), measured by the demo. When `colsfm.benchmark` measures the blob itself it gets **5.00 mm** and **1.550 px**. So the absolute 5 mm bound told the port to beat the binary it reproduces, and would have failed a bit-perfect clone. Absolute figures also break on a new machine or a re-run of A. Every bound is now a ratio against run A as this harness measures it. |
| 9 | **Vocabulary-tree retrieval above 500 images** | `RetrievalConfig.backend` is `auto`: brute-force mutual-nearest-neighbour voting at or below 500 images, a hierarchical k-means vocabulary with TF-IDF and the DBoW2 L1 score above it. Brute force is quadratic in images times descriptors (gotcha 12). The vocab backend builds in 42 s on RoboCap and answers all 4528 queries in 1.5 s, and it recovers every one of the blob's 90 loop pairs. |
| 10 | **Loop closure off by default** | On Galileo, turning loops on moves camera positions by 5 to 13 mm against a 5 mm ATE budget: a short, low-drift sweep has nothing for a loop to fix. On RoboCap the retrieval is right and `colsfm.loop_pose`'s metric rig-to-rig measurement takes the pose graph from 609.0 mm to **408.1 mm** against the blob's PGO, past the 460.8 mm of the input trajectory alone — but not past the 135.0 mm the blob's own edges reach, and the remaining gap is a 6.5 deg rotation disagreement two independent image-based estimators put on the blob's side (see the deviations below). `LoopClosureConfig.enabled` stays False and the caller decides per dataset. |
| 11 | **Its own pixi environment and solve group** | `colsfm` is `no-default-feature`, so the fragile CUDA 13 plus TensorRT solve of the default environment is untouched. Verified: the `default`, `raco` and `bench` blocks of `pixi.lock` are byte-identical to `HEAD`, and no package was removed. |

### Results: blob against colsfm

Both runs measured by `colsfm.benchmark` on the same host (Ubuntu 24.04, RTX 5090, driver
580.173). A is the blob with `data/cusfm_configs/loop-closure-fixed` and all eight stages;
B is `colsfm` with loop closure off. Reprojection error is recomputed from the tracks,
because `kpmap_to_colmap` hardcodes `ERROR = 2.0` on every point.

**Galileo** (226 keyframes, 29 rig frames, 8 pinhole cameras, 0.66 m sweep), measured in
`data/bench/galileo_compare.md`:

| Metric | A blob | B colsfm |
|---|---|---|
| registered / total images | 224 / 226 | 225 / 226 |
| 3D points | 5065 | 6195 |
| observations | 37 820 | 30 910 |
| mean reprojection error (px) | 1.550 | 1.334 |
| track length mean / median | 7.47 / 5.00 | 4.99 / 4.00 |
| rig rigidity spread (mm) | 0.000 | 0.000 |
| ATE vs input trajectory (mm RMSE) | 1.82 | 1.26 |
| ATE vs ground truth (mm RMSE) | 5.00 | 4.33 |

| Stage | A (s) | B (s) | B/A |
|---|---|---|---|
| 1 feature extraction + keyframe selection | 8.64 | 8.01 | 0.93 |
| 2 BoW vocabulary + index | 11.73 | none | none |
| 3 loop-closure association | 3.17 | 0.00 | 0.00 |
| 4 pose graph optimisation | 3.58 | 0.01 | 0.00 |
| 5 match pair selection | 1.94 | 0.00 | 0.00 |
| 6 feature matching | 2.25 | 8.15 | 3.63 |
| 7 triangulation + bundle adjustment | 6.69 | 2.70 | 0.40 |
| 8 COLMAP + TUM export | 2.07 | 0.47 | 0.23 |
| **total** | **40.07** | **19.35** | **0.48** |

All four acceptance bounds pass: registered 225 against `>= 220`, reprojection 1.334 px
against `<= 1.704`, ATE 4.327 mm against `<= 5.495`, runtime ratio 0.483 against `<= 2.0`.

**RoboCap** stride 4 (4528 keyframes, 1132 rig frames, 4 fisheye cameras, 124 m walk),
measured in `data/bench/robocap_compare.md`. No ground truth ships with this segment, so
there is no acceptance table and the trajectory row is a disagreement, not an error:

| Metric | A blob | B colsfm |
|---|---|---|
| registered / total images | 4526 / 4528 | 4528 / 4528 |
| 3D points | 168 874 | 157 887 |
| observations | 947 651 | 714 413 |
| mean reprojection error (px) | 1.486 | 1.501 |
| track length mean / median | 5.61 / 4.00 | 4.52 / 3.00 |
| rig rigidity spread (mm) | 0.000 | 0.000 |
| vs input basalt trajectory (mm RMSE) | 334.10 | 461.25 |

| Stage | A (s) | B (s) | B/A |
|---|---|---|---|
| 1 feature extraction + keyframe selection | 183.91 | 123.32 | 0.67 |
| 2 BoW vocabulary + index | 243.39 | none | none |
| 3 loop-closure association | 75.66 | 0.00 | 0.00 |
| 4 pose graph optimisation | 386.61 | 0.34 | 0.00 |
| 5 match pair selection | 5.62 | 0.04 | 0.01 |
| 6 feature matching | 52.53 | 152.03 | 2.89 |
| 7 triangulation + bundle adjustment | 178.75 | 138.02 | 0.77 |
| 8 COLMAP + TUM export | 7.07 | 8.05 | 1.14 |
| **total** | **1133.53** | **421.79** | **0.37** |

The blob spends 705.7 s of its 1133.5 s on stages 2, 3 and 4, which exist only to find loop
closures. colsfm skips all three by default, which is most of the 0.37x. Matching is the one
stage that is slower, at 2.89x here and 3.63x on Galileo.

**RoboCap with loop closure** (`--loop-closure`, measured in
`data/bench/robocap_loops_compare.md`; the blob reference always runs with loop closure):

| | blob | colsfm, no loops | colsfm, `--loop-closure` |
|---|---|---|---|
| loop edges in the pose graph | 90 | 0 | 74 |
| vs input trajectory (mm RMSE / max) | 334.1 / 627.3 | 461.3 / 1134.9 | **312.7 / 608.4** |
| would-be scale | 0.9401 | 0.9306 | 0.9402 |
| mean reprojection (px) / points | 1.486 / 168 874 | 1.501 / 157 887 | 1.871 / 215 434 |
| rig poses vs blob (mm RMSE / max) | — | 238.3 / 796.4 | **146.4 / 289.6** |
| total runtime | 1133.5 s | 421.8 s (0.37x) | 1723.0 s (1.52x) |

With loops the colsfm trajectory sits closer to the input than the blob's does and its rig
poses land within 146 mm of the blob's. Neither is ground truth (RoboCap ships none), but
this is the direction the blob's loop closure moves the trajectory, and the 2x runtime bound
still holds. The cost is the loop stage (721 s, of which most was re-matching 8 400 pairs
stage 4 had already matched; the pipeline now skips those, not yet re-measured end to end)
and a larger bundle adjustment (684 s against 138 s without loops). Loop closure stays off
by default because the Galileo-class case gains nothing from it; for long sequences with
revisits, turn it on.

### Where colsfm deviates from the blob

1. **The point cloud is a superset, from LO-RANSAC.** cuSFM grows one disjoint track per
   connected component of the match graph and triangulates it with plain MSAC: no local
   optimisation, no final refit. It rejects roughly half its candidate tracks. COLMAP's
   LO-RANSAC keeps them. On Galileo that is 14 496 points against 5065 on the blob's own
   matches, at a lower reprojection error. 98 to 99 % of the blob's points have one of ours
   within 5 cm, and merging is already at a fixed point, so the difference is structural.
2. **No per-match scores, so no SSC.** See decision 7. The grid subsample runs after
   verification and only reduces the count.
3. **No pose priors in bundle adjustment.** pycolmap's `PosePrior` is 3-DoF position plus an
   optional gravity direction. There is no 6-DoF absolute-pose prior and no relative-pose
   prior in the BA path, so cuSFM's `use_relative_pose_constraint` with its
   `relative_pose_translation_error_meters` (0.1 m) has no equivalent. colsfm registers every
   rig frame before the first triangulation and lets only bundle adjustment move it; nothing
   pulls it back.
4. **No unweighted constant-frame quirk.** cuSFM whitens every reprojection residual by
   `1/sigma` except the ones observing the gauge keyframe, whose functor has no weight member:
   13 of 6276 blocks on Galileo. COLMAP applies one weighting uniformly, and there is no way to
   ask for the inconsistency. This is a blob defect, so it is left out on purpose.
5. **No 3-D depth residual.** `VehicleCameraReprojectionCost3D` handles keypoints that carry
   depth. pycolmap's bundle adjuster is 2-D only, and Galileo sets
   `keypoint_feature_has_depth: false`.
6. **The loop-edge estimator is a local metric map plus generalized resection**
   (`colsfm.loop_pose`), not the blob's four-view stereo estimator with baseline-locked
   scale and `StereoPoseRefineSolver`. It triangulates points in the source rig frame from
   that rig's four cameras and its temporal neighbours, then resections the whole target rig
   with `pycolmap.estimate_and_refine_generalized_absolute_pose`, so the edge takes nothing
   from the prior pose of the pair. Measured on RoboCap against the blob's 90 LOOP edges:
   retrieval and pair selection are right (all 90 blob pairs are covered by our 488 edges to
   within 2 rig frames; the blob's own edges through `solve_pose_graph` land 135.0 mm from
   its result and oracle poses on our own pairs land at 39.9 mm), the old two-view plus
   prior-scale measurement landed at 609.0 mm, and this one lands at **408.1 mm** on that pair set and
   404.4 mm end to end — better than the 460.8 mm of the input trajectory, not as good as
   135.0 mm. What is left is
   rotational: on the blob's own 90 pairs our translations move off the odometry towards the
   blob's (median 132 mm against the odometry's 355 mm and the blob's 160 mm), while our
   rotations sit 6.5 deg from the blob's and 1.5 deg from the odometry's — and
   `pycolmap.estimate_generalized_relative_pose`, which shares only the correspondences,
   agrees with ours to 0.35 deg. Substituting the blob's rotations into our edges takes the
   pose graph from 392 mm to 266 mm; substituting its translations changes nothing. RoboCap
   ships no ground truth, so this is where it rests; the full tables are in
   `colsfm/loop_closure.py`.
7. **The epipolar gate is in pixels.** The blob's
   `max_mean_point_to_epipolarline_error: 10e-6` rejects 168 of 180 RoboCap candidates and
   makes the blob produce zero loop associations. colsfm uses `ransac_max_error_px = 4.0`,
   COLMAP's own default.
8. **No BoW artifacts.** Nothing downstream reads the word ids, the tree, the IDF values or the
   file formats, so `colsfm.retrieval` keeps only the ranked candidate list.
9. **The LightGlue score threshold is 0.1, not 0.3.** The blob's 0.3 is calibrated for its own
   engine. Against COLMAP's graph, 0.3 makes the low-texture `left_stereo_*` pairs collapse:
   22 of 331 Galileo pairs come back empty (6.6 %) against the blob's 8 (2.4 %).

### Gotchas found

1. **`cuda-version = "13.0.*"` does not select the CUDA pycolmap build.** pycolmap 4.2.0 has
   `cpu_*`, `cuda_129_*` and `cuda_130_*` builds on conda-forge, and the `cpu_*` build declares
   no `cuda-version` constraint at all, so it satisfies a CUDA 13 environment and the solver
   takes it. The first solve here silently installed `pycolmap-4.2.0-cpu_h874a1db_0` and
   `pycolmap.has_cuda` was False. The build string is the only thing that selects it:
   `pycolmap = { version = "4.2.0.*", build = "cuda_130*" }`.
2. **pixi will not feed `__cuda` to the solver unless the feature asks.** With the build string
   pinned the solve failed with `pycolmap 4.2.0 would require __cuda >=13.0, for which no
   candidates were found`, even though `pixi info` reports `__cuda=13.0=0`.
   `[feature.colsfm.system-requirements] cuda = "13.0"` fixes it. pixi 0.77.1 prints a
   deprecation warning for that table and points at a `platforms = [{ ... }]` form which its own
   parser rejects and which is workspace-scoped anyway. The warning is accepted on purpose.
3. **A bare `pyserde` breaks the PyPI solve.** simplecv 0.7.2 requires
   `pyserde>=0.31.2,<0.32`; the conda solver pins 0.32.1 first and the PyPI solve then has no
   solution. Same shape as the `typing-extensions` cap in the fork's own gotcha list above.
   Both caps are declared conda-side so one solver owns the constraint.
4. **ONNX Runtime's CUDA provider needs cuDNN, and its absence is not an exception.** The
   provider dlopens `libcudnn.so` for the convolutions. Without it, ONNX Runtime throws inside
   a COLMAP worker thread and the process aborts with SIGABRT. `colsfm.features.resolve_device`
   probes for the library first and falls back to the CPU provider, which is about 50x slower
   (1.8 s per 1920x1200 image against 0.03 s). The real fix is `cudnn = "9.*"` in the feature.
5. **`Camera.has_prior_focal_length` defaults to False and silently downgrades two-view
   geometry.** PINHOLE falls back to a fundamental matrix (`UNCALIBRATED`, `cam2_from_cam1 is
   None`); OPENCV_FISHEYE returns `DEGENERATE` with zero inliers, because F cannot model
   fisheye. Nothing warns. Every camera built from `frames_meta.json` sets it True explicitly.
6. **pycolmap overrides Ceres' solver tolerances.** pycolmap ships `function_tolerance = 0.0`,
   `gradient_tolerance = 1e-4`, `parameter_tolerance = 0.0`. cuSFM leaves Ceres' own defaults
   in place, so `colsfm.mapping` sets 1e-6, 1e-10 and 1e-8 back. A port that trusts the
   pycolmap defaults solves a different problem.
7. **The beartype claw makes per-point pycolmap calls quadratic.** With
   `beartype_this_package()` active, a loop of `Image.project_point(xyz)` calls carrying
   jaxtyping-annotated locals leaks about 15 uncollectable objects per call. Repeated
   `build_scene()` calls went 0.54 s, 1.60 s, 2.64 s, 4.05 s, and the suite took 367 s.
   Batching through `Camera.img_from_cam(points_in_cam)` removed it: `build_scene()` is 0.004 s
   and the suite is 37 s. Rule for `colsfm/`: never call a pycolmap per-item accessor inside a
   Python loop when a batched overload exists.
8. **Copying the blob's `--num_thread 1` is wrong.** That gflag counts the blob's *own worker
   processes*, and the runner launches one per camera. COLMAP is a single process, so 1
   serialises JPEG decode against the GPU and costs 4.4x on Galileo: 33.9 s against 7.6 s.
   `--num-threads` defaults to -1 here, and Ceres takes the config's 8.
9. **`write_keypoints` accepts any column width.** It casts to float32 (so
   `123.456789012345` comes back as `123.45679`) and stores whatever column count it is handed.
   COLMAP writes Nx2, Nx4 or Nx6, but Nx1, Nx3, Nx5 and Nx8 all round-trip unchanged. A
   malformed width is a silent corruption downstream, not an error at write time.
10. **`demo_rerun.read_colmap_model` mis-parses a model with a zero-observation image.**
    COLMAP writes two lines per image, and an image with no 2D points has an *empty* second
    line. The parser filters empty lines before taking every second line, so the pose/points
    alternation shifts from that point on and `POINTS2D` lines are read as poses. It does not
    raise: on a three-image model it returned a nonsense entry with a 201 m translation and
    dropped a real pose. `colsfm.benchmark` reads models with
    `pycolmap.Reconstruction.read_text` instead. `demo_rerun.py` is not changed here.
11. **The 10 s loop gate rejects every candidate on Galileo.** `generate_association_main`
    drops candidates with `|dt| < loop_interval_threshold_in_seconds` (10 s), while
    `pose_graph_main` drops `|dt| < loop_closure_interval_ratio * session_duration` (0.08 of
    the session). The shipped Galileo sequence lasts **0.933 s** end to end, so the fixed gate
    kills everything and only the ratio gate (0.075 s, about two rig frames) is live.
    `LoopClosureDiagnostics.min_time_gap_seconds` reports which gap was applied.
12. **Brute-force retrieval collapses at 4528 images.** The cost is
    `(n_images * n_descriptors)^2 * 128` inner products, so `max_total_descriptors` divides a
    fixed budget across the images. Galileo's 226 images get 256 descriptors each and the
    retrieval works (9.4 s). RoboCap's 4528 images get 22 each, the largest off-diagonal score
    is 0.227, and after the temporal gate nothing clears the score threshold: zero loop
    candidates on a 124 m walk with 90 genuine revisits. Raising the cap is not the fix. Giving
    those 4528 images 256 descriptors each costs 3.4e14 FLOP, about 40 minutes at this
    machine's measured 147 GFLOP/s float32 GEMM; using every descriptor costs 2.2e16 FLOP,
    about 41 hours. That quadratic is why the vocabulary backend exists.

### How to run

```bash
pixi run colsfm-galileo    # 226 frames, 8 pinhole cameras, about 19 s
pixi run colsfm-robocap    # 4528 frames, 4 fisheye cameras, about 7 min
pixi run colsfm-test       # tests/colsfm
pixi run colsfm-probe      # tests/colsfm_probes, the pycolmap capability suite
pixi run colsfm-lint       # ruff over colsfm and tests/colsfm
```

Both tasks call `python -m colsfm run`, which takes the same `--input-dir` as
`cusfm_cli` and writes the same output layout. Useful flags:

```bash
--loop-closure                 # run stage 5; off by default (decision 10)
--no-use-gpu                   # force the ONNX CPU provider
--max-matches-per-pair 500     # verified matches kept per pair; None keeps every inlier
--ba-num-threads 1             # a bit-reproducible Ceres solve
--optimize-extrinsics          # refine sensor_from_rig during bundle adjustment
```

`python -m colsfm stage --stage pair_selection` runs a metadata-only stage. It needs neither
images nor a GPU, so a dataset's keyframe and pair counts can be checked before paying for
feature extraction.

The benchmark compares two runs in the same output layout and knows nothing about which
producer wrote which:

```bash
pixi run -e colsfm python -m colsfm.bench_cli \
    --dataset galileo \
    --run-a data/cusfm_runs/galileo_blobref/cusfm \
    --run-b data/cusfm_runs/galileo_colsfm/cusfm \
    --input-dir data/r2b_galileo \
    --save data/bench/galileo_compare.rrd \
    --report data/bench/galileo_compare.md
```

It prints the tables, writes markdown and JSON, checks the acceptance bounds against run B,
and writes one Rerun recording with run A at `/world/rig_00` and run B at `/world/rig_01`.

To validate that recording without a display, open it in a headless viewer and take pixel
evidence through the Rerun viewer MCP:

```bash
pixi run -e colsfm rerun --headless data/bench/galileo_compare.rrd
```

Then `connect`, `viewer_state` for the timelines, `set_time` onto `frame_time`, and
`screenshot`. Screenshots of the accepted result are kept in `data/bench/screenshots/`. A
successful exit and a written `.rrd` are not evidence that anything rendered.

### Reverse-engineering toolchain

The 21 executables are **unstripped**, with C++ symbols and embedded source paths
(`/home/jryu/workspaces/visual_mapping/src/...`), which is what makes the specs possible at
all. Decompilation used **Ghidra 12.1.3**, unpacked project-local under
`data/cusfm_re/ghidra_12.1.3_PUBLIC/` and driven headless with a JDK supplied per invocation
by `pixi exec -s openjdk=21`, so nothing is installed on the host and nothing lands in a
pixi environment. Output goes to `data/cusfm_re/decomp/<binary>/`, one `.c` per function
(389 functions for `pose_graph_main` alone), selected by
`ghidra_scripts/DecompMatching.java` (name substrings) or `DecompAt.java` (entry addresses,
needed to tell overloaded symbols such as the two `EstimateRelativePose` apart). A second
shell with radare2, rizin, pyghidra and binwalk comes from a nix flake through
`data/cusfm_re/tools/re-shell.sh`, which runs **nix-portable** in its proot fallback:
`pixi exec -s nix` cannot build here, because the host has no user namespaces and a rootless
daemon therefore cannot start. The protobuf schema comes out of the binaries themselves: each
one embeds `FileDescriptorProto`s in its `protodesc_cold` ELF section, and all **31** `.proto`
files across the 21 binaries are now vendored as
`data/cusfm_schema/cusfm_protos.fdset` (up from the original 19), verified by re-serialising
all 516 keyframes of a real `feature_extractor_main` run byte for byte. gflags help for every
binary is dumped to `data/cusfm_re/help/<binary>.txt`, which is how flags that no config
mentions were found. All of `data/cusfm_re/` is gitignored scratch; only the specs, the
fdset and the numbers in this file are kept.
