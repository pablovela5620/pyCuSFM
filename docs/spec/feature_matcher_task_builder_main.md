# `feature_matcher_task_builder_main` — implementation spec

Status: reverse-engineered from `pycusfm/x86_cuda13/bin/feature_matcher_task_builder_main`
(unstripped ELF, C++ symbols, embedded source paths `/home/jryu/workspaces/visual_mapping/src/...`),
Ghidra decompilation under `data/cusfm_re/decomp/task_builder/`, the shipped configs, the completed
runs under `data/cusfm_runs/galileo/`, and **re-runs of the binary with mutated configs** under
`data/cusfm_re/runs/`. Evidence and confidence per claim: **[C]** confirmed (decompiled body,
string literal, or exact numeric match against a real or re-run artifact), **[L]** likely,
**[I]** inferred.

---

## 1. Purpose and position in the pipeline

Stage 5. It reads keyframe metadata plus a pose graph (or does a spatial search), decides
**which image pairs will be matched**, and writes them as `FeatureMatchingTask` protobufs that
`feature_matcher_main` consumes — one task file per worker process.

```
keyframes/                        ─┐
pose_graph/frames_meta.json       ─┤
pose_graph/pose_graph.pb.txt      ─┼─► feature_matcher_task_builder_main ─► matches/tasks/matching_task_000.pb
cuvgl_map/{vocabulary,bow_index}  ─┤                                        matches/tasks/matching_task_001.pb ...
configs/<profile>/match_pair_...  ─┘
```

It runs no GPU work despite linking TensorRT: it constructs a `FeatureMatcher` (which can build a
LightGlue matcher) but only calls `BuildMatchingTask`. **[C]**
(`data/cusfm_re/decomp/task_builder/main_00138820.c:163`)

Call site: `pycusfm/cusfm_runner.py:665-694`. The runner always passes
`--pose_graph_association_dir` (either `pose_graph/` when the pose-graph stage ran, or
`associations/` from `generate_association_main`), and `--downsampling_matches=<bool>`. **[C]**

### Headline finding for the Python port

With the shipped `isaac` config (`search_method: POSE_GRAPH_ASSOCIATION`) the task builder is a
**pure filter over pose-graph edges**. It performs no spatial search, no image retrieval and
**no downsampling at all** — every downsampling knob (`use_downsampling`,
`max_keyframes_per_collection`, `collection_time_interval_seconds`,
`min_time_interval_between_keyframes_seconds`, `min_distance_between_keyframes_meters`) and the
`--downsampling_matches` flag are dead in this mode. Verified by a 13-point config sweep on
Galileo: every variant produced exactly 339 candidate pairs
(`data/cusfm_re/analysis/sweep_taskbuilder.sh`). **[C]**

The cuSfM paper (arXiv:2510.15271, §3.4) states the same design in prose: "constructing the view
graph exclusively from pose graph edges achieves successful SfM reconstruction while significantly
reducing computational overhead", and lists the pose graph's three constraint categories as
sequential pairs from the initial trajectory (adjacent frames, k=1), loop closures, and "extrinsic
constraints between synchronized multi-camera captures" — exactly the `CONSECUTIVE` / `LOOP` /
`EXTRINSIC` types. Its radius-search example uses "20 meters with an angular deviation limit of
90 degrees", i.e. the `av` profile. **[C]**

The whole stage in this mode reduces to:

> For each `ConstraintPair` in `pose_graph.pb.txt`: keep it if its `type` is not `EXTRINSIC`;
> if it *is* `EXTRINSIC`, keep it only when the two frames' cameras form a declared stereo pair.
> Normalise each surviving pair to `(min(id), max(id))`, deduplicate, and write out in batches of
> `max_frame_pair_size_per_file`.

---

## 2. CLI flags

From `data/cusfm_re/help/feature_matcher_task_builder_main.txt` and
`pycusfm/cusfm_runner.py:673-707`. **[C]**

| Flag | Type | Default | Used by the isaac pipeline | Meaning |
|---|---|---|---|---|
| `--current_keyframe_directory` | string | "" | yes (`keyframes/`) | Keyframe root; also where `frames_meta.json` is looked for when `--keyframe_metadata_file` is unset |
| `--output_dir` | string | "" | yes (`matches/tasks/`) | Destination for `matching_task_%03zu.pb` |
| `--match_pair_select_config` | string | `configs/keymatch_pair_select_config.pb.txt` | yes | `MatchPairSelectConfig` text proto |
| `--pose_graph_association_dir` | string | "" | yes | Directory holding `pose_graph.pb.txt` |
| `--keyframe_metadata_file` | string | "" | yes | Overrides the metadata path (the runner points it at `pose_graph/frames_meta.json`) |
| `--downsampling_matches` | bool | true | yes | **Overwrites** `MatchPairSelectConfig.use_downsampling` (see §4.1) |
| `--current_retrieval_db_dir` | string | "" | when `cuvgl_map/` exists | BoW vocabulary + index; only consulted by `IMAGE_RETRIEVAL` / `TRACK_2D_OVERLAP_IR` |
| `--max_buffer_size` | int32 | 2000 | no (default) | Keyframe LRU cache size |
| `--previous_cusfm_ws` | string | "" | patch mode only | Enables incremental/patch mode; also triggers `WriteNewSessionFrameMetadata` |
| `--built_map_cusfm_directory` | string | "" | no | Prebuilt map for localisation-style matching |

`--v=N --alsologtostderr` give the useful progress lines (`match_stats.cc`, `feature_matcher.cc`).

---

## 3. Input and output formats

### 3.1 `MatchPairSelectConfig` (`protos/visual/cusfm/matching_config.proto`)

Decoded from `data/cusfm_schema/cusfm_protos.fdset` → `data/cusfm_re/proto/matching_config.proto`.
**[C]**

```proto
message MatchPairSelectConfig {
  optional double search_translation_threshold_meters          = 1;   // struct +0x10
  optional double search_rotation_threshold_degrees            = 2;   // +0x18
  optional int64  max_frame_pair_size_per_file                 = 3;   // +0x20
  optional double min_time_interval_between_keyframes_seconds  = 4;   // +0x28
  optional double min_distance_between_keyframes_meters        = 5;   // +0x30
  optional double collection_time_interval_seconds             = 6;   // +0x38
  optional uint32 max_keyframes_per_collection                 = 7;   // +0x40
  optional KeyFrameSearchMethod search_method                  = 9;   // +0x44
  optional bool   use_downsampling                             = 10;  // +0x48
  optional int32  query_result_number                          = 11;
  optional int32  min_shared_word_number                       = 12;
  optional double far_plane_distance                           = 13;
}
enum KeyFrameSearchMethod { RADIUS_SEARCH = 0; IMAGE_RETRIEVAL = 1; TRACK_2D_OVERLAP_IR = 2; POSE_GRAPH_ASSOCIATION = 3; }
```

The struct offsets in the comments are read off the decompiled code
(`BuildMatchingTask` dispatches on `*(int *)(cfg + 0x44)`, `BuildMatchingTaskByRadiusSearch`
squares `*(double *)(cfg + 0x10)`, `UpdateMatchingTaskAndSave` flushes on
`*(long *)(cfg + 0x20)`) and confirmed by the config sweep. **[C]**

Shipped values, `pycusfm/configs/isaac/match_pair_select_config.pb.txt:1-15`:

```
search_translation_threshold_meters: 3            # dead in POSE_GRAPH_ASSOCIATION
search_rotation_threshold_degrees: 60             # dead in POSE_GRAPH_ASSOCIATION
max_frame_pair_size_per_file: 100000              # LIVE
min_time_interval_between_keyframes_seconds: 0.5  # dead
min_distance_between_keyframes_meters: 0.05       # dead
collection_time_interval_seconds: 10              # dead
max_keyframes_per_collection: 6                   # dead
search_method: POSE_GRAPH_ASSOCIATION             # LIVE
use_downsampling: true                            # overwritten by --downsampling_matches
query_result_number: 20                           # dead
min_shared_word_number: 50                        # dead
```

`pycusfm/configs/av/match_pair_select_config.pb.txt` differs: `search_method:
TRACK_2D_OVERLAP_IR`, `use_downsampling: false`, thresholds 20 m / 90°,
`collection_time_interval_seconds: 300`, `far_plane_distance: 50`. **[C]**

### 3.2 Pose graph input (`protos/visual/cusfm/pose_graph.proto`)

This proto is **not** in `cusfm_schema/cusfm_protos.fdset`; it was carved out of the binary with
`data/cusfm_re/analysis/extract_fdp.py` and written to `data/cusfm_re/proto/pose_graph.proto`. **[C]**

```proto
enum ConstraintType { UNKNOWN = 0; CONSECUTIVE = 1; LOOP = 2; EXTRINSIC = 3; }
message ConstraintPair {
  optional FramePair       frame_pair            = 1;
  repeated FeatureMatch    matches               = 2;
  optional RigidTransform3d pose_source_to_target = 3;
  optional MatrixD         pose_covariance       = 4;
  optional ConstraintType  type                  = 5;
  repeated bool            matches_inliers       = 6;
  optional int32           matched_pairs_num     = 7;
  optional bool            is_good               = 8;
  optional double          time_delta_source_to_target = 9;
}
message PoseGraph { repeated ConstraintPair pairs = 1; }
```

Read as **text proto** from `<pose_graph_association_dir>/pose_graph.pb.txt`
(`FileUtils::ReadProtoFileByExtension`). **[C]**

### 3.3 Task output

`FeatureMatchingTask { repeated FramePair frame_pairs = 1; }`, `FramePair { uint64
source_frame_id = 1; uint64 target_frame_id = 2; }`, serialised as **binary** protobuf to
`<output_dir>/matching_task_%03zu.pb` (zero-padded 3-digit counter starting at 000). **[C]**
(`FeatureMatcher::SaveTaskFile`, `data/cusfm_re/decomp/task_builder/…SaveTaskFile_00146700.c`)

Frame ids are the `KeyframeMetaData.id` values from `frames_meta.json`; **`source_frame_id <
target_frame_id` always** (see §4.3). Verified over all 339 pairs of the Galileo run. **[C]**

In patch mode (`--previous_cusfm_ws` non-empty) an extra `frames_meta.json` is written into
`--output_dir` by `WriteNewSessionFrameMetadata`, and `cusfm_runner.py:709-712` then feeds
`matches/tasks/frames_meta.json` to the matcher. **[C]**

---

## 4. Algorithm

`FeatureMatcher::BuildMatchingTask(config, output_dir)`
(`data/cusfm_re/decomp/task_builder/…BuildMatchingTask_00150b00.c`): **[C]**

```
EnsureDirectoryExists(output_dir)
switch (config.search_method):
  0 RADIUS_SEARCH          -> BuildMatchingTaskByRadiusSearch
  1 IMAGE_RETRIEVAL        -> BuildMatchingTaskByImageRetrieval
  2 TRACK_2D_OVERLAP_IR    -> BuildMatchingTaskByTrack
  3 POSE_GRAPH_ASSOCIATION -> BuildMatchingTaskByPoseGraph
  else                     -> LOG(FATAL)
```

All four paths funnel into the same `UpdateMatchingTaskAndSave` accumulator (§4.3), then
`PrintMatchPairStats` (§4.5).

### 4.1 The `--downsampling_matches` flag overwrites the config

`main` parses the `MatchPairSelectConfig`, then unconditionally assigns the gflag into the
config's `use_downsampling` slot (`+0x48`) before calling `BuildMatchingTask`:

```c
// data/cusfm_re/decomp/task_builder/main_00138820.c:100-102
if (fLB::FLAGS_downsampling_matches != local_318) { local_318 = fLB::FLAGS_downsampling_matches; }
// local_318 == local_360 + 0x48 == MatchPairSelectConfig::use_downsampling
```
**[C]** — confirmed dynamically: with `search_method: RADIUS_SEARCH`,
`use_downsampling: false` in the config still yields 768 pairs (the *downsampled* count) while
`--downsampling_matches=false` yields 6273. `data/cusfm_re/analysis/sweep_radius.sh`.

### 4.2 `POSE_GRAPH_ASSOCIATION` (the shipped isaac path)

`FeatureMatcher::BuildMatchingTaskByPoseGraph`
(`data/cusfm_re/decomp/task_builder/…BuildMatchingTaskByPoseGraph_00150380.c`): **[C]**

1. Require `pose_graph_association_dir` non-empty and existing; otherwise `LOG(FATAL)`
   (`pose_graph_dir is missing, but choose POSE_GRAPH_ASSOCIATION`).
2. Read `<dir>/pose_graph.pb.txt` into a `PoseGraph`.
3. Build `left_camera_params_id -> right_camera_params_id` from the metadata's `stereo_pair`
   list (`KeyframesMetadataCollection::GetLeftToRightCameraIDMap`).
4. For each `ConstraintPair`:
   * if `type != EXTRINSIC` (i.e. `CONSECUTIVE`, `LOOP`, `UNKNOWN`) → **accept**;
   * if `type == EXTRINSIC` → look up both frames' `camera_params_id`; accept **iff** the pair is
     a declared stereo pair, tested in the orientation given by
     `KeyframesMetadataCollection::IsLeftCamera(cam_of_source)`:
     `left_to_right[left_cam] == right_cam`. Reject otherwise.
   * accepted → `UpdateMatchingTaskAndSave(source, target, …)`.
5. Log `add <N> stereo pair extrinsic frames.` where `N` counts the **directed** EXTRINSIC edges
   that passed the stereo test.

Numeric confirmation on `data/cusfm_runs/galileo/cusfm` (226 keyframes, 8 cameras, 4 stereo rigs):

| quantity | pose graph | emitted |
|---|---|---|
| `CONSECUTIVE` edges | 218 | 218 (all) |
| `LOOP` edges | 8 | 8 (all, re-ordered to `min<max`) |
| `EXTRINSIC` edges (directed) | 1533 | 113 undirected stereo pairs |
| **total task pairs** | | **339** |

Re-running the binary reproduces 339 exactly. Replacing the pose graph with a
CONSECUTIVE-only copy yields `add 0 stereo pair extrinsic frames` and 218 pairs; an empty pose
graph yields 0 pairs and 0 task files (`data/cusfm_re/runs/pg_variants/`). An independent
Python reimplementation of the stereo test reproduces the log counter exactly on both runs
(221 for the 226-frame run, 24 for the 32-frame smoke run). **[C]**

Note the stage therefore *cannot* invent cross-camera pairs beyond the stereo baseline; all
cross-camera and loop connectivity comes from `generate_association_main` / `pose_graph_main`.

### 4.3 `UpdateMatchingTaskAndSave` — ordering, dedup, file split

(`data/cusfm_re/decomp/task_builder/…UpdateMatchingTaskAndSave_0014d780.c`) **[C]**

```
lo = min(source, target); hi = max(source, target)
if (lo, hi) not in emitted_set:            # std::set<FramePairID>, lexicographic on (lo, hi)
    task.frame_pairs.add(source_frame_id=lo, target_frame_id=hi)
    emitted_set.insert((lo, hi))
if config.max_frame_pair_size_per_file > 0 and task.frame_pairs_size() >= that value:
    SaveTaskFile(index, output_dir, task)   # matching_task_%03zu.pb
    task.frame_pairs.Clear()
    index += 1
```

The tail (a partial batch) is flushed by the caller after the loop. With
`max_frame_pair_size_per_file: 100` the Galileo run splits 339 pairs into **4** files
(100/100/100/39) — verified by re-run. **[C]**

Consequence for a Python port: pair ids are canonically ordered and unique, so a `set` of
`tuple(sorted(pair))` is a faithful model.

### 4.4 `RADIUS_SEARCH` and its downsampling (not used by isaac, used by nothing shipped)

`FeatureMatcher::BuildMatchingTaskByRadiusSearch`
(`…BuildMatchingTaskByRadiusSearch_0014d9b0.c`): **[C]**

1. `ExtractFramePosesAndLocations` builds a `nanoflann` KD-tree over the float32 camera
   positions of all keyframes (plus a second tree over the prebuilt map, if any).
2. For each frame `q`: `RadiusSearch(pos[q], r² = search_translation_threshold_meters²)` →
   candidate ids (self excluded).
3. If `search_rotation_threshold_degrees > 0`: `DataSelector::FilterByAngularDistance` keeps
   candidate `c` iff `2·atan2(‖vec(q_q⁻¹·q_c)‖, |w|) ≤ threshold·π/180`, i.e. the relative
   rotation angle is within the threshold. Rejected candidates increment the counter logged as
   `… candidates by angular distance = N`. **[C]**
4. If `use_downsampling` (post-flag-override): `DataSelector::FrameSubsetByTimeSpace(q,
   candidates, metadata_map, pose_map, collection_time_interval_seconds,
   search_rotation_threshold_degrees, min_distance_between_keyframes_meters,
   min_time_interval_between_keyframes_seconds, (double)max_keyframes_per_collection)`.
5. Emit each surviving candidate through `UpdateMatchingTaskAndSave`.

`FrameSubsetByTimeSpace` (`…FrameSubsetByTimeSpace_00189360.c`): **[C]**

* Drop candidates missing from either the metadata map or the pose map.
* Sort by `timestamp_microseconds`.
* Split into **collections**: walk in time order, start a new collection whenever
  `|t_i − t_first_of_current_collection| ≥ collection_time_interval_seconds · 1e6`.
* For each collection with more than `max_keyframes_per_collection` members, sort it with the
  comparator `DataSelector::SortKeyframesByTimeSpace` and truncate to
  `max_keyframes_per_collection`.
* Concatenate the kept ids.

`SortKeyframesByTimeSpace(a, b, query, pose_a, pose_b, pose_query, rot_rad, min_time, min_dist)`
returns "a before b" using three tiers (`…SortKeyframesByTimeSpace_00186600.c`): **[C]**

```
ang_x = 2*atan2(|vec(q_query^-1 * q_x)|, |w|)            # relative rotation, radians
if ang_a <= rot and ang_b >  rot: a first
if ang_a >  rot and ang_b <= rot: b first
d_x = ||t_query - t_x||                                   # metres
if d_a <= min_dist and d_b >  min_dist: b first           # PREFERS the more distant candidate
if d_a >  min_dist and d_b <= min_dist: a first
dt_x = |t_query - t_x| / 1e6                              # SECONDS
if dt_a <= min_time and dt_b >  min_time: b first         # prefers the more separated candidate
if dt_a >  min_time and dt_b <= min_time: a first
return dt_a < dt_b                                        # else: temporally nearest first
```

**Unit bug [C]:** the caller passes `min_time_interval_between_keyframes_seconds · 1e6` while the
comparator compares it against a value already converted to *seconds*. With the shipped 0.5 the
effective threshold is 5·10⁵ s, so the third tier degenerates to "temporally nearest first". This
is observable: setting `min_time_interval_between_keyframes_seconds: 0.0` changes the Galileo
RADIUS_SEARCH pair count from 768 to 756 (only the `dt == 0` stereo partners change rank), while
0.1 and 0.5 are indistinguishable. A Python port should reproduce the *behaviour*, not the
formula, or simply drop the tier.

Sweep evidence for the RADIUS_SEARCH path on Galileo (226 frames, all 25425 possible pairs
within 3 m of each other) — `data/cusfm_re/analysis/sweep_radius.sh`: **[C]**

| variant | pairs |
|---|---|
| baseline (3 m, 60°, downsampling on) | 768 |
| `--downsampling_matches=false` | 6273 |
| `max_keyframes_per_collection` 2 / 6 / 12 | 260 / 768 / 1527 |
| `collection_time_interval_seconds` 10 / 0.2 | 768 / 4141 |
| `min_distance_between_keyframes_meters` 0 / 0.05 / 0.2 | 759 / 768 / 827 |
| `min_time_interval_between_keyframes_seconds` 0 / 0.1 / 0.5 | 756 / 768 / 768 |
| `search_translation_threshold_meters` 0.1 / 3 / 10 (no downsampling) | 1077 / 6273 / 6273 |
| `search_rotation_threshold_degrees` 10 / 60 / 180 (no downsampling) | 4300 / 6273 / 25425 |
| `use_downsampling: false` in the config file | 768 (ignored — see §4.1) |

### 4.5 Statistics logging

`PrintMatchPairStats<KeyframesMetadataCollection>` (`src/visual/utils/match_stats.cc`)
classifies every emitted pair into `In-track sequential(<10s)` / `In-track loop close` /
`Cross-track` and prints per-track counts plus `Num image pairs per frame: Min/Max/Median/10%/90%`.
The 10 s constant is a format-string literal `In-track sequential(<%.0fs)`. Purely diagnostic. **[C]**

---

## 5. Real-artifact statistics (Galileo, `data/cusfm_runs/galileo/cusfm`)

Parsed with `data/cusfm_re/analysis/parse_matches.py` and
`data/cusfm_re/analysis/pairs_vs_graph.py`. **[C]**

* 226 keyframes, 8 cameras (4 stereo rigs), 105 distinct timestamps spanning 0.93 s,
  29 synced samples.
* 339 pairs in one task file. Pairs per frame: min 2, max 4, median 3; every frame is covered.
* 224 same-camera pairs (218 consecutive-in-time, index delta 1; 6 with delta 3 from LOOP edges)
  and 115 cross-camera pairs (113 same-timestamp stereo EXTRINSIC edges, plus 2 LOOP edges
  between the left rig's two cameras across a 0.167 s gap).
* Cross-camera pairs occur **only** inside a stereo rig — never front↔left etc.
* Pair separations: `dt` median 0.033 s (max 0.167 s), baseline median 0.026 m (max 0.150 m),
  relative rotation median 0.78° (max 3.44°).

---

## 6. Python replacement

### 6.1 What already exists in this repo

Nothing. `tools/raco_extract.py` stops at keyframe extraction; there is no pair-selection code.

### 6.2 What to write (isaac / POSE_GRAPH_ASSOCIATION parity)

```python
def select_pairs(pose_graph: PoseGraph, meta: KeyframesMetadataCollection) -> list[tuple[int, int]]:
    left_to_right = {sp.left_camera_param_id: sp.right_camera_param_id for sp in meta.stereo_pair}
    cam = {kf.id: kf.camera_params_id for kf in meta.keyframes_metadata}
    seen: set[tuple[int, int]] = set()
    for edge in pose_graph.pairs:
        s, t = edge.frame_pair.source_frame_id, edge.frame_pair.target_frame_id
        if edge.type == ConstraintType.EXTRINSIC:
            cs, ct = cam[s], cam[t]
            if left_to_right.get(cs) != ct and left_to_right.get(ct) != cs:
                continue
        seen.add((min(s, t), max(s, t)))
    return sorted(seen)          # insertion order in the blob is graph order; sort is a superset
```

Ordering caveat: the blob emits pairs in **pose-graph file order**, not sorted order. That only
matters if you want byte-identical task files or identical file splits; the downstream matcher is
order-independent. **[I]**

Batch into files of `max_frame_pair_size_per_file` if you keep the multi-process matcher; a
single-process Python matcher can skip the task files entirely and pass the list in memory.

### 6.3 Mapping to pycolmap

pycolmap has no equivalent of this stage — its built-in matchers (`match_exhaustive`,
`match_sequential`, `match_spatial`, `match_vocabtree`) choose their own pairs. The faithful
route is `pycolmap.match_pairs` / a custom pair list fed to
`pycolmap.Database.write_two_view_geometry`, driven by the list produced above.

Rough equivalences if you ever want to *replace* rather than mirror cuSFM's choice:

| cuSFM | pycolmap |
|---|---|
| `POSE_GRAPH_ASSOCIATION` (consecutive + loop + stereo) | `SequentialMatchingOptions` (overlap) + explicit rig pairs + `loop_detection` |
| `RADIUS_SEARCH` (`search_translation_threshold_meters`) | `SpatialMatchingOptions(max_distance=…, max_num_neighbors=…)` |
| `IMAGE_RETRIEVAL` (`query_result_number`, `min_shared_word_number`) | `VocabTreeMatchingOptions(num_images, num_nearest_neighbors)` |
| `FrameSubsetByTimeSpace` downsampling | no equivalent; `max_num_neighbors` is the closest knob |

For Galileo parity, the pair set must include **every consecutive same-camera pair and every
stereo pair**; that is what produces the 176 points/image density noted in `NOTES.md`.

### 6.4 Cost model

Trivially cheap: the blob spends 1.2 s on this stage, almost all of it loading the 5.4 M-node BoW
vocabulary that `POSE_GRAPH_ASSOCIATION` never uses (`Total time used: 1.30199s` of which
~1.25 s is `visual_vocabulary.cc`). Skipping the vocabulary makes the Python version
sub-millisecond. **[C]**

---

## 7. Evidence index

| Claim | Evidence |
|---|---|
| Search-method dispatch | `data/cusfm_re/decomp/task_builder/nvidia__isaac__visual__cusfm__FeatureMatcher__BuildMatchingTask_00150b00.c` |
| Pose-graph filter + stereo test | `…BuildMatchingTaskByPoseGraph_00150380.c`; `feature_matcher.cc:0x31e,0x32a,0x369` |
| `min<max` ordering, dedup, file split | `…UpdateMatchingTaskAndSave_0014d780.c`, `…SaveTaskFile_00146700.c` |
| Radius search + KD-tree | `…BuildMatchingTaskByRadiusSearch_0014d9b0.c` |
| Angular filter | `…FilterByAngularDistance_00188890.c` |
| Collection downsampling | `…FrameSubsetByTimeSpace_00189360.c`, `…SortKeyframesByTimeSpace_00186600.c` |
| Flag overrides config | `data/cusfm_re/decomp/task_builder/main_00138820.c:100` |
| `ConstraintType` enum values | `data/cusfm_re/proto/pose_graph.proto` (carved by `data/cusfm_re/analysis/extract_fdp.py`) |
| Config sweeps | `data/cusfm_re/analysis/sweep_taskbuilder.sh`, `data/cusfm_re/analysis/sweep_radius.sh`, outputs in `data/cusfm_re/runs/` |
| Artifact statistics | `data/cusfm_re/analysis/parse_matches.py`, `data/cusfm_re/analysis/pairs_vs_graph.py` |

## 8. Open questions

1. **Emission order.** The blob writes pairs in pose-graph order. If task-file byte parity ever
   matters (it should not), the Python port must preserve that order rather than sorting. **[I]**
2. **`IsLeftCamera` tie-breaking.** For an EXTRINSIC edge where both cameras are "left" of
   different rigs the decompiled branch structure is asymmetric; on all real data the test is
   symmetric and my Python reimplementation matches the blob's counter exactly, so the branch
   never diverges in practice. **[L]**
3. **`TRACK_2D_OVERLAP_IR`** (the `av` profile) was not analysed: it uses
   `GetCameraFOVArea2D` / `GetFrustum2DFromProject` / `CalculateOverlapRatio` plus BoW
   verification, and `far_plane_distance`. Out of scope for the isaac pipeline. **[I]**
4. **`--built_map_cusfm_directory` / `--previous_cusfm_ws`** (patch and localisation modes) add a
   second KD-tree and a second keyframe pool; not exercised by `cusfm_runner.py` defaults. **[I]**
