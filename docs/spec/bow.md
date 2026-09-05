# Stage 2 — BoW retrieval: `generate_bow_vocabulary_main` + `generate_bow_index_main`

Reverse-engineered spec for the cuSFM image-retrieval stage. Target of the Python +
pycolmap reimplementation described in `docs/open-pipeline-plan.md`.

Confidence tags used below:

- **CONFIRMED** — read from decompiled code, disassembly, string literals, or reproduced
  numerically from the shipped artifacts.
- **INFERRED** — deduced from artifacts or partial code; plausible but not read directly.

Headline result: a 60-line NumPy reimplementation of the *query/encode* half reproduces the
blob's `bow_index.pb` **exactly** (identical word sets, weights to 1.1e-5 relative) on both
Galileo runs. See "Numerical verification" below.

---

## 1. Purpose and position in the pipeline

| | |
|---|---|
| Consumes | `keyframes/` from stage 1 (`feature_extractor_main`): `frames_meta.json` + one `Keyframe` proto per image |
| Produces | `cuvgl_map/vocabulary/` (visual vocabulary tree) and `cuvgl_map/bow_index.pb` (inverted file) |
| Consumed by | `generate_association_main` (loop-closure candidates → pose graph), `pose_graph_main`, `feature_matcher_task_builder_main` (`--current_retrieval_db_dir`), `kpmap_to_colmap` (`--vocabulary_dir`, `--bow_index_file`) |

The stage exists purely to answer "which other keyframes look like this keyframe?" so that the
association/pose-graph stages can propose loop-closure pairs without all-pairs matching. It has
**no effect on geometry** by itself: it produces candidate frame ids and similarity scores.

The vocabulary is **environment-specific**: it is trained on the very keyframes it will index
(no pretrained/off-the-shelf tree ships with the blob). CONFIRMED — `build_history` in the
shipped `bow_vocabulary.pb.txt` is the run's own keyframe directory, and
`cusfm_runner.py:1007` passes the same `self.keyframe_dir` to both binaries.

Paper (arXiv 2510.15271, HTML) corroborates: "recursive K-means clustering to generate a
hierarchical tree structure … leaf nodes define the final vocabulary words"; "each keyframe is
added to a database augmented with an inverted index (loop index) that links visual words to
keyframes".

---

## 2. CLI flags actually used

From `pycusfm/cusfm_runner.py::generate_bow` (line 528) — the only caller. Both are skipped
entirely when `--skip_vocab_generator` is set.

### `generate_bow_vocabulary_main`

| Flag | Meaning |
|---|---|
| `--keyframe_directory` | Directory holding `frames_meta.json` and the per-camera `<cam>/<ts>.pb` keyframe protos. Training corpus. |
| `--image_retrieval_config` | Path to `configs/<profile>/image_retrieval_config.pb.txt` (`protos.visual.loop_closing.ImageRetrievalConfig`, text format). |
| `--output_voc_dir` | Output directory. The runner sets it to `<cuvgl_dir>/vocabulary`; it is wiped and recreated first. |

Declared but **never passed** by the runner:

| Flag | Meaning |
|---|---|
| `--input_voc_dir` | If non-empty *and* the directory exists, the existing vocabulary is loaded and extended incrementally (BIRCH path). Empty ⇒ "Creating a new vocabulary". |

### `generate_bow_index_main`

| Flag | Meaning |
|---|---|
| `--keyframe_directory` | Same keyframe directory. Every frame id in `frames_meta.json` is indexed. |
| `--output_map_dir` | The **cuvgl map** directory. The binary reads the vocabulary from `<output_map_dir>/vocabulary` and writes `<output_map_dir>/bow_index.pb`. |
| `--image_retrieval_config` | Same config file. Only `shared_node_levelsup` is read. |

CONFIRMED — `main` disassembly of `generate_bow_index_main` (`0x23550`): `JoinPath(FLAGS_output_map_dir, kVocabularyFolderName="vocabulary")`, then
`LoopClosureIndex::Init(vocabulary_dir, config.shared_node_levelsup())` (CHECK string
`"db.Init(vocabulary_dir, image_retrieval_config.shared_node_levelsup())"`), then
`JoinPath(FLAGS_output_map_dir, kImageRetrievalDatabaseFileName="bow_index.pb")`.

Both binaries force `FLAGS_alsologtostderr=true` and `FLAGS_colorlogtostderr=true` in `main`.

---

## 3. Config: `protos.visual.loop_closing.ImageRetrievalConfig`

Schema rendered from `data/cusfm_re/protos/generate_bow_index_main.fdset`. Values are from
`pycusfm/configs/isaac/image_retrieval_config.pb.txt` (identical to the `av/` copy).

| # | Field | Type | isaac value | Used by | Effect | Confidence |
|---|---|---|---|---|---|---|
| 1 | `node_branch_number` | uint32 | `9` | voc | k of the per-node k-means; branching factor of the tree. | CONFIRMED (`VisualVocabulary::Init` arg 1 → `this+0x4c`; expected-node-number log = `(9^8-1)/8 = 5380840`) |
| 2 | `tree_depth_levels` | uint32 | `7` | voc, idx | Max tree depth; also fixes the "shared node" level at query time. | CONFIRMED (`Init` arg 2 → `this+0x50`; used in `SearchFeature` as `depth - levelsup`) |
| 3 | `incremental_build_radius_threshold` | float | `0.1` | voc | BIRCH CF-tree radius threshold. Only reached on the incremental path (`--input_voc_dir`, or >1 segment). Not exercised by the isaac runs. | CONFIRMED (`Init` arg 3 → `this+0x98`) |
| 4 | `weighted_kmeans_number_of_thread` | uint32 | `12` | voc | Worker threads in `MultiThreadKmeansBuilder`. Performance only. | CONFIRMED (`cfg+0x1c` → ctor `param_4` → `this+0x1f8`) |
| 5 | `weighted_kmeans_max_number_of_iterations` | uint32 | `10` | voc | Max Lloyd iterations per node split; early exit when the assignment hash is unchanged. | CONFIRMED (`cfg+0x20` → ctor `param_3` → `this+0x1fc`, loop in `PerformKMeansCluster`) |
| 6 | `shared_node_levelsup` | uint32 | `2` | idx | Levels above the leaf at which the "direct index" node is recorded (per-frame `node → feature indices` map, used for guided matching downstream). Stored in `bow_index.pb`. | CONFIRMED (main disassembly; `LoopClosureIndex.shared_node_levelsup = 2` in the artifact) |
| 7 | `query_result_number` | uint32 | `1` | *neither* | Top-N truncation at query time. Not used by these two binaries. | CONFIRMED (used by `LoopClosureIndex::CalculateCandidatesScore`, called from `DetectCandidates*` in the *association / matcher / pose-graph* binaries) |
| 8 | `min_shared_word_number` | uint32 | `0` | *neither* | Minimum number of words shared with the query for a candidate to survive. Same story. | CONFIRMED (`DetectCandidates` filter `cand.shared_words >= min`) |
| 9 | `min_inlier_number` | uint32 | `20` | *neither* | Geometric-verification threshold used downstream. | INFERRED |
| 10 | `max_reprojection_error` | float | `10` | *neither* | Downstream verification. | INFERRED |
| 11 | `remove_duplicate_feature` | bool | `true` | *neither* | No read of `cfg+0x38` found in either binary's BoW code paths. | INFERRED (absence of evidence) |
| 12 | `select_clustering_map_frames` | bool | `true` | *neither* | Ditto (`cfg+0x39`). Both runs trained on **all** frames (32/32 and 226/226). | CONFIRMED-by-artifact that no subsampling happened; INFERRED that the flag is unread here |
| 13 | `apple_compress_for_super_point` | bool | `true` | voc | Only fires when the keyframe descriptor type is `SUPER_POINT_DESCRIPTOR(1)`; then the working descriptor type becomes `SUPER_POINT_QUANTIZED(5)` (uint8). **No-op for ALIKED.** | CONFIRMED (`main` at `0x235d0`: `if (desc_type==1 && flag) desc_type=5`; `BuildVocByKmeans` line `if (!cfg[0x3a] \|\| meta.descriptor_type != 1)`) |
| 14 | `float_node_centroid` | bool | `true` | voc | Selects `centroid_type`: `CENTROID_FLOAT32(103)` when true, `CENTROID_UINT8(102)` when false. Centroids are stored as raw float32 in `VocabularyNode.centroid`. | CONFIRMED (`Init`: `centroid_type = 0x66 + flag`; artifact header says `CENTROID_FLOAT32`) |
| 15 | `max_frame_number_in_memory` | int32 | `20000` | voc | Frames per build *segment*. `segments = ceil(num_frames / this)`. Segment 0 uses global weighted k-means; segments 1..N use the BIRCH incremental builder. `0` is a fatal-ish error ("please configure max_frame_number_in_memory in the configuration!"). | CONFIRMED (`BuildVoc`, `cfg+0x3c`) |
| 16 | `good_score_threshold` | double | *unset (0)* | *neither* | Minimum L1 similarity for a candidate to be kept. Set to `0.1` in `configs/isaac/association_config.pb.txt`'s nested `image_retrieval_config`. | CONFIRMED (`DetectCandidatesForMapper` param 4: `if (threshold <= cand.score)`) |
| 17 | `pose_translation_threshold_meters` | float | *unset (0)* | *neither* | Candidate rejected if the relative translation exceeds it (`1` in association config). | CONFIRMED (`DetectCandidatesForMapper`) |
| 18 | `pose_rotation_threshold_degrees` | float | *unset (0)* | *neither* | Same for rotation, computed as `2*acos(qw)*180/pi` (`10` in association config). | CONFIRMED (`acos(...)*57.29577951308232` in `DetectCandidatesForMapper`) |

**Key point for the reimplementation:** the config passed to these two binaries is a *build*
config. The retrieval knobs that actually matter live in the **downstream** configs:

- `configs/isaac/association_config.pb.txt` → `image_retrieval_config { min_shared_word_number: 50, query_result_number: 20, good_score_threshold: 0.1, pose_translation_threshold_meters: 1, pose_rotation_threshold_degrees: 10 }`
- `configs/isaac/pose_graph_config.pb.txt` → `loop_closure_min_words: 50`, `loop_closure_query_count: 20`, `loop_closure_interval_ratio: 0.08`, `good_loop_threshold: 150`
- `configs/isaac/match_pair_select_config.pb.txt` → `query_result_number: 20`, `min_shared_word_number: 50`

So the effective retrieval policy is *top-20 candidates, at least 50 shared words, L1 score
≥ 0.1, pose gate 1 m / 10°*.

---

## 4. Inputs

### 4.1 `frames_meta.json`

`protos.visual.general.KeyframesMetadataCollection` in protobuf-JSON. Fields read by the BoW
stage: CONFIRMED

- `keyframes_metadata[].id` (uint64) — the frame id used as the index key. Must fit in
  `uint32` (`"frame_id exceed the max uint32_t for keyframe database"`, `loop_closure_index.cc:64`).
- `keyframes_metadata[].image_name` — relative path (`front_stereo_camera_left/<ts>.jpeg`).
  The keyframe proto is the same path with the extension replaced by `.pb`.
- `descriptor_type` (enum) — must match what the vocabulary was trained on; the multi-thread
  builder logs `"Feature descriptor does not match, got: … Expecting: …"` and skips otherwise.
- `camera_params_id_to_camera_params` — only used to build a camera projector inside
  `KeyframeCacher::Init` ("Creating pinhole camera projector"). **Not** used for retrieval.

Poses are *not* used by these two binaries (they are used by the query-time pose gate in
`DetectCandidatesForMapper`, which lives downstream).

### 4.2 Keyframe protos

`protos.visual.general.Keyframe`, one per image, binary.

Only `keyframe.keypoint_vector` (`protos.common.image.KeypointVector`) is used:

- `descriptors` (bytes) — row-major `N × D` float32 for ALIKED / SuperPoint, `N × D` uint8 for
  SIFT / quantised SuperPoint.
- `descriptor_type` — `ALIKED_DESCRIPTOR = 4` for the isaac profile.
- `x`, `y` give `N`; keypoint coordinates themselves are unused by the BoW stage.
- The legacy repeated `keypoints` field is not used (`"Avoid using keypoint, use
  keypoint_vector instead"`).

Galileo observation (CONFIRMED): `N = 2048` keypoints/frame, `D = 128`, float32, and every
descriptor is already **L2-normalised** (‖d‖ = 1.0000 ± 3e-4). ALIKED emits unit-norm
descriptors; the BoW code never re-normalises.

`feature_dimension_size` is *not* read from the data — it is hard-coded per descriptor type in
`VisualVocabulary::Init`: 128 for `SIFT_CV_CUDA(2)` and `ALIKED(4)`, 256 for
`SUPER_POINT(1)` and `SUPER_POINT_QUANTIZED(5)`; anything else is `LOG(FATAL)`. CONFIRMED
(disassembly at `0x413a9`–`0x41417`).

---

## 5. Outputs

### 5.1 `<output_voc_dir>/bow_vocabulary.pb.txt` — `protos.visual.loop_closing.VisualVocabulary`

Text-format proto (extension decides the codec: `WriteProtoFileByExtension`). Real Galileo
226-frame artifact:

```
descriptor_type: ALIKED_DESCRIPTOR
apply_merge_before_split: true
total_training_feature_number: 462848
total_trained_image_frame_number: 226
feature_dimension_size: 128
branch_number: 9
cluster_tree_depth_levels: 7
node_collection_proto_file_number: 2
build_history: "data/cusfm_runs/galileo/cusfm/keyframes"
centroid_type: CENTROID_FLOAT32
```

| Field | Semantics | Confidence |
|---|---|---|
| `descriptor_type` | The working type (post `apple_compress` substitution). | CONFIRMED |
| `apply_merge_before_split` | **Hard-coded `true`** in `VisualVocabulary::Init` (`movb $0x1, 0x34(%rdi)`). BIRCH-path knob only. | CONFIRMED |
| `total_training_feature_number` | Sum over training frames of `len(keypoints)`. `226 × 2048 = 462848` — **no subsampling at all**. | CONFIRMED |
| `total_trained_image_frame_number` | `N`, the IDF document count. Accumulates across incremental builds. | CONFIRMED (`BuildVoc`: `this[0x40] += num_frames`) |
| `feature_dimension_size` | Fixed per descriptor type (see §4.2). | CONFIRMED |
| `branch_number`, `cluster_tree_depth_levels` | Copies of config 1 and 2. | CONFIRMED |
| `node_collection_proto_file_number` | Number of `bow_vocabulary_<i>.pb` shards. | CONFIRMED |
| `root_node_id` | 0 in practice (absent from the text proto because proto3 omits zeros). | CONFIRMED |
| `build_history` | Appended `--keyframe_directory` per build. | CONFIRMED |
| `centroid_type` | `CENTROID_FLOAT32 = 103` or `CENTROID_UINT8 = 102`. | CONFIRMED |

### 5.2 `<output_voc_dir>/bow_vocabulary_<i>.pb` — `protos.visual.loop_closing.VocabularyNodeCollection`

The **flat node array**, dumped in index order and split across shards. CONFIRMED by parsing
both Galileo artifacts.

- Node id == its index in the concatenation of all shards. (Verified: `id != index` count = 0
  over all 84 029 occupied nodes of the 32-frame run.)
- The array is a slab allocator: most entries are **free**, marked by `squared_sum == -1.0`
  (and `id == 0`, empty `centroid`, empty `children_ids`). 32-frame run: 5 354 668 slots,
  84 029 occupied, 5 270 639 free (matches the log `load 5354668,free_nodes:5270639`).
- Shard split: `nodes_per_file = 2000000000 / NodeSizeInByte` (integer division), file count =
  `ceil(total_nodes / nodes_per_file)`. CONFIRMED (`SaveVocFiles`, literal `2000000000`) —
  it is a 2 GB protobuf-size budget.
- `NodeSizeInByte` is `(feature_dimension_size + 6 + branch_number*2) * 4` when the descriptor
  type is ALIKED/SuperPoint **or** `centroid_type == FLOAT32`, else
  `feature_dimension_size + 24 + branch_number*8`. For isaac: `(128 + 6 + 18)*4 = 608`,
  matching the log `NodeSizeInByte:608`, so `nodes_per_file = 3289473`. CONFIRMED.

`VocabularyNode` fields:

| Field | Semantics | Confidence |
|---|---|---|
| `id` (uint64) | Index in the flat array. | CONFIRMED |
| `appeared_image_number` (uint32) | Document frequency `df` — number of training images in which this word occurs (deduplicated within a frame). Leaves only. | CONFIRMED (`SetWordWeight`) |
| `sample_number` (int32) | `N_i`, number of training descriptors assigned to this node's subtree. | CONFIRMED (root children sum to 65 536) |
| `squared_sum` (double) | BIRCH CF: `Σ‖x‖²` over member descriptors. `-1.0` means "free slot". For unit descriptors ≈ `sample_number`. | CONFIRMED |
| `children_ids` (repeated uint64) | Child node indices. Empty ⇒ leaf/word. Observed sizes 2..9 (k-means with fewer points than `k`, or empty clusters dropped). | CONFIRMED |
| `centroid` (bytes) | Raw little-endian float32 `feature_dimension_size` values (when `centroid_type == FLOAT32`); raw uint8 otherwise. Equals the arithmetic mean of member descriptors — **not** re-normalised (observed ‖c‖ ∈ [0.9993, 1.0007] for leaves, ~0.59 for a 121-member internal node). | CONFIRMED |

The node **weight** and **word id** are *not* serialised; both are recomputed at load time
(see §7.3).

Node layout in memory (`sizeof == 0x60`), useful when reading disassembly: `0x08`
`children_ids` vector, `0x20` `centroid` vector, `0x40` `appeared_image_number` (int), `0x50`
`squared_sum` (double), `0x58` `weight` (float), `0x5c` `word_id` (int). CONFIRMED.

### 5.3 `<output_map_dir>/bow_index.pb` — `protos.visual.loop_closing.LoopClosureIndex`

```proto
message OccurenceInfoCollection { repeated uint32 frame_ids = 1; repeated float weights = 2; }
message LoopClosureIndex {
  uint32 shared_node_levelsup = 1;
  repeated OccurenceInfoCollection word_to_occs = 3;   // one entry per word, dense
}
```

- `word_to_occs` has exactly one entry per word (65 533 / 436 358 for the two Galileo runs),
  in word-id order; words with no occurrence are present but empty.
- `frame_ids[j]` is `KeyframeMetaData.id` truncated to uint32.
- `weights[j]` is the **L1-normalised TF-IDF weight** of that word in that frame — i.e. the
  index stores the BoW vectors transposed, already normalised. Verified: for every frame,
  `Σ_w weight = 1.0` to 1.4e-5.
- The per-frame *direct index* (`node_at_levelsup → feature indices`) is built in memory by
  `LoopClosureIndex::Add` but **is not serialised**; only `shared_node_levelsup` is stored so
  a consumer can rebuild it. CONFIRMED (proto has no such field; `ToProto` writes only the two
  fields).

Galileo sizes: 32 frames → 751 KB; 226 frames → 5.1 MB. Vocabulary: 103 MB / 350 MB.

---

## 6. Algorithm — `generate_bow_vocabulary_main`

```
main:
  EnsureDirectoryExists(output_voc_dir)
  read ImageRetrievalConfig
  KeyframeCacher::Init(keyframe_directory, cache_size, "")
  frame_ids = metadata.GetFrameIds()
  LOG "total frame number: N"
  if input_voc_dir non-empty and exists:  VisualVocabulary::Load(input_voc_dir)     # incremental
  else:                                   VisualVocabulary::Init(branch, depth, radius,
                                                                 descriptor_type, float_centroid)
  voc.BuildVoc(cacher, config)
  voc.build_history += keyframe_directory
  voc.Save(output_voc_dir)
```

### 6.1 `VisualVocabulary::Init` — CONFIRMED

- `descriptor_type = meta.descriptor_type`, substituted to `SUPER_POINT_QUANTIZED(5)` when the
  metadata says `SUPER_POINT(1)` **and** `apple_compress_for_super_point`.
- `feature_dimension_size`: 128 for SIFT/ALIKED, 256 for SuperPoint variants, else `LOG(FATAL)`.
- `apply_merge_before_split = true` (hard-coded).
- `centroid_type = 102 + float_node_centroid` → `CENTROID_FLOAT32` for isaac.
- `root_node_id = 0`.

### 6.2 `VisualVocabulary::BuildVoc` — CONFIRMED

```
segments = ceil(num_frames / max_frame_number_in_memory)        # 1 for both Galileo runs
for s in range(segments):
    if s == 0 and tree_was_empty:  BuildVocByKmeans(cacher, s, config)     # global k-means
    else:                          BuildVocByBirchGC(cacher, s, config)    # incremental BIRCH
total_trained_image_frame_number += num_frames
InitLeafWords()
SetWordWeight(total_trained_image_frame_number, cacher)
PrintWarningWhenNotMature()
```

### 6.3 `BuildVocByKmeans` — the happy path — CONFIRMED

1. Load every keyframe in the segment and collect its `KeypointVector`. If
   `apple_compress_for_super_point && descriptor_type == SUPER_POINT`, quantise first
   (`CompressSuperPointsDescriptor`); otherwise use the descriptors verbatim. **No sampling,
   no capping, no deduplication** — all `frames × keypoints` descriptors go in
   (`226 × 2048 = 462848`, exactly `total_training_feature_number`).
2. `MultiThreadKmeansBuilder(nodes, voc_proto, max_iters, num_threads)`.
3. `BuildVisualVocabulary` collects raw pointers to all descriptors, then `RunCluster`.

### 6.4 `MultiThreadKmeansBuilder::RunCluster` — CONFIRMED / INFERRED mix

- Logs `Total feature number: F`.
- `expected_nodes = (branch^(depth+1) - 1) / (branch - 1)` — for 9/7 that is
  `(9^8-1)/8 = 5380840`, matching the log exactly. The flat node vector is `reserve`d to this
  size up front. CONFIRMED.
- Creates the root node holding all features, then runs a thread pool (`ConsumingThread`) over a
  work queue of `(node, feature indices)` tasks. Each task runs `OneLevelClusterStep`:
  `PerformKMeansCluster` on the node's features, then `AddNodes` to allocate one child per
  non-empty cluster and push the children as new tasks. Node ids come from a free list
  (`GetNewNodeID` / `ReturnNodeID`, log `,free_nodes:`). CONFIRMED (symbols + logs).
- **Stopping rule**: recursion stops at `tree_depth_levels`, or when a node holds ≤ 1 feature.
  INFERRED, but strongly supported by the artifacts: leaf `sample_number` ∈ {1, 2}, exactly 3
  leaves out of 65 533 hold 2 descriptors, and the deepest level (7) has 2514 nodes.
- **Branching**: `k = min(node_branch_number, n_points)`; empty clusters are dropped, so
  observed child counts are 2…9 (32-frame run: 4048 nodes with 9 children, 6608 with 2).
  INFERRED from the artifact histogram.

### 6.5 `PerformKMeansCluster` — CONFIRMED

```
centers = InitClusterKmeansppNoSampling(points)   # k-means++ over the FULL point set,
                                                  # first seed via rand() % n
if centroid_type == FLOAT32: ConvertDescriptorToFloatCentroid(seed) else copy raw bytes
for it in range(weighted_kmeans_max_number_of_iterations):     # 10
    ComputeFeaturesAssignment(points, clusters)                # nearest centroid
    UpdateClusterCenters(assignments)                          # arithmetic mean
    h = AssignmentHash(assignments)
    if h == previous_h: break                                  # converged
```

- Distance: `descriptor_utils::ComputeDescriptorDistance(desc_type, desc, centroid_type,
  centroid)`. For float32 centroids with a float descriptor it dispatches to
  `ComputerFloatDescriptorL2DistanceSSE`; uint8 paths go to
  `ComputerVectorDescriptorL2DistanceSSE`; ORB goes to Hamming. All are monotone in squared
  L2, so plain squared Euclidean is equivalent for assignment. CONFIRMED.
- Centre update is the plain arithmetic mean (`ComputeMeanFloatDescriptor*`), **not**
  re-normalised to unit length. CONFIRMED (internal-node centroid norms < 1 in the artifact).
- Seeding uses libc `rand()` with no seeding call ⇒ deterministic per process but not
  reproducible across implementations. The tree is therefore **not** bit-reproducible; only its
  statistics are.

### 6.6 `InitLeafWords` — CONFIRMED

Walks the flat node array in **index order**; a node is a word iff
`children_ids.empty() && squared_sum >= 0.0`. Assigns `word_id = 0,1,2,…` in that order and
stores the pointer list. Logs `total leaf number: W`. This ordering is exactly what
`word_to_occs` uses. (Verified numerically — reproducing this ordering yields identical word
sets to the shipped index.)

### 6.7 `SetWordWeight(N, cacher)` — CONFIRMED, and it contains a quirk

```
for each training frame:
    reset a bitset over words
    parallel-for each descriptor: word = SearchFeature(descriptor)     # OpenMP
    for each word found: if not seen_in_this_frame: df[word] += 1      # dedup within frame
for each word:
    node.appeared_image_number += df[word]
    node.weight = logf(N / node.appeared_image_number)
    if df[word] == 0 and node.appeared_image_number == 0:
        node.appeared_image_number = 1;  node.weight = logf(N)
```

**The IDF uses integer division.** The weight actually observed on disk is
`logf(float(N // df))`, not `logf(N / df)`. CONFIRMED numerically: on the 32-frame run, words
with `df = 3` carry weight `ln(10) = 2.302585`, not `ln(32/3) = 2.367124`; using `N // df`
reproduces all 32 frames to 1.5e-5 relative. (The `SetWordWeight` decompilation shows a float
divide, so the integer division most likely lives in the *load* path —
`VocabularyNode::VocabularyNode(proto const&, int)` — which is where the index binary gets its
weights, since `generate_bow_index_main` never logs `"set weight for total_word_number:"`.)

Consequence worth flagging for a reimplementation: any word with `df > N/2` gets
`N // df == 1` ⇒ weight `0`, and `TransformImage` drops zero-weight features entirely
(`if (0.0 < weight)`). So the most common half of the vocabulary is silently discarded. On
Galileo this never bites (max `df` = 7 out of 32) because the tree is so over-fitted that
almost every word has `df = 1`.

### 6.8 `PrintWarningWhenNotMature` — CONFIRMED

```
occupied_ratio = num_words / pow(branch_number, tree_depth_levels)
if occupied_ratio < 0.7: LOG(ERROR) "The built vocabulary contains too few visual words, ..."
```

Non-fatal. Galileo: `65533 / 9^7 = 0.0137` and `436358 / 9^7 = 0.0912`. Both runs are far below
0.7 — the shipped vocabularies are *heavily* over-fitted (≈1 descriptor per word). This is
important: the blob's own retrieval is closer to nearest-neighbour descriptor matching through
a tree than to a classic compact BoW.

### 6.9 `Save` — CONFIRMED

Writes `bow_vocabulary.pb.txt` (header, text) then `bow_vocabulary_<i>.pb` shards (binary), as
described in §5.2. Takes ~0.7 s of the 1.3 s total on the 32-frame run.

---

## 7. Algorithm — `generate_bow_index_main`

```
KeyframeCacher::Init(keyframe_directory, /*max_cache=*/2000, "")
read ImageRetrievalConfig
db.Init(output_map_dir + "/vocabulary", config.shared_node_levelsup())
for frame_id in metadata.GetFrameIds():
    kf = cacher.GetKeyFrame(frame_id)      # on failure: LOG(WARNING) "Can't read image_keyframe"
    db.Add(kf)                             # logs "add:i/N" every 200
db.Save(output_map_dir + "/bow_index.pb")
```

### 7.1 `VisualVocabulary::SearchFeature(desc, type, &word_id, &weight, &node_id, levelsup)` — CONFIRMED

```
node = root
level = 0
while node has children:
    level += 1
    node = argmin over children c of ComputeDescriptorDistance(desc, c.centroid)
    if node_id_out != null and level == tree_depth_levels - levelsup:
        *node_id_out = node
word_id_out = node.word_id
weight_out  = node.weight
```

Pure greedy descent, no beam, no backtracking. If the descent never leaves the root it is a
`LOG(FATAL)` ("SearchFeature return root node!"). The `node_id` output is the ancestor at
absolute level `tree_depth_levels - shared_node_levelsup` (= level 5 for isaac); if the leaf is
shallower than that, the leaf itself is recorded.

### 7.2 `VisualVocabulary::TransformImage(kv, &bow, &direct_index, levelsup)` — CONFIRMED

```
for i, descriptor in enumerate(kv):
    if kv.descriptor_type == SUPER_POINT and voc.descriptor_type == SUPER_POINT_QUANTIZED:
        descriptor = QuantizeSuperPointDescriptor(descriptor)
    word, weight, node = SearchFeature(descriptor, ..., levelsup)
    if weight > 0:                       # zero-IDF words are dropped
        bow[word] += weight              # <-- tf * idf accumulation
        direct_index[node].append(i)
bow.Normalize()                          # L1: w /= Σ|w|
```

`WordWeightMap` is a `std::map<int word_id, float weight>`. `Normalize()` divides by
`Σ|w|` and logs `"all the weights are zeros"` if the sum is non-positive.

So the BoW vector is **TF-IDF with raw term counts, L1-normalised**:
`v_w = (tf_w · idf_w) / Σ_u (tf_u · idf_u)`.

### 7.3 `LoopClosureIndex::Init(vocabulary_dir, levelsup)` / `Add` / `Save` — CONFIRMED

- `Init` loads the vocabulary (`VisualVocabulary::Init(dir)` → `LoadVocFiles` → `InitLeafWords`,
  logging `total leaf number:`), and stores `levelsup`. The word weights come from the loaded
  `appeared_image_number` + `total_trained_image_frame_number`.
- `Add(keyframe)` checks `frame_id < 2^32`, calls `TransformImage`, stores
  `FrameInfo{ bow, direct_index }` in a `std::map<uint64, FrameInfo>`, and appends
  `(frame_id, weight)` to each word's occurrence list.
- `Save` writes `shared_node_levelsup` and the dense `word_to_occs`.

### 7.4 Query (not in these binaries, but the contract they serve) — CONFIRMED

`LoopClosureIndex::DetectRawCandidates(query_kv, &raw)`:

```
q = TransformImage(query_kv)                     # L1-normalised TF-IDF
for (word w, weight qw) in q:
    for (frame f, weight dw) in word_to_occs[w]:
        raw[f].score       += |qw - dw| - |qw| - |dw|
        raw[f].shared_words += 1
```

`DetectCandidates(query_kv, min_shared_word_number, query_result_number, &out)`:

```
raw = DetectRawCandidates(query_kv)
out = [c for c in raw if c.shared_words >= min_shared_word_number]     # filter skipped if min <= 0
CalculateCandidatesScore(query_result_number, out):
    sort(out) ascending by raw score                # raw is negative; ascending == best first
    if 0 < query_result_number < len(out): truncate to query_result_number
    for c in out: c.score = -0.5 * c.raw
```

This is **exactly the DBoW2 L1 score**: `s(q,d) = 1 - ½‖q - d‖₁` restricted to shared words,
i.e. `s = -½ Σ_{w ∈ q∩d} (|q_w - d_w| - |q_w| - |d_w|)`, giving `s ∈ [0, 1]`, 1 == identical.
CONFIRMED from `WordWeightMap::operator*` (identical expression, `* -0.5`) and the vectorised
`* -0.5` in `CalculateCandidatesScore`.

`DetectCandidatesForMapper(cacher, query_frame_id, min_shared_word_number,
good_score_threshold, time_gap_seconds, pose_translation_threshold_meters,
pose_rotation_threshold_degrees, query_result_number, &out)` adds, after scoring:

- drop candidates with `score < good_score_threshold`;
- if `pose_translation_threshold_meters > 0`: keep only candidates whose relative pose is
  within `(translation ≤ thr_m, 2·acos(qw)·180/π ≤ thr_deg)` of the query;
- else: keep only candidates whose timestamp differs by at least `time_gap_seconds`;
- stop once about `len/2 + 1` candidates are collected.

CONFIRMED from the decompilation; this is the function `generate_association_main` and
`pose_graph_main` use.

---

## 8. Numerical verification

Script: `/tmp/verify_bow_core.py` (scratch; helper `data/cusfm_re/tools/parse_bow.py` builds
dynamic protobuf classes from the shipped `FileDescriptorSet`).

Reimplementation used: parse the vocabulary shards → flat node array; keep nodes with
`squared_sum != -1`; `word_id` = rank among childless occupied nodes in index order;
`idf[w] = float32(log(float32(N // max(df, 1))))`; per frame, greedy descend all 2048
descriptors with squared-L2 to child centroids; `tf` counts; `v = tf*idf`; L1 normalise.

| Run | Frames | Words | Word-set match | Max relative weight error |
|---|---|---|---|---|
| `data/cusfm_runs/galileo` (32 frames) | 32/32 verified | 65 533 | exact for all frames | **1.5e-5** |
| `data/cusfm_runs/galileo/cusfm` (226 frames) | 4 frames spot-checked | 436 358 | exact | **1.1e-5** |

The residual is float32 rounding in the stored weights. This pins down, beyond doubt: word
ordering, greedy descent, the integer-division IDF, TF accumulation, and L1 normalisation.

Blob runtimes (from `data/cusfm_runs/galileo/runtime.csv`, 226 frames / 462 848 descriptors):
`generate_bow_vocabulary_main` 6.9 s (of which ~1.3 s serialisation), `generate_bow_index_main`
3.5 s. On the 32-frame run: 1.5 s and 1.2 s. Vocabulary on disk: 350 MB / 103 MB.

---

## 9. Fast happy path for the Python reimplementation

### 9.1 What the downstream consumer actually needs

`generate_association_main` / `pose_graph_main` / `feature_matcher_task_builder_main` want, per
query keyframe, **a ranked list of ≤ 20 candidate frame ids** with a similarity score, filtered
by "≥ 50 shared words" and "score ≥ 0.1", then geometrically verified anyway
(`min_matches_num: 30`, fundamental-matrix check, `min_inlier_number`). The retrieval stage is a
**recall device in front of an expensive verifier**. Nothing downstream depends on the word ids,
the tree shape, the IDF values, or the file formats — provided the Python pipeline replaces the
*consumers* too (which the plan of record does).

### 9.2 Must keep

- Deduplicated candidate ranking per query frame, top-N with N ≈ 20.
- Excluding temporal neighbours (the pose-graph stage already has sequential edges; loop
  closures need `loop_interval_threshold_in_seconds: 10` / `loop_closure_interval_ratio: 0.08`
  separation).
- A score in a comparable range so that `good_score_threshold`-style gating still means
  something, or a re-tuned threshold.
- Symmetry/idempotence: the index is built over the same frames that query it, so a frame
  always retrieves itself with score 1 — the consumers must skip self-matches.

### 9.3 Can drop

- The whole `VocabularyNode` slab format, the 2 GB shard split, `bow_vocabulary.pb.txt`, and
  `bow_index.pb` — unless an overlay run must interoperate with unreplaced blob binaries. If
  it must, the writer is simple: the formats above are fully specified and my NumPy reader
  round-trips them.
- `apply_merge_before_split`, BIRCH, `incremental_build_radius_threshold`,
  `max_frame_number_in_memory` segmentation, `--input_voc_dir`: not exercised at Galileo/RoboCap
  scale (single segment for ≤ 20 000 frames).
- The integer-division IDF quirk. Reproduce it only if bit-comparing against the blob.
- The 5.4 M-slot preallocation. Build the tree as a Python list of nodes.
- `apple_compress_for_super_point` (no-op for ALIKED).

### 9.4 Concrete options, cheapest first

**Option A — brute-force cosine on descriptors, no vocabulary (recommended for ≤ ~2000 frames).**
ALIKED descriptors are unit-norm 128-d. Build one `(F, 2048, 128)` array, and for each pair
compute a mutual-nearest-neighbour count or a simple `max`-pooled similarity. For 226 frames
that is `226² / 2 ≈ 25 k` pairs × `2048²` inner products — too slow naively, but a single
`faiss.IndexFlatIP` over all `F·2048` descriptors with `k = 2` and a per-frame vote histogram
does the whole Galileo run in a couple of seconds on CPU and is *strictly more accurate* than
the blob's over-fitted tree (which is already almost a 1-NN index). Sub-option: `torch` on GPU,
`descriptors @ descriptors.T` in chunks.

**Option B — global descriptors (NetVLAD / DINOv2 / SALAD) + cosine.** One vector per image,
`faiss.IndexFlatIP`, top-20. This is what modern hloc-style pipelines do and it is the most
robust choice for large RoboCap sequences. Cost: an extra model download and a forward pass per
keyframe (~5 ms/img on GPU). Score range differs from the L1 BoW score, so re-tune
`good_score_threshold`.

**Option C — faithful vocabulary tree in Python.** `sklearn.cluster.MiniBatchKMeans` (or
`faiss.Kmeans`) applied recursively with `k = 9`, `depth = 7`, stopping at ≤ 1 point per leaf,
then the exact TF-IDF/L1/DBoW2-L1 machinery above. ~80 lines. Use this only if the blob's
retrieval behaviour must be matched. Note the blob's tree is degenerate (≈1 descriptor per
leaf) so it will be *slower and no better* than Option A.

**Option D — pycolmap's vocabulary tree.** `pycolmap` can do `VocabTreeMatching` with a
pretrained tree (SIFT-based, e.g. `vocab_tree_flickr100K_words32K.bin`). It is **not usable
here**: the trees are built for 128-d SIFT with a different quantisation, and cuSFM's
descriptors are ALIKED. Retraining one is Option C with extra steps.

**Recommendation:** Option A for Galileo-scale scenes and RoboCap stride ≥ 4; keep Option B
behind a flag for long sequences. Both are acceptable — the plan of record explicitly targets a
"fast happy path", the downstream verifier is geometric, and the acceptance bound is on
registered images / ATE / reprojection error, not on retrieval recall.

### 9.5 Reference encode/query (matches the blob exactly, for Option C)

```python
import numpy as np

def idf(df: np.ndarray, n_images: int) -> np.ndarray:
    """cuSFM IDF: note the integer division (a quirk, reproduced for parity)."""
    return np.log(np.maximum(n_images // np.maximum(df, 1), 1).astype(np.float32))

def bow_vector(word_ids: np.ndarray, weights: np.ndarray) -> dict[int, float]:
    """TF-IDF with raw counts, L1-normalised; zero-weight words are dropped."""
    keep = weights[word_ids] > 0.0
    ids = word_ids[keep]
    tf = np.bincount(ids, minlength=len(weights))
    v = tf * weights
    return {w: float(x / v.sum()) for w, x in enumerate(v) if x > 0.0}

def l1_score(q: dict[int, float], d: dict[int, float]) -> float:
    """DBoW2 L1 score in [0, 1]; 1.0 == identical."""
    shared = q.keys() & d.keys()
    acc = sum(abs(q[w] - d[w]) - abs(q[w]) - abs(d[w]) for w in shared)
    return -0.5 * acc
```

---

## 10. Evidence index

| Claim | Evidence |
|---|---|
| Source tree layout | `/home/jryu/workspaces/visual_mapping/src/visual/loop_closing/{visual_vocabulary,vocabulary_node,word_weight_map,loop_closure_index,base_voc_builder,birch_voc_builder,multi_thread_kmeans_builder,descriptor_utils}.cc` (embedded paths) |
| Class inventory | `data/cusfm_re/analysis/bow/nm_voc.txt`, `nm_idx.txt` (`nm -C --defined-only`) |
| Strings | `data/cusfm_re/analysis/bow/strings_voc.txt`, `strings_idx.txt` |
| Decompilation | `data/cusfm_re/decomp/bow_idx/` (index binary: `LoopClosureIndex::*`, `WordWeightMap::*`, `VisualVocabulary::{TransformImage,SearchFeature,SetWordWeight,InitLeafWords}`) and `data/cusfm_re/decomp/bow_voc/` (voc binary: `BuildVoc*`, `MultiThreadKmeansBuilder::*`, `BirchVocabularyBuilder::*`, `SaveVocFiles`, `PrintWarningWhenNotMature`, `ComputeDescriptorDistance`) |
| L1 normalisation | `WordWeightMap::Normalize` — `w /= Σ|w|`, `word_weight_map.cc:43` |
| DBoW2 L1 score | `WordWeightMap::operator*` — `Σ(|a-b| - |a| - |b|)`, returns `* -0.5`; same expression inlined in `DetectRawCandidates` + `CalculateCandidatesScore` |
| Word-id ordering | `VisualVocabulary::InitLeafWords`, `visual_vocabulary.cc:34..47` |
| IDF | `VisualVocabulary::SetWordWeight`, `visual_vocabulary.cc:813..899`; integer division established numerically (§8) |
| Tree descent | `VisualVocabulary::SearchFeature`, `visual_vocabulary.cc:116..146` |
| Node/file sizing | `VisualVocabulary::SaveVocFiles` — `NodeSizeInByte` formula, literal `2000000000`, `visual_vocabulary.cc:599..625` |
| Maturity check | `VisualVocabulary::PrintWarningWhenNotMature`, `visual_vocabulary.cc:730` |
| Expected node count | `MultiThreadKmeansBuilder::RunCluster`, `multi_thread_kmeans_builder.cc:63..70` |
| Init constants | disassembly `generate_bow_vocabulary_main` `0x41390` (`VisualVocabulary::Init`) and `0x235d0` (`main`, apple-compress substitution) |
| Index main flow | disassembly `generate_bow_index_main` `0x23550`; CHECK string `db.Init(vocabulary_dir, image_retrieval_config.shared_node_levelsup())` |
| Config values | `pycusfm/configs/isaac/image_retrieval_config.pb.txt:1-15`; downstream overrides `pycusfm/configs/isaac/association_config.pb.txt:8-14`, `pose_graph_config.pb.txt:2-8`, `match_pair_select_config.pb.txt:14-15`, `query_map_config.pb.txt:13-23` |
| Invocation | `pycusfm/cusfm_runner.py:528-556` (`generate_bow`), `:162` (`voc_dir = cuvgl_dir/vocabulary`), `:1007` |
| Artifacts | `data/cusfm_runs/galileo/cuvgl_map/{bow_index.pb,vocabulary/*}` (32 frames), `data/cusfm_runs/galileo/cusfm/cuvgl_map/{bow_index.pb,vocabulary/*}` (226 frames), and the two `generate_bow_*_main.txt` logs |
| Schema | `data/cusfm_re/protos/generate_bow_index_main.fdset` rendered with `data/cusfm_re/tools/fdset_to_proto.py` |
| Paper | arXiv 2510.15271 (HTML): recursive k-means tree, BIRCH CF-tree for incremental, inverted loop index, two-view fundamental check |

---

## 11. Open questions

1. **Where exactly the integer division lives.** The weight is definitely `logf(N // df)` on the
   read path (proven numerically on both runs). `SetWordWeight`'s decompilation shows a *float*
   divide, so the load path (`VocabularyNode::VocabularyNode(proto const&, int)` /
   `LoadVocFiles`) must recompute it with integer division. I did not decompile that
   constructor. Immaterial for the reimplementation; matters only for a bit-exact writer.
2. **`remove_duplicate_feature` and `select_clustering_map_frames`.** No read of the
   corresponding config offsets (`+0x38`, `+0x39`) appears in either binary's BoW paths, and
   both Galileo runs trained on 100 % of frames and 100 % of descriptors. Which binary consumes
   them (`query_map_config.pb.txt` also carries them) is unresolved.
3. **Exact recursion-stop rule** in `OneLevelClusterStep`. I inferred "≤ 1 point or max depth"
   from the leaf statistics rather than reading the 956-line decompilation. The 3 leaves holding
   2 descriptors are consistent with depth-limited termination, but a minimum-cluster-size
   parameter could also explain them.
4. **k-means++ D² sampling detail.** `InitClusterKmeansppNoSampling` seeds with
   `rand() % n` and then picks `k-1` more; I confirmed the entry point and the no-subsampling
   property but did not verify that the remaining seeds use squared-distance weighting rather
   than plain farthest-point.
5. **Empty-cluster policy.** Child counts of 2..8 for nodes with many points imply clusters
   collapse or are dropped, but I did not read `AddNodes`'s exact condition.
6. **Multi-track behaviour.** `cusfm_runner.multi_track_keyframe_for_isaac` builds a per-anchor
   vocabulary under `anchor_cuvgl_map/vocabulary` and aligns other tracks against it. Whether
   the Python pipeline needs the same anchor/track split is a stage-0 question, not a BoW one.
7. **`time_gap_seconds` source.** `DetectCandidatesForMapper`'s 5th parameter is a `double`
   seconds gap that is *not* in `ImageRetrievalConfig`; it most likely comes from
   `association_config.loop_interval_threshold_in_seconds: 10` or
   `pose_graph_config.loop_closure_interval_ratio`. To be resolved in the
   `generate_association_main` / `pose_graph_main` specs.
