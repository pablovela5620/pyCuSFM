# Auditing the cuSfM paper against the shipped binaries, the `colsfm` port, and measurement

The paper is *CuSfM: CUDA-Accelerated Structure-from-Motion*, arXiv:2510.15271v1
(Yu, Liu, Ren, Biswas, Ye, Wu, Majithia, Zeng; NVIDIA). Plain text of the HTML
version is the source for every quotation below.

Three independent things are checked for each claim:

| Column | What it is |
|---|---|
| **Blob** | What `pycusfm/x86_cuda13/bin/*` actually does, per `docs/spec/*.md` (reverse-engineered from the unstripped binaries, the embedded protobuf descriptors, the shipped configs, and A/B re-runs of the real binaries). |
| **colsfm** | What `colsfm/*.py`, the open pycolmap/pyceres reimplementation, does. |
| **Evidence** | Numbers measured on this host, with the file or command that produced them. |

**Hardware and software**: Ubuntu 24.04, RTX 5090, driver 580.173, CUDA 13,
pycolmap 4.2.0 (`cuda_130*` build), pyceres 2.6 / Ceres 2.2.0. The paper's own
bench is an i9-9820X + RTX 6000 Ada, so absolute seconds are not comparable
across the two; ratios between systems measured on *this* host are.

**Datasets**. `data/r2b_galileo` — 226 keyframes, 8 rectified PINHOLE cameras
(1920x1200), 29 rig frames, a 0.66 m sweep, with a `ground_truth.txt` — is the
primary proxy. `data/cusfm_runs/robocap_blobref/input` — 4528 keyframes, 4
OPENCV_FISHEYE cameras, 1132 rig frames, a 124 m walk, **no ground truth** —
carries the long-sequence and loop-closure claims. **KITTI imagery and the
paper's SDG dataset are not available here**, so every number in the paper's
Tables 2-6 is out of reach; that is stated per claim rather than glossed over.

**Scripts written for this audit** (all under `tools/audit/`, run with
`pixi run -e colsfm python -m tools.audit.<name>`):

| Script | Claim | Output |
|---|---|---|
| `galileo_metrics.py` | shared | per-image ground truth, SIM(3) fit, rig track from a rig-free model |
| `trajectory_audit.py` | C12 | `data/bench/audit_galileo_trajectory.json` |
| `extrinsics_audit.py` | C8 | `data/bench/audit_galileo_extrinsics.json` |
| `colmap_baseline_galileo.py` | C10, C11 | `data/bench/audit_colmap_<features>_<matcher>.json`, `data/cusfm_runs/audit_colmap_*/` |
| `configs/radius/` | C3 | a config copy with `search_method: RADIUS_SEARCH`, driven through `demo_rerun.py`; reports in `data/bench/audit_galileo_radius_compare.md` and `audit_galileo_radius_full_compare.md`, runs in `data/cusfm_runs/galileo_audit_radius{,_full}/` |

---

## Summary table

| # | Claim (short) | Blob verdict | colsfm verdict |
|---|---|---|---|
| C1 | GPU ALIKED + LightGlue; matching is the most expensive stage | feature stack **holds**; "matching is most expensive" **does not hold** | feature stack **holds**; "matching is most expensive" **holds** |
| C2 | Environment-specific BoW dictionary + loop detection | **does not hold as shipped** (0 loops with stock configs; gated to death) | **holds with caveats** (retrieval reproduces the blob's pairs; the pose measurement does not) |
| C3 | View-graph beats radius search: ~7x fewer pairs, better accuracy | cost half **holds** (18.7x pairs, 2.05x runtime); accuracy half **does not hold** on Galileo — full radius is *better*, 4.70 vs 5.00 mm | **not applicable** — colsfm implements view-graph only |
| C4 | Arbitrary camera types and quantities | **holds** | **holds** |
| C5 | Stereo relative pose estimation for loop edges | **holds** as described, but is **inert as shipped** | **holds with caveats** — a different estimator, measurably worse |
| C6 | Pose graph optimisation over rig nodes | **holds** | **holds** — parity to 1e-4 m / 1e-3 deg |
| C7 | Iterative triangulation with decaying thresholds + robust loss | **holds** | **holds** |
| C8 | Extrinsic refinement | mechanism **holds**; the accuracy benefit is **not testable here** | **holds** after the fix (commits 7ec2f37, 2f453e1): regularised pass, ATE 4.28 mm vs 4.33 fixed, 4/4 bounds |
| C9 | COLMAP-compatible output, TUM poses | **holds with caveats** (no `rigs.txt`/`frames.txt`; a fake reprojection error) | **holds** |
| C10 | Order-of-magnitude runtime win over COLMAP; 20x mapping | **holds with caveats** — 17.0 % of COLMAP's wall clock, mapping 29x faster, one model vs two to four | **holds** — 8.2 % of COLMAP's wall clock, mapping 72x faster |
| C11 | Better accuracy than COLMAP | **holds** on Galileo | **holds** on Galileo |
| C12 | Refinement improves the initial trajectory | **does not hold** on Galileo | **does not hold** on Galileo |
| C13 | Localization / map-update mode | **holds** (binary present, not exercised) | **does not hold** — no equivalent exists |
| C14 | The KITTI and SDG numbers of Tables 2-6 | **not testable here** | **not testable here** |

---

## C1 — GPU-accelerated ALIKED + LightGlue, and "matching is the most expensive stage"

> "Based on these comprehensive evaluation results, ALIKED is selected as the
> default feature extraction method and LightGlue as the default matching method
> within the cuSfM pipeline." (§3.1)

> "To address the computational overhead of the original Python implementations
> (about 200 ms per image pair) of ALIKED and LightGlue, TensorRT-optimized
> engines based on C++ were employed, reducing inference time to approximately
> 20 ms while maintaining matching quality." (§3.1)

> "Experimental results reveal that feature matching constitutes the most
> time-consuming stage in cuSfM." (§6.2)

**Blob.** The feature stack is exactly as described.
`feature_extractor_main --feature_type` accepts `aliked | superpoint | sift`
(`docs/spec/feature_extractor_main.md` §2), the default is `aliked`, and the
graph is run through a TensorRT FP16 engine built from
`pycusfm/models/aliked_lightglue/aliked.onnx` with a content-addressed engine
cache (§1, §6.2). CV-CUDA SIFT is real too — `cvcudaSIFTCreate`,
`cvcudaPairwiseMatcherCreate` are linked in (§6). Matching is LightGlue through
TensorRT with a `match_threshold: 0.3` score gate, then SSC spatial NMS to
`match_top_k: 500`, then `cv::findEssentialMat` RANSAC on bearing vectors
(`docs/spec/feature_matcher_main.md` §5). The measured per-pair matching cost on
Galileo is **4.57 ms**, an order below the paper's ~20 ms figure, because the
paper's number is per pair on KITTI-sized workloads with more keypoints
(`data/cusfm_runs/galileo_blobref/cusfm/matches/feature_matcher_main_matching_task_000.pb.txt`:
"Number of frame pair: 336, Num Matches per frame pair: 399 ... Time used per
frame pair: 4.57052ms").

**The "most time-consuming stage" claim does not survive contact with the
binaries on either dataset here.** From `data/bench/galileo_compare.md` and
`data/bench/robocap_compare.md`, column A (the blob):

| Stage | Galileo (s) | share | RoboCap (s) | share |
|---|---|---|---|---|
| 1 feature extraction + keyframe selection | 8.64 | 21.6 % | 183.91 | 16.2 % |
| 2 BoW vocabulary + index | 11.73 | **29.3 %** | 243.39 | 21.5 % |
| 3 loop-closure association | 3.17 | 7.9 % | 75.66 | 6.7 % |
| 4 pose graph optimisation | 3.58 | 8.9 % | 386.61 | **34.1 %** |
| 5 match pair selection | 1.94 | 4.8 % | 5.62 | 0.5 % |
| **6 feature matching** | **2.25** | **5.6 %** | **52.53** | **4.6 %** |
| 7 triangulation + bundle adjustment | 6.69 | 16.7 % | 178.75 | 15.8 % |
| 8 COLMAP + TUM export | 2.07 | 5.2 % | 7.07 | 0.6 % |
| total | 40.07 | | 1133.53 | |

Matching is the *second cheapest* stage of eight on both datasets. The single
largest is BoW dictionary construction on Galileo and pose graph optimisation on
RoboCap — and the latter is dominated by loop *search*, not by Ceres:
`docs/spec/pose_graph_main.md` §9.2 measures 307 s of loop search against 6.5 ms
in Ceres for a 1132-node graph.

This matters beyond bookkeeping. **Table 2 of the paper itemises only three
stages — Feature Extraction, Feature Matching, Mapping — and its "Total" is
exactly their sum**: 4.996 + 37.370 + 16.458 = 58.824 for cuSfM view-graph,
4.996 + 76.600 + 26.397 = 107.993 for cuSfM radius, 1.660 + 3.262 + 97.440 =
102.362 for GLOMAP, and 2.004 + 3.690 + 340.467 = 346.161 against a printed
346.162 for COLMAP. §6.2 itself acknowledges that
"the view-graph strategy requires additional computational overhead for Bag of
Words dictionary construction and search tree building", yet none of that time
appears in the table or in the total. On the two datasets measured here those
missing stages are **stages 2 + 3 + 4 = 46.1 % of Galileo's wall clock and
62.3 % of RoboCap's**. The reported 16.9 % of COLMAP's runtime is therefore an
undercount of cuSfM's own cost by construction.

**colsfm.** `colsfm/features.py` drives `pycolmap`'s own
`FeatureExtractorType.ALIKED_N16ROT`; `colsfm/matching.py` drives
`FeatureMatcherType.ALIKED_LIGHTGLUE`, both ONNX graphs downloaded by COLMAP
into `~/.cache/colmap/`. The variant is matched to the blob by reading the
shipped ONNX: `aliked.onnx` carries a 32-channel SDDH offset convolution, so
`M = 16`, the ALIKED-n16 architecture (`colsfm/features.py` docstring). The
weights are rotation-augmented, so descriptors are not the blob's bit for bit —
that was never the goal. SuperPoint and CV-CUDA SIFT have no colsfm equivalent.

For colsfm the paper's stage-cost claim **does** hold: matching is 8.15 s of
19.35 s (42 %) on Galileo and 152.03 s of 421.79 s (36 %) on RoboCap — the
largest single stage in both, because colsfm skips stages 2-4 by default.

**Verdict — Blob:** feature stack **holds**; "matching is the most expensive
stage" **does not hold**, and Table 2's accounting omits 46-62 % of the
pipeline's measured wall clock.
**Verdict — colsfm:** feature stack **holds** (ALIKED + LightGlue only, no
SuperPoint or SIFT); "matching is the most expensive stage" **holds**.

---

## C2 — Environment-specific bag-of-words dictionary and loop detection

> "Rather than utilizing generic shared dictionaries, the system constructs
> environment-specific vocabularies that capture unique visual characteristics of
> the input datasets, thereby enhancing discriminative power." (§3.2)

> "The number of inliers resulting from this validation step serves as a key
> indicator of the reliability and success of loop frame detection, with a higher
> number of inliers generally signifying a more confident and accurate match."
> (§3.3)

**Blob — the machinery is all there.** `generate_bow_vocabulary_main` builds a
per-dataset hierarchical vocabulary with `node_branch_number: 9` and
`tree_depth_levels: 7`, by multi-threaded weighted k-means
(`weighted_kmeans_number_of_thread: 12`), with the BIRCH incremental path
reachable through `--input_voc_dir` and `incremental_build_radius_threshold`
(`docs/spec/bow.md` §3, §5). `generate_bow_index_main` writes the inverted index
with L1-normalised TF-IDF weights (§5.3). Retrieval is top-20 candidates, at
least 50 shared words, DBoW2 L1 score above `good_score_threshold: 0.1` (§3).
Every element of §3.2 and §3.3 exists in the binary. Nothing is generic; the
vocabulary is built from the input sequence.

**Blob — but as shipped it detects nothing.** Two independent gates, both at
protobuf-default or near-default values, close the loop-closure path:

1. `pose_graph_main`'s `loop_edge_translation_threshold_meters` and
   `loop_edge_rotation_threshold_degrees` are **unset**, i.e. proto3 default
   `0.0`, and the acceptance test rejects any candidate whose relative-pose
   magnitude *exceeds* the threshold. With the shipped `isaac` and `av` configs
   the pose graph therefore **never produces a loop edge**; the graph is a pure
   odometry chain with one fixed anchor and the optimum reproduces the input
   trajectory exactly (`docs/spec/pose_graph_main.md`, Headline finding, **[C]**,
   confirmed by re-running the blob).
2. `generate_association_main`'s `max_mean_point_to_epipolarline_error: 10e-6`
   is expressed on the normalised plane and rejects 168 of 180 RoboCap
   candidates that reach it; 12 more die on "Scale too large after pose
   refinement", leaving **0** loop associations
   (`docs/spec/generate_association_main.md` §6.4, three independent runs).

Also, `skip_data_association` defaults to `True` in
`get_default_cusfm_params`, so `generate_association_main` — the paper's Figure 2
"Loop Closure Detection" box — **does not run at all** in the default pipeline
(NOTES.md, "Running the full pipeline"). With it off the pose graph falls back to
`image_retrieval_config.pb.txt`, whose `query_result_number: 1` returns one BoW
candidate per query, nearly always the temporally adjacent frame.

**Measured, with the repaired config** (`data/cusfm_configs/loop-closure-fixed`,
which sets non-zero loop-edge thresholds and enables the association stage):

| | Galileo | RoboCap |
|---|---|---|
| `generate_association_main` LOOP associations | **0** (`data/bench/galileo_blob.log`: "0 loop associations are retrieved") | **7** (`data/bench/robocap_blob.log`) |
| `pose_graph_main` loop constraints in the final solve | **5** | **90** |

So even with the repair the *association* binary — the one implementing §3.3's
four-stage retrieve/match/verify pipeline — finds essentially nothing, and the
loop edges that reach Ceres come from `pose_graph_main`'s own, more lenient,
internal retrieval. With the stock `isaac` config the count is 0 everywhere.

**colsfm.** `colsfm/retrieval.py` keeps the ranked candidate list and drops the
vocabulary artifacts, the word ids and the file formats, because nothing
downstream reads them. It has two backends chosen by sequence length: brute-force
mutual-nearest-neighbour voting at or below 500 images, and a hierarchical
k-means vocabulary with TF-IDF and the blob's own DBoW2 L1 score above it
(NOTES.md colsfm decision 9). The vocab backend builds in 42 s on RoboCap,
answers all 4528 queries in 1.5 s, and **recovers every one of the blob's 90 loop
pairs**. `colsfm/loop_closure.py` then applies real gates rather than the blob's
`10e-6` (deviation 7: `ransac_max_error_px = 4.0`, COLMAP's own default) and
produces 74 loop edges on RoboCap. Retrieval is therefore not the weak link on
this side; the *measurement* is (see C5). Loop closure is off by default
(`LoopClosureConfig.enabled = False`) because on Galileo turning it on moves
camera positions 5-13 mm against a 5 mm budget — a short low-drift sweep has
nothing for a loop to fix.

**Verdict — Blob:** the described dictionary and detector exist and are
faithful to §3.2/§3.3, but **the claim does not hold for the shipped
configuration**: loop detection is disabled by default and gated to zero output
even when enabled.
**Verdict — colsfm:** **holds with caveats** — retrieval reproduces the blob's
loop pairs and the gates are sane, but the pose measurement behind each edge is a
different algorithm and measurably worse (C5).

---

## C3 — Non-redundant view-graph construction from pose-graph priors

> "Experimental results demonstrate that constructing the view graph exclusively
> from pose graph edges achieves successful SfM reconstruction while
> significantly reducing computational overhead through non-redundant data
> association." (§3.4)

> "Comparative analysis reveals that the radius search method identifies
> approximately seven times more image pair matches than the view-graph based
> approach, which inevitably leads to increased feature matching time." (§6.2)

**Blob.** `feature_matcher_task_builder_main` dispatches on
`match_pair_select_config.pb.txt`'s `search_method`:
`RADIUS_SEARCH = 0`, `IMAGE_RETRIEVAL = 1`, `TRACK_2D_OVERLAP_IR = 2`,
`POSE_GRAPH_ASSOCIATION = 3`. The shipped `isaac` and `av` configs both use
`POSE_GRAPH_ASSOCIATION`, in which the task builder is a **pure filter over
pose-graph edges** — no spatial search, no image retrieval, and no downsampling
at all: every downsampling knob and the `--downsampling_matches` flag are dead in
this mode, verified by a 13-point config sweep that produced exactly 339 pairs
for every variant (`docs/spec/feature_matcher_task_builder_main.md`, Headline
finding, **[C]**). The three edge kinds are exactly the paper's three:
`CONSECUTIVE`, `LOOP`, `EXTRINSIC`
(`docs/spec/generate_association_main.md` §1).

**The pair-count ratio holds and is if anything understated.** From the archived
radius sweep on Galileo (`docs/spec/feature_matcher_task_builder_main.md` §4.4,
`data/cusfm_re/analysis/sweep_radius.sh`, **[C]**):

| Mode | Pairs on Galileo |
|---|---|
| `POSE_GRAPH_ASSOCIATION` (shipped) | **339** |
| `RADIUS_SEARCH`, downsampling on | 768 |
| `RADIUS_SEARCH`, `--downsampling_matches=false` | **6273** |
| all pairs within 3 m of each other | 25425 |

6273 / 768 ≈ 8.2 and 6273 / 339 ≈ 18.5; the paper's "approximately seven times"
sits between the two depending on whether the radius mode's own downsampling is
left on. The direction of the claim is right on this dataset.

**Measured end to end.** The pair counts above come from an archived sweep of the
task builder alone. To test the *accuracy* half of the claim the whole blob
pipeline was re-run in radius mode, changing exactly one line —
`search_method: POSE_GRAPH_ASSOCIATION` becomes `RADIUS_SEARCH` in
`tools/audit/configs/radius/match_pair_select_config.pb.txt`, everything else
copied verbatim from `data/cusfm_configs/loop-closure-fixed`:

```
pixi run -- python demo_rerun.py --rr-config.headless \
    --run.variant _audit_radius --run.config-dir tools/audit/configs/radius \
    dataset:galileo
pixi run -e colsfm python -m colsfm.bench_cli --dataset galileo \
    --run-a data/cusfm_runs/galileo_blobref/cusfm \
    --run-b data/cusfm_runs/galileo_audit_radius/cusfm \
    --input-dir data/r2b_galileo --name-a view-graph --name-b radius \
    --report data/bench/audit_galileo_radius_compare.md
```

Result (`data/bench/audit_galileo_radius_compare.md`), both runs the same
binaries on the same host:

| | view-graph | radius | radius / view-graph |
|---|---|---|---|
| image pairs matched | 336 | **772** | 2.30x |
| registered / total images | 224 / 226 | **216** / 226 | — |
| 3D points | 5065 | 7497 | 1.48x |
| observations | 37 820 | 90 131 | 2.38x |
| mean reprojection error (px) | **1.550** | 1.885 | 1.22x |
| track length mean / median | 7.47 / 5.00 | 12.02 / 7.00 | — |
| **ATE vs ground truth (mm RMSE)** | **5.00** | **4.98** | **1.00x** |
| ATE vs ground truth (mm max) | 8.31 | 8.61 | 1.04x |
| feature matching (s) | 2.25 | 4.03 | 1.79x |
| triangulation + BA (s) | 6.69 | 13.41 | 2.00x |
| **total runtime (s)** | **40.07** | **49.13** | **1.23x** |

The two trajectories land 0.54 mm apart (RMSE over the 29 rig frames).

**This half-confirms and half-refutes the paper.** Confirmed: radius search is
strictly more expensive — 2.3x the pairs, 1.8x the matching time, 2.0x the
mapping time, 1.23x end to end — and it buys nothing, so the view-graph is the
right default. Refuted: the paper's §6.3.2 comparison (Table 4) reports
cuSfM-Radius as *less accurate* than cuSfM-ViewGraph on every KITTI sequence,
and §3.4 argues redundant pairs "generate potentially conflicting matches". On
Galileo they are **the same to 0.02 mm** — 4.98 against 5.00 mm, a difference far
inside any reasonable noise band. What the extra pairs do change is the
reconstruction, and for the worse: 8 fewer registered images and a 22 % higher
reprojection error, on 2.4x the observations.

The honest reading is that on this dataset the non-redundancy argument is a
**cost** argument that holds, not an **accuracy** argument. Galileo is a 0.66 m
sweep with 29 rig instants, though, and the paper's claim is made on KITTI
sequences with hundreds of metres of drift, where a radius search reaches across
loop closures the view graph would not have. That regime is not testable here.

**And with the radius mode's own downsampling turned off**, which is the
configuration the paper describes (§6.2: "selects candidate frames within a
20-meter radius and 90-degree angular deviation", with no mention of a per-window
keyframe cap), the picture changes again. Re-running the same command with
`--run.variant _audit_radius_full --run.no-downsampling-matches` produces exactly
the 6273 pairs the archived sweep predicted (`data/bench/audit_galileo_radius_full_compare.md`):

| | view-graph | radius, downsampled | **radius, full** |
|---|---|---|---|
| image pairs matched | 336 | 772 | **6273** (18.7x) |
| registered / total images | 224 / 226 | 216 / 226 | **224** / 226 |
| 3D points | 5065 | 7497 | 5667 |
| observations | 37 820 | 90 131 | **100 579** |
| mean reprojection error (px) | 1.550 | 1.885 | 2.033 |
| track length mean / median | 7.47 / 5.00 | 12.02 / 7.00 | **17.75 / 11.00** |
| **ATE vs ground truth (mm RMSE)** | 5.00 | 4.98 | **4.70** |
| feature matching (s) | 2.25 | 4.03 | **25.54** (11.4x) |
| triangulation + BA (s) | 6.69 | 13.41 | 24.03 (3.6x) |
| **total runtime (s)** | **40.07** | 49.13 | **82.33** (2.05x) |

**The full radius mode is the most accurate of the three** — 4.70 mm against the
view graph's 5.00 mm, a 6 % improvement — for 2.05x the wall clock and 11.4x the
matching time. It registers the same 224 images and builds tracks 2.4x longer.
The higher reprojection error is what longer tracks over more observations
produce, not a failure: the same points are seen from more views and the residual
is spread over 2.7x the observations.

That is the opposite of what §6.3.2 reports. On KITTI the paper has
cuSfM-ViewGraph beating cuSfM-Radius on RMSE in every sequence of Table 4; on
Galileo the ordering reverses. The cost claim survives intact — nobody would
argue for spending 2x the time to gain 0.3 mm — but the paper's framing, that
redundant pairs actively *hurt* accuracy by generating "potentially conflicting
matches" (§3.4), is not what this dataset shows. Here redundancy buys a small,
real accuracy gain at a large, real cost.

**colsfm.** `colsfm/pairs.py` implements the view-graph rule **only**; there is
no radius or spatial mode and no `search_method` knob, so the paper's comparison
cannot be reproduced on this side. The rule reproduces 331 of the blob's 339
Galileo task pairs (218 consecutive + 113 stereo) with no extras, the remaining
eight being LOOP edges, and all 40 pairs exactly on the 32-keyframe run
(`colsfm/pairs.py` docstring). Note the structural consequence the module states
plainly: "the stage cannot invent cross-camera pairs beyond the stereo baseline;
all other cross-camera connectivity comes from loop closure."

**Verdict — Blob:** the **cost half holds** and is understated — full radius
search matches 18.7x as many pairs, spends 11.4x the matching time and 2.05x the
total wall clock. The **accuracy half does not hold on this dataset**: full
radius search is *better*, 4.70 mm against 5.00 mm, and the downsampled variant
is a tie at 4.98 mm. The view graph is the right default here because it is
cheap, not because it is more accurate. Whether the accuracy claim holds on
KITTI-scale drift is **not testable here**.
**Verdict — colsfm:** **not applicable** — view-graph is the only mode
implemented, which is a deliberate scope decision, not a reproduction of the
comparison.

---

## C4 — Arbitrary camera type and quantity support

> "The current implementation of cuSfM provides built-in support for two
> representative camera models: standard cameras with undistortion capabilities
> and F-theta cameras incorporating windshield models." (§3.5)

> "The camera projection modular design ensures that feature matching and
> triangulation-based mapping components operate independently of the underlying
> projection model, treating it as an abstract interface." (§3.5)

**Blob.** Four `CameraProjectionModelType` values are handled end to end, and
the COLMAP exporter's mapping is byte-verified against the shipped models
(`docs/spec/export.md` §3.3):

| cuSfM model | COLMAP model | Source | PARAMS |
|---|---|---|---|
| `PINHOLE` (0) | `PINHOLE` | `projection_matrix` 3x4 | `fx fy cx cy` |
| `DISTORTED_PINHOLE` (1) | `FULL_OPENCV` | `camera_matrix` 3x3 + distortion | `fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6` |
| `FTHETA_WINDSHIELD` (2) | `PINHOLE` | `projection_matrix` | `fx fy cx cy`, **lossy**, with a warning |
| `OPENCV_FISHEYE` (3) | `OPENCV_FISHEYE` | `camera_matrix` 3x3 + distortion | `fx fy cx cy k1 k2 k3 k4` |

Quantity is unrestricted: Galileo runs 8 cameras in one rig, RoboCap 4.

One correction to the paper's own phrasing: `docs/camera.md`'s summary table
calls `FTHETA_WINDSHIELD` unsupported, but the binary **emits a PINHOLE line and
only warns** ("the output in rectified model may be inaccurate"). The table is
misleading; the export is lossy rather than refused (`docs/spec/export.md` §3.3).

**colsfm.** `colsfm/cameras.py` implements the same four-model table, including
the lossy `FTHETA_WINDSHIELD` -> `PINHOLE` fallback and its warning
(`colsfm/cameras.py` lines 45-48, 135-157). Anything else raises
`ValueError: Unsupported camera projection model type`. The module also records
a real interoperability hazard the paper does not mention: cuSfM copies both
keypoints and `cx, cy` in the OpenCV convention (top-left pixel centre at
`(0, 0)`) while COLMAP places it at `(0.5, 0.5)`, so the exported model is
self-consistent but sits half a pixel off a COLMAP-native one.

**Evidence.** Both systems reconstruct Galileo's 8 PINHOLE cameras and
RoboCap's 4 OPENCV_FISHEYE cameras: `data/bench/galileo_compare.md`
(224/226 and 225/226 registered) and `data/bench/robocap_compare.md`
(4526/4528 and 4528/4528). `DISTORTED_PINHOLE` and `FTHETA_WINDSHIELD` are
exercised by neither dataset here, so their support is read from code and specs,
not measured.

**Verdict — Blob:** **holds**.
**Verdict — colsfm:** **holds** — the same four models, the same mapping.

---

## C5 — Stereo relative pose estimation for loop edges

> "This work presents a novel algorithm that directly estimates the translation
> scale using only 2D observations, thereby circumventing the need for both
> landmark triangulation and the SolvePnP procedure." (§4.1)

**Blob.** `StereoPoseEstimator::EstimateRelativePose`
(`docs/spec/generate_association_main.md` §5.4, §6.5) is exactly the three-step
algorithm of §4.1, and the description is accurate:

1. Two-view essential matrices for left/map and map/right, giving rotation and
   translation *direction* only.
2. Scale from the three-view relation with the calibrated stereo baseline —
   `EstimateScaleByLinearFitting(t_lc_rot, t_cr, baseline * dir, &scale, &err)`,
   with `use_estimated_baseline_direction: false` hard-setting the left-to-right
   direction to `(-1, 0, 0)`. No triangulation, no PnP: the claim is literally
   true of the code.
3. `RelativePoseRefine` -> `StereoPoseRefineSolver`, minimising the Sampson
   distance over both pairs jointly.

Two things the paper does not mention. First, when the fitted scale falls
outside `[0, scale_upper_limit = 4]` and `early_return_invalid_scale` is false —
the shipped default — the scale is **silently replaced by the magic constant
0.47** rather than the candidate being rejected (the literal
`0x3fde147ae147ae14` is stored in the `else` branch, **[C]**). Second, the
`max_mean_point_to_epipolarline_error: 10e-6` acceptance gate makes the whole
estimator inert on real data (C2): 168 of 180 RoboCap candidates fail it with
measured errors two orders of magnitude above the threshold.

**colsfm.** `colsfm/loop_pose.py` deliberately does **not** port this. It builds
a local metric point cloud in the source rig frame — triangulated from that rig's
own cameras plus its temporal neighbours — and then resections the whole target
rig with `pycolmap.estimate_and_refine_generalized_absolute_pose`. That is a
triangulate-then-generalized-PnP design, i.e. precisely the approach §4.1 argues
against, chosen because the blob's baseline-locked two-view scale measured worse
here.

**Evidence** (`colsfm/loop_closure.py` docstring, NOTES.md colsfm deviation 6,
all on RoboCap against the blob's own 90 LOOP pairs):

| Loop-edge estimator | Pose graph result vs the blob's PGO |
|---|---|
| no loop edges (input trajectory alone) | 460.8 mm |
| the blob's own edges through `solve_pose_graph` | **135.0 mm** |
| oracle poses on colsfm's own pairs | 39.9 mm |
| two-view + prior scale (the first colsfm attempt) | 609.0 mm — *worse than no loops* |
| local metric map + generalized resection (current) | **408.1 mm** on that pair set, 404.4 mm end to end |

The residual gap is rotational: colsfm's loop rotations sit 6.5 deg from the
blob's and 1.5 deg from the odometry's, while
`pycolmap.estimate_generalized_relative_pose` — an independent estimator sharing
only the correspondences — agrees with colsfm's to 0.35 deg. Substituting the
blob's rotations into colsfm's edges takes the pose graph from 392 mm to 266 mm;
substituting its translations changes nothing. RoboCap ships no ground truth, so
which of the two rotations is right cannot be settled here.

**Verdict — Blob:** the algorithm **holds** as described — it is genuinely
scale-recovering from 2-D observations only — but it is **inert as shipped**
because of the epipolar gate, and it carries an undocumented `0.47` fallback.
**Verdict — colsfm:** **holds with caveats** — the *function* is provided by a
different, explicitly documented algorithm that lands 408 mm from the blob
against 135 mm for the blob's own edges.

---

## C6 — Pose graph optimisation

> "Experimental results demonstrate that the vehicle rig configuration often
> achieves superior pose estimation accuracy due to its reduced parameter space
> and implicit handling of inter-camera constraints." (§4.2)

**Blob.** `pose_graph_main` matches §4.2 in structure
(`docs/spec/pose_graph_main.md` §6, §7):

* Nodes are **rig (vehicle) frames**, one `world_T_vehicle` per
  `synced_sample_id`. The camera-frame configuration also exists; the shipped
  `isaac` profile uses the rig one, which is what the paper recommends.
* Three edge kinds — sequential from the input odometry, loop, extrinsic — as
  Eqs. 5-7 describe. With rig nodes the extrinsic constraints are absorbed into
  the node definition, exactly as §4.2 says.
* Residual: `OptimizePoseRelativeConstraintCost`, a
  `ceres::AutoDiffCostFunction<..., 6, 4, 3, 4, 3>` on `[log(R_err); t_err]`,
  whitened by `chol(information)`. `pose_covariance` in the protos is consumed as
  an **information** matrix despite its name.
* Gauge: `SetPositionConstraintFrame(anchor_id)` calls
  `SetParameterBlockConstant` on the anchor's quaternion *and* translation — one
  fully fixed node, no soft prior. The anchor is the source of the first edge.
* Solver: Ceres, `SPARSE_SCHUR` by default, `SPARSE_NORMAL_CHOLESKY` +
  `CUDA_SPARSE` on the cuDSS path, 200 iterations.

One measured wrinkle the paper does not mention: **the loop-edge weights the
binary writes are not the weights it solves with.** `pose_graph_main` writes
`Λ = I₆ · (score · loop_residual_weight)` into `vehicle_pose_graph.pb.txt`, but
overwrites every stored 6x6 with the default information matrix unless
`relative_constraint_use_pose_covariance` is set — and no shipped config sets it
(`tests/colsfm/test_pose_graph.py::test_galileo_loop_solve_reproduces_the_blob_rig_poses`).

**colsfm.** `colsfm/pose_graph.py` implements the same problem on
`pyceres.Problem` with a Python residual, because pycolmap has no pose-graph
optimiser: `pycolmap.PoseGraph` is a container with no `optimize`,
`PoseGraphEdge` carries no information matrix, and every
`pycolmap.cost_functions` factory raises `TypeError: Unregistered type` against
the standalone `pyceres` (NOTES.md colsfm decision 4).

**Evidence — parity, from `tests/colsfm/test_pose_graph.py`:**

| Case | Agreement with the blob's `vehicle_frames_meta.json` |
|---|---|
| Galileo, 29 rig nodes, 28 sequential edges, no loops | **< 1e-4 m and < 1e-3 deg** |
| Galileo, 29 rig nodes + the blob's 8 archived loop edges, identity-weighted | **< 1e-3 m and < 0.05 deg** (the loops move the trajectory 12.3 mm) |
| the same with the *stored* score weights | > the identity-weighted error — the evidence that the stored weights are not what the blob solved with |
| 1000 nodes + 50 loops | solves in < 1.0 s (the blob does 1132 nodes in 6.5 ms of Ceres time) |

The one place colsfm is slower is the residual, which is Python: NOTES.md colsfm
decision 4 records 24.6 s for an early 1000-node solve, of which the sparse
Cholesky was 40 ms. The current implementation does the same 1000-node,
50-loop graph in **0.80 s** (measured just now,
`pytest tests/colsfm/test_pose_graph.py -k thousand --durations=3`), against the
blob's 6.5 ms of Ceres time. In pipeline terms it never binds: PGO is 0.01 s on
Galileo and 0.34 s on RoboCap against the blob's 3.58 s and 386.61 s — because
the blob's PGO stage is loop *search*, not solving.

**Verdict — Blob:** **holds**.
**Verdict — colsfm:** **holds** — numerical parity with the blob to 1e-4 m /
1e-3 deg without loops and 1e-3 m / 0.05 deg with them.

---

## C7 — Iterative triangulation and mapping with decaying thresholds and a robust loss

> "The first stage adopts relaxed outlier thresholds and robust loss functions to
> ensure rapid convergence. ... The second stage applies stringent outlier
> criteria without loss functions to achieve high-precision results." (§4.3)

**Blob.** `keypoints_mapper_main` runs the described alternation, and every
number is confirmed against the shipped `--v=3` log
(`docs/spec/keypoints_mapper_main.md` §3, §5.6):

* `num_ba_iterations: 5` outer triangulate / merge / complete / filter / BA
  rounds, logged as "Bundle Adjustment Iterative number: 5".
* A **linear** pixel gate from `initial_max_pixel_error: 25.0` to
  `final_max_pixel_error: 5.0`: a `--v=3` re-run prints
  `max_pixel_error = 25, 20, 15, 10, 5` across the five rounds.
* `loss_type: CAUCHY` with `reprojection_error_standard_deviation: 4.0` px
  whitening the residual, and `refine_ba_avoid_loss_function: true` for the final
  refine stage — the paper's two-stage story exactly.
* Tracks are grown by BFS over the match graph with `search_depth: 15`; DLT
  triangulation plus a midpoint method, as §4.3 says.
* `max_observation_change: 0.0005` is an early-exit on the outer loop; in the
  shipped run it never fired and all five rounds ran. Points went
  1435 -> 1336 -> 1335 -> ... -> 1275 over the five rounds.

One inert knob: `min_triangulation_deg: 2.0` filtered **0 points in almost every
round**, because it is applied as the *maximum* over all observing pairs
(`HasLargeEnoughTriangulateAngle`) rather than a minimum requirement.

**colsfm.** `colsfm/mapping.py` reproduces the schedule on pycolmap's
`IncrementalTriangulator`, `ObservationManager` and
`create_default_bundle_adjuster`, with the gate `g_k` also driving
`merge_max_reproj_error` and `complete_max_reproj_error`. The Cauchy whitening is
handled by setting `ceres.loss_function_scale = sigma * loss_function_scale = 4.0`,
which is algebraically identical: `CauchyLoss(a) = a² log(1 + s/a²)` evaluated at
`s/σ²` with `a = 1` equals evaluating it at `s` with `a = σ`, up to a constant
that cannot move the minimum (NOTES.md colsfm decision 6). Two blob behaviours
are deliberately not reproduced, both documented as defects: the unweighted
gauge-keyframe residuals (13 of 6276 blocks on Galileo) and the 3-D depth
residual, which Galileo does not use.

**Evidence — Galileo, `data/bench/galileo_compare.md`:**

| Metric | blob | colsfm |
|---|---|---|
| registered / total images | 224 / 226 | 225 / 226 |
| 3D points | 5065 | 6195 |
| observations | 37 820 | 30 910 |
| mean reprojection error (px, recomputed) | 1.550 | 1.334 |
| track length mean / median | 7.47 / 5.00 | 4.99 / 4.00 |
| stage 7 runtime | 6.69 s | 2.70 s |

colsfm ends with more points at a lower reprojection error and shorter tracks —
a structural difference, not a tuning one: cuSfM grows one disjoint track per
connected component and triangulates with plain MSAC, rejecting roughly half its
candidate tracks, while COLMAP's LO-RANSAC keeps them. 98-99 % of the blob's
points have a colsfm point within 5 cm (NOTES.md colsfm deviation 1).

**Verdict — Blob:** **holds** — the schedule, the loss, the sigma and the
two-stage design are all in the binary as described.
**Verdict — colsfm:** **holds**.

---

## C8 — Extrinsic refinement

> "This extension ensures that camera-to-vehicle transformations remain within
> reasonable bounds of their initial calibration values, which are typically
> obtained through manual measurement or prior calibration procedures." (§4.4)

> "The experimental results demonstrate significant improvements across all
> evaluated sequences when extrinsic refinement is enabled." (§6.4.1)

**Blob.** `--optimize_extrinsics` requires `ba_frame_type=vehicle_rig` and is
**off by default** (NOTES.md, "Running the full pipeline"). When on, the rig
extrinsics enter the bundle adjustment with the absolute-pose prior of Eq. 14,
whose sigmas are `extrinsic_error_meters: 0.01` and `extrinsic_error_degrees: 2`
in the `isaac` profile (`docs/spec/keypoints_mapper_main.md` §3.2).

**Evidence — it does move, and by about the size of the prior.**
`data/cusfm_runs/galileo/cusfm` was written with `--optimize_extrinsics=True`.
Measured by `pixi run -e colsfm python -m tools.audit.extrinsics_audit`
(`data/bench/audit_galileo_extrinsics.json`):

| Quantity | Max over Galileo's 8 cameras / 28 pairs |
|---|---|
| absolute `vehicle_T_cam` movement | 3.99 mm / 1.23 deg |
| **relative `cam_i_T_cam_j` movement (gauge-free)** | **10.91 mm / 1.40 deg** |

The relative figure is the meaningful one: it cannot be explained by the bundle
adjuster rotating the whole rig frame. The largest movers are the front-to-back
pairs (`back_stereo_camera_left -> front_stereo_camera_left`, 10.91 mm), which
share the least field of view; the stereo pairs, which see almost the same scene,
move least (`right_stereo_camera_left -> right_stereo_camera_right`, 0.20 mm).
That pattern is what a weakly-observed extrinsic looks like, and 10.9 mm against
a 10 mm prior sigma means the data moved the calibration by about one sigma.

**Whether that movement is an accuracy *improvement* is not testable here.**
The paper's Table 6 is KITTI-only, and Galileo has no independent extrinsic
ground truth to score the refined calibration against.

**colsfm — the flag is a silent no-op.** `MappingOptions.optimize_extrinsics`
maps to `refine_sensor_from_rig`, but COLMAP's `ParameterizeRigsAndFrames`
(`estimators/bundle_adjustment_ceres.cc`) sets the rig poses constant when the
reference sensor is not part of the problem, because "otherwise, the relative
pose between the sensors is not well constrained". colsfm registers the vehicle
body as `sensor_t(SensorType.IMU, 0)`, a reference sensor that owns no images
(NOTES.md colsfm decision 5), so the condition fires and every
`sensor_from_rig` is frozen — with no warning. **A fix is in progress by another
worker** — `colsfm/extrinsic_refinement.py` and
`tests/colsfm/test_extrinsic_refinement.py` appeared in the working tree while
this audit was being written — **and was deliberately not run here**; the finding
is cited from that worker's analysis, not re-measured, and this verdict describes
the state before the fix.

**Verdict — Blob:** the mechanism **holds** and demonstrably moves the
calibration by ~1 prior sigma; the *accuracy benefit* of §6.4.1 is **not
testable here** for want of extrinsic ground truth and the KITTI sequences.
**Verdict — colsfm:** **does not hold** — the flag is accepted and silently does
nothing.

---

**Addendum (after the audit ran).** The no-op was fixed in two commits. `7ec2f37` gives the rig a
camera reference sensor so COLMAP no longer freezes `sensor_from_rig`; a 20 mm perturbation
then recovers to 0.7 mm. Unregularised, the refinement overfit Galileo (extrinsics walked
9-235 mm, ATE 4.33 -> 5.70 mm). `2f453e1` adds the blob's prior terms on pyceres (absolute per
camera, relative per co-observed camera pair per frame; sigmas 0.01 m / 2 deg recovered from
the blob's own logged costs), alternating with pycolmap BA. Measured in
`data/bench/galileo_ext_compare.md`: ATE **4.281 mm** (fixed 4.327, blob 5.00), reprojection
0.898 px, extrinsic walk 1.2-2.8 mm against the blob's 1.1-10.9 mm, 4 of 7 shift directions
aligned with the blob's, 4/4 acceptance bounds. Remaining difference: the blob moves the front
stereo pair ~10 mm, colsfm ~2 mm, with prior budget unspent, so the difference sits in the
reprojection evidence, not the priors. Verdict for colsfm therefore changes to **holds**.

## C9 — Format support

> "The system exports sparse point cloud maps using COLMAP's sparse
> reconstruction format, enabling visualization and further processing of the
> reconstructed 3D structure." (§5)

> "Optimized camera trajectories are exported in TUM format, facilitating
> quantitative evaluation using established tools such as evo and other
> trajectory analysis frameworks." (§5)

**Evidence — both models load with `pycolmap.Reconstruction.read_text`:**

| | blob (`galileo_blobref/cusfm/sparse`) | colsfm (`galileo_colsfm/cusfm/sparse`) |
|---|---|---|
| files | `cameras.txt`, `images.txt`, `points3D.txt` | `cameras.txt`, `images.txt`, `points3D.txt`, **`rigs.txt`**, **`frames.txt`** |
| `num_rigs` / `num_frames` as read | 8 / 224 (synthesised: one trivial rig and one frame per image) | **1 / 29** (the real rig and its 29 rig instants) |
| `num_cameras` | 8 | 8 |
| `num_images` | 224 | 226 |
| `num_points3D` | 5065 | 6195 |
| `mean_reprojection_error` as written | **2.0** | 1.334 |
| TUM output | `output_poses/merged_pose_file.tum` | `output_poses/merged_pose_file.tum` |

Two caveats on the blob's side, both from `docs/spec/export.md`:

1. **The reprojection error in `points3D.txt` is not a measurement.**
   `kpmap_to_colmap` hardcodes `ERROR = 2.0` on every point (§3.5). Any tool that
   reads that column — including COLMAP's own GUI — is reading a constant.
   `colsfm.benchmark` recomputes it from the tracks for exactly this reason.
2. **No rig or frame model is exported**, so a COLMAP 4.x consumer sees eight
   unrelated cameras rather than one rig. That is a real loss of information for
   a system whose central claim (§4.2, §4.4) is the rig formulation.

A third, smaller one: the blob's `images.txt` can be mis-parsed by hand-rolled
readers when an image has zero observations
(`docs/spec/pycolmap-capabilities.md` §10); `pycolmap.Reconstruction.read_text`
handles it correctly, which is why the benchmark uses it.

**Verdict — Blob:** **holds with caveats** — the formats are real and load, but
the reprojection column is a constant and the rig structure is dropped on export.
**Verdict — colsfm:** **holds** — same formats, plus `rigs.txt`/`frames.txt`,
with a true reprojection error.

---

## C10 — Efficiency against COLMAP

> "Specifically, in view-graph mode, the mapping time is reduced by a factor of
> 20 (from 340.467s to 16.458s per 100 frames) compared to COLMAP, and by a
> factor of 6 compared to GLOMAP. This significant improvement leads to superior
> overall performance, with cuSfM's total processing time being only 16.931% of
> COLMAP's (58.824s vs. 346.162s) ..." (§6.2)

> "Moreover, COLMAP was unable to reconstruct the entire trajectory within a
> single model, fragmenting the reconstruction into multiple disconnected
> components." (§6.3.1)

The paper's numbers are KITTI's and cannot be reproduced (C14). What can be done
is to run COLMAP's own pipeline on Galileo, where both cuSfM systems have already
been measured, and see whether the *ratios* survive.

**How COLMAP was configured**, by
`pixi run -e colsfm python -m tools.audit.colmap_baseline_galileo`:

* Features extracted once per camera folder, so each of the eight folders gets
  its own COLMAP camera carrying the dataset's **known** rectified PINHOLE
  intrinsics with a prior focal length (COLMAP logs
  `Focal Length: 852.92px (Prior)`). This is deliberately generous — cuSfM is
  handed the same calibration, and withholding it from COLMAP would inflate the
  gap.
* Two feature stacks: SIFT-GPU + brute force (the paper's COLMAP configuration)
  and COLMAP's own ALIKED-n16rot + LightGlue (which isolates the mapper from the
  features).
* Two pairings: `match_sequential` with `overlap = 10` and loop detection off
  (again the paper's configuration) and `match_exhaustive`. Sequential matching
  orders images by name, and Galileo's names are
  `<camera_folder>/<timestamp>.jpeg`, so a sequential pass walks one camera at a
  time and only touches another at a folder boundary. Reporting only that
  configuration on an eight-camera rig would be a straw man, so exhaustive is
  reported beside it.
* `pycolmap.incremental_mapping` with default `IncrementalPipelineOptions`.

**Measured, all on this host** (blob and colsfm rows from
`data/bench/galileo_compare.md`; COLMAP rows from
`data/bench/audit_colmap_*.json`):

| Pipeline | pairs | extract (s) | match (s) | map (s) | **total (s)** | models | largest model |
|---|---|---|---|---|---|---|---|
| **blob** (cuSfM, all 8 stages) | 336 | 8.64 | 2.25 | 6.69 | **40.07** | **1** | 224 / 226 |
| **colsfm** (6 stages, loops off) | 331 | 8.01 | 8.15 | 2.70 | **19.35** | **1** | 225 / 226 |
| COLMAP SIFT + sequential | 1553 | 3.86 | 9.52 | 49.51 | **62.90** | 4 | 56 / 226 |
| COLMAP SIFT + exhaustive | 25425 | 5.37 | 34.75 | 195.84 | **235.96** | 2 | 195 / 226 |
| COLMAP ALIKED + LightGlue + sequential | 1553 | 10.24 | 24.88 | 366.57 | **401.68** | 3 | 117 / 226 |
| COLMAP ALIKED + LightGlue + exhaustive | 25425 | ~12 | 314 | **> 585, killed** | **> 900** | — | 169 of 226 registered when the time box expired |

The ALIKED + exhaustive row is a **time-box failure, reported as one**: COLMAP's
own timer logs 5.228 minutes for the matching stage, mapping then ran for another
585 s and had registered 169 of 226 images when the 15-minute limit killed it.
That is itself evidence for the claim under audit — the configuration that gives
COLMAP the same features cuSfM uses, on the same 226 images, does not finish in
the time cuSfM needs 40 seconds for — but it is an incomplete measurement and no
accuracy number is quoted from it.

The blob's 40.07 s includes stages 2-5 (BoW 11.73 + association 3.17 + pose
graph 3.58 + pair selection 1.94 = 20.42 s) for which COLMAP has no counterpart.
Its extract + match + map subtotal — the only three stages the paper's Table 2
itemises — is **17.58 s**.

**Mapping-stage ratios** (the paper's 20x):

| COLMAP configuration | vs blob's 6.69 s | vs colsfm's 2.70 s |
|---|---|---|
| SIFT + sequential (49.51 s) | 7.4x | 18.3x |
| SIFT + exhaustive (195.84 s) | **29.3x** | **72.5x** |
| ALIKED + sequential (366.57 s) | **54.8x** | **135.8x** |

**Total-runtime ratios** (the paper's 16.931 %):

| | as a fraction of COLMAP SIFT + exhaustive (235.96 s) |
|---|---|
| blob, all 8 stages | **17.0 %** |
| blob, Table-2 stages only (17.58 s) | 7.5 % |
| colsfm | **8.2 %** |

The blob's 17.0 % lands within 0.1 points of the paper's own 16.931 %, on a
different dataset, a different host and a different camera configuration. That is
a coincidence of magnitudes rather than a reproduction, but the order of
magnitude is right and the direction is not in doubt.

**Fragmentation (§6.3.1) reproduces exactly.** COLMAP never produced a single
model on Galileo in any of the configurations run: 4, 2 and 3 disconnected
components. Both cuSfM systems produce one model covering 224 or 225 of 226
images.

**Three caveats that the paper's framing also carries.**

1. **cuSfM is handed the trajectory; COLMAP is not.** §6.2 says so ("the system
   effectively utilizes coarse initial trajectory estimates"), and it is the
   honest explanation for most of the mapping-time gap: COLMAP is solving a
   harder problem — global registration from scratch — while cuSfM is refining a
   prior. Calling that an "order-of-magnitude runtime improvement" over COLMAP is
   true of the wall clock and misleading about the algorithms.
2. **cuSfM is handed the rig; COLMAP is not.** Eight cameras rigidly coupled is
   a much smaller parameter space than eight free cameras, and COLMAP 4.2's rig
   support was not used here because the paper's baseline is monocular.
3. **Sequential matching is a bad fit for this data** and produces the worst
   coverage (56/226). The exhaustive rows are the fair comparison and are the
   ones quoted above.

**Verdict — Blob:** the runtime claim **holds** on this dataset — 17.0 % of
COLMAP's wall clock, mapping 29x faster than COLMAP's, and a single model where
COLMAP fragments into two to four. It **holds with a caveat**: the comparison
measures a prior-refining pipeline against a from-scratch one, and Table 2's own
totals exclude 46 % of cuSfM's measured wall clock (C1).
**Verdict — colsfm:** **holds, more strongly** — 8.2 % of COLMAP's wall clock
and mapping 72x faster, from the same prior.

---

## C11 — Accuracy against COLMAP

> "The experimental results demonstrate cuSfM's superior performance across most
> sequences. In Sequence 00, cuSfM achieved remarkable accuracy with RMSE of
> 0.040m, representing a 90.1% improvement over COLMAP (0.406m) ..." (§6.3.1)

The SDG numbers themselves are out of reach (C14). What Galileo gives is the same
comparison on the same host, against real ground truth.

**Measured** — all ATE figures are rig-frame, joined to `ground_truth.txt` on
timestamp within 20 microseconds and aligned rigidly with the scale held at 1.0,
which is `colsfm.benchmark`'s definition:

| System | registered | points | mean reproj (px) | **ATE rigid, scale 1 (mm)** | ATE after SIM(3) (mm) | scale the fit needed |
|---|---|---|---|---|---|---|
| input trajectory (the prior) | — | — | — | 3.97 | 2.82 | 0.98593 |
| **blob** | 224 / 226 | 5065 | 1.550 | **5.00** | 2.52 | 0.97844 |
| **colsfm** | 225 / 226 | 6195 | 1.334 | **4.33** | 2.62 | 0.98270 |
| COLMAP SIFT + sequential (largest of 4) | 56 / 226 | 7148 | 0.691 | 0.87 \* | 0.78 \* | **0.05305** |
| COLMAP SIFT + exhaustive (largest of 2) | 195 / 226 | 29814 | 0.931 | **26.06** \* | 25.49 \* | **0.09306** |
| COLMAP ALIKED + sequential (largest of 3) | 117 / 226 | 11918 | 1.390 | 109.35 \* | 97.45 \* | **0.07216** |
| COLMAP ALIKED + exhaustive | did not finish in 15 min | — | — | — | — | — |

\* **The COLMAP rows are only meaningful after a SIM(3) fit has handed them the
scale from ground truth.** Without it their rigid ATE is 3.3-3.5 m, because a
monocular COLMAP reconstruction of this rig carries no metric scale at all: the
fitted scales are 0.053-0.093, i.e. the reconstructions came out 11 to 19 times
too small. Those rows therefore answer "how good is COLMAP's *shape*", not "how
good is COLMAP's trajectory", and the cuSfM rows answer the second question. On
the paper's own metric — ATE against metric ground truth — COLMAP does not
produce a comparable output at all.

**With the scale granted**, the ordering still favours cuSfM on every
configuration that reconstructs most of the sequence:

* the best-covering COLMAP model (SIFT + exhaustive, 195/226 images) is
  **26.06 mm**, 5.2x the blob's 5.00 mm and 6.0x colsfm's 4.33 mm;
* ALIKED + sequential (117/226) is **109 mm**, 22x the blob's.

**One row goes the other way and is reported rather than buried.** COLMAP
SIFT + sequential reaches **0.87 mm** — five times better than either cuSfM
system. It covers 56 of 226 images, which is one stereo pair over 28 rig
instants: two rigidly-mounted, heavily-overlapping cameras with 7148 points at
0.691 px. Reconstructing one stereo pair over a 0.66 m sweep is an easy problem
and COLMAP solves it very well. It is not a trajectory for the rig, it has the
wrong scale by a factor of 19, and it drops 75 % of the imagery — but it does
say plainly that COLMAP's bundle adjustment is not the weak part. What cuSfM buys
is coverage, metric scale and a single connected model.

**Verdict — Blob:** **holds** on Galileo — 5.00 mm against 26.06 mm for the
best-covering COLMAP model, and COLMAP cannot produce a metric trajectory at all.
The margin is smaller than the paper's 90 %, and the mechanism is different from
what §6.3.1 implies: COLMAP loses on scale and coverage, not on bundle adjustment
accuracy.
**Verdict — colsfm:** **holds** on Galileo, marginally ahead of the blob
(4.33 mm against 5.00 mm).

---

## C12 — Does refinement improve the initial trajectory?

> "The results demonstrate that cuSfM can effectively refine trajectories
> regardless of the initializer used, while maintaining or improving upon the
> accuracy of the input trajectories." (§6.3.2)

This is the one claim that can be tested directly against ground truth on the
data available here, and **on Galileo it does not hold**.

**Method.** `pixi run -e colsfm python -m tools.audit.trajectory_audit` scores
the input trajectory, the blob's output and colsfm's output through one code
path, so no difference can come from the metric:

* the same rig-pose rule — `FramesMeta.rig_frames`, lowest keyframe id of each
  `synced_sample_id`, the rule verified to reproduce cuSfM's own
  `vehicle_frames_meta.json` to 1.4e-16 m
  (`docs/spec/pose_graph_main.md` §4);
* the same join to `ground_truth.txt` — nearest timestamp within 20 microseconds
  (`colsfm.benchmark.GROUND_TRUTH_TOLERANCE_MICROSECONDS`);
* the same rigid alignment with the scale held at 1.0
  (`colsfm.benchmark.align_rigid`);
* the same sample set: all three cover the same 29 rig frames, so the "own" and
  "common" rows are identical.

**Result** (`data/bench/audit_galileo_trajectory.json`):

| Trajectory | matched | ATE RMSE (mm) | max (mm) | would-be scale | ATE after SIM(3) (mm) |
|---|---|---|---|---|---|
| **input** (the prior handed to cuSfM) | 29 | **3.97** | 6.82 | 0.98593 | **2.82** |
| blob | 29 | **5.00** (+1.03) | 8.31 | 0.97844 | **2.52** |
| colsfm | 29 | **4.33** (+0.36) | 7.75 | 0.98270 | **2.62** |

The blob's 5.00 mm and colsfm's 4.33 mm reproduce
`data/bench/galileo_compare.md` exactly, so the harness is the same one that
produced the benchmark report. The input row is new: **NOTES.md never scores the
input trajectory against ground truth**, and neither does any benchmark report —
`galileo_compare.md` compares each run against the input and against ground
truth, but never the input against ground truth. That missing cell is what this
claim turns on, which is why nobody had noticed the sign.

**Both systems made Galileo's trajectory worse against ground truth**, the blob
by 26 % and colsfm by 9 %.

**Why, precisely.** The last column separates shape from scale. Under a
scale-free SIM(3) fit the blob is the *best* of the three (2.52 mm against the
input's 2.82 mm): bundle adjustment genuinely improved the shape of the
trajectory. What it also did was shrink it. The input's would-be scale against
ground truth is 0.98593 — already 1.4 % small — and the blob's output is
0.97844, i.e. 2.2 % small. On a 0.66 m sweep that extra 0.75 % of scale error is
worth more than the shape gain, and the rigid, scale-1 metric — the honest one
for a system that claims metric output — comes out worse.

So the mechanism is not broken; the *scale* is drifting, and cuSfM's bundle
adjustment has nothing anchoring it except the
`absolute_pose_translation_error_meters: 1` prior, a 1 m leash on a 0.66 m
trajectory.

**RoboCap has no ground truth**, so nothing there can confirm or refute §6.3.2.
What the existing reports show is disagreement, not error
(`data/bench/robocap_compare.md`, `data/bench/robocap_loops_compare.md`):

| | blob | colsfm, no loops | colsfm, `--loop-closure` |
|---|---|---|---|
| loop edges | 90 | 0 | 74 |
| vs input trajectory (mm RMSE / max) | 334.1 / 627.3 | 461.3 / 1134.9 | **312.7 / 608.4** |
| rig poses vs the blob (mm RMSE / max) | — | 238.3 / 796.4 | **146.4 / 289.6** |

Loop closure moves colsfm's trajectory towards the blob's (238 mm -> 146 mm) and
towards the input (461 mm -> 313 mm), which is the direction the blob's own loop
closure moves it. That is consistent with §6.3.2's mechanism, and it is the
strongest statement RoboCap supports.

**Verdict — Blob:** **does not hold** on the only sequence here with ground
truth. Refinement improved the trajectory's shape (2.82 -> 2.52 mm under SIM(3))
and degraded its metric accuracy (3.97 -> 5.00 mm rigid) through a 0.75 % scale
contraction.
**Verdict — colsfm:** **does not hold**, by a smaller margin (3.97 -> 4.33 mm),
from the same cause.

---

## C13 — Localization mode

> "CuSfM offers two operational modes for existing maps: adjust mode and fixed
> mode, in addition to supporting single-frame localization against the map. To
> the best of our knowledge, no existing open-source SfM project provides such
> comprehensive functionality for diverse mapping and localization scenarios."
> (§6.4.2)

**Blob.** `run_global_localization_main` is one of the 21 shipped binaries in
`pycusfm/x86_cuda13/bin`, alongside `pose_graph_association_main`,
`query_map_config.pb.txt` and `localizer_config.pb.txt` in every config
directory. The map-fixed / map-adjust distinction of §4.3 Eq. 11 is real in
`keypoints_mapper_main`'s absolute-pose constraints. This audit does **not**
exercise the mode: the SDG two-trajectory dataset the paper demonstrates it on is
not available, and neither Galileo nor RoboCap has a second session.

**colsfm.** There is no equivalent. `colsfm/` implements the eight mapping
stages only; nothing reads or writes a prior map, and there is no localization
entry point.

**Verdict — Blob:** **holds** to the extent that the binary and its configs
exist; **not exercised here**.
**Verdict — colsfm:** **does not hold** — no such functionality exists.

---

## C14 — The specific KITTI and SDG numbers

Tables 2 (runtime on KITTI), 3 (ATE on SDG), 4 (cuSfM vs CuVSLAM on KITTI),
5 (cuSfM vs ORB-SLAM2 on KITTI) and 6 (extrinsic refinement on KITTI) are **not
reproducible here**, for three separate reasons:

1. **No KITTI imagery.** The repository ships `data/kitti/config/` but no
   sequences, and downloading KITTI is outside this audit's network budget.
2. **No SDG dataset.** The paper points at
   `github.com/nvidia-isaac/pyCuSFM/tree/main/data/SDG`; it is not present in
   this checkout.
3. **No CuVSLAM or ORB-SLAM2 initial trajectories** for either dataset, which
   Tables 4 and 5 compare against by construction. `skip_cuvslam=True` is the
   default here because both datasets already ship a trajectory (NOTES.md).

One structural observation can still be made without the data, and it is stated
under C1: **Table 2's "Total" is the sum of only its three itemised stages**, so
the BoW construction and search-tree build that §6.2's own prose says the
view-graph mode requires are absent from both the table and the totals. On the
two datasets measured here those stages are 46-62 % of cuSfM's wall clock. Any
reproduction of Table 2 should first establish what the paper's "Total" is
measuring.

**Verdict — both:** **not testable here.**

---

## What would settle the open ones

1. **Download KITTI odometry sequences 00, 02, 05, 06, 09** (the ones Tables 2,
   4, 5 and 6 use). That unlocks C10's runtime comparison on the paper's own
   data, C3's radius-vs-view-graph accuracy comparison at the scale where it
   matters (a 0.66 m sweep is the wrong regime for a claim about drift), C12 on
   sequences long enough for refinement to have something to fix, and C8's
   Table 6. It also allows a direct check of whether Table 2's "Total" excludes
   the BoW stages, by running the blob on KITTI with `runtime.csv` enabled and
   comparing stage by stage.
2. **Obtain the SDG dataset** from the pyCuSFM repository. It is the only
   evidence offered for C11's headline numbers (90.1 % and 88.1 % RMSE
   improvements over COLMAP) and the only demonstration of C13's localization
   mode, which needs two trajectories through one intersection.
3. **Land the `colsfm` extrinsic-refinement fix** (in progress). Until COLMAP's
   `ParameterizeRigsAndFrames` sees a reference sensor that participates in the
   problem — or colsfm switches the rig reference to a camera and re-bases every
   pose — `optimize_extrinsics` will keep accepting the flag and freezing the
   extrinsics. Once it lands, C8 becomes a two-sided comparison against the
   blob's measured 10.91 mm / 1.40 deg on Galileo.
4. **Find or build extrinsic ground truth for Galileo.** The 10.91 mm of
   relative extrinsic movement measured here is real, but whether it is a
   correction or a drift cannot be decided without an independent calibration.
5. **Settle C3 on a drifting sequence.** On Galileo full radius search is 6 %
   *more* accurate than the view graph, the opposite of Table 4's KITTI ordering.
   The two claims are not necessarily in conflict — a 0.66 m sweep has no drift
   for redundant pairs to confuse — but only KITTI-scale sequences can show which
   effect dominates in the regime the paper targets.
6. **Re-measure C12 on a sequence with metric ground truth and real drift.**
   Galileo's answer — better shape, worse scale — is a scale-observability
   result on a 0.66 m sweep, and may not generalise. A 100 m+ sequence with
   ground truth would separate "cuSfM loses scale" from "cuSfM loses scale on
   short sequences".
7. **Settle the loop-edge rotation disagreement of C5.** Two independent
   estimators (colsfm's generalized resection and
   `pycolmap.estimate_generalized_relative_pose`) agree with each other to
   0.35 deg and disagree with the blob by 6.5 deg on RoboCap. Ground truth for
   any sequence with revisits would decide which is right, and would tell whether
   the paper's §4.1 estimator is the better one or merely a different one.
