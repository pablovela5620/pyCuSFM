# Where the loop-closure stage's time goes, and why the 721 s number was wrong twice

Written 2026-09-05 after the question "why is the BoW/loop/PGO part so much faster than
cuSFM? that is a code smell". Method: observe before guessing — the blob's own timer
summaries, `colsfm`'s stage clock, and one `cProfile` of the stage alone
(`tools/audit/profile_loop_stage.py`, dump in `data/bench/robocap_loop_stage_raco.prof`).

## 1. The smell was real: a skipped stage reported as a fast one

`docs/caspar-fisheye-adapter.md` §9.5 reported 0.25 s for "BoW + loop + pose graph" against
the blob's 705.66 s. Loop closure was off (`PipelineOptions.loop_closure` defaulted to
False), so `colsfm` built no vocabulary, searched for nothing, and its pose graph re-solved a
chain the input already satisfied (cost 1e-27, one iteration). The blob's 705.66 s built a
vocabulary, searched, verified 3190 candidate pairs and optimised 90 loop edges. The default
is now True (NOTES.md decision 10, revised); `--no-loop-closure` is the labelled ablation.

## 2. The blob's "pose graph optimisation" is 87 % loop verification

The harness labelled `pose_graph_main`'s whole runtime "4 pose graph optimisation". The
binary's own timer summary (`data/cusfm_runs/robocap_blobref/cusfm/pose_graph/pose_graph_main.txt`)
says what it did in its 386.61 s on RoboCap:

| timer | events | total (s) |
|---|---:|---:|
| Incremental graph (the whole loop) | 1 | 380.58 |
| PoseEstimateFromTwoViewNormalized | 3190 | **336.06** |
| LightGLue TensorRT End to end | 3190 | 11.95 |
| RelativePoseRefine | 1591 | 11.70 |
| Optimize pose graph | 25 | **3.74** |

So the blob's loop cost on RoboCap is `generate_bow_*` 243.39 s + the association stage's
loop pass 43.02 s (`RetrievalLoopAssociations` in `associations_main.txt`; its 11 323
association pairs are all but 7 consecutive) + 380 s of candidate verification inside
`pose_graph_main`: about **666 s of loop work and 4 s of optimisation**. The optimiser is
cheap in both systems. The report label is now
"4 pose graph (blob: loop verification + optimisation)" (`colsfm/bench_report.py`), and the
earlier RoboCap row "3 loop-closure association 75.66 vs 721.29, 9.53x" compared `colsfm`'s
whole loop stage with a third of the blob's.

## 3. The 721 s was three code versions ago

`data/bench/robocap_loops_compare.md` (06:20 today) measured a `colsfm` loop stage of
721.29 s with the pycolmap matcher. The commit that recorded it, `20b194d`, is the one that
*fixed* its main cost: the stage re-matched all 14 443 requested pairs, 8 400 of which stage
4 had already matched (NOTES.md: "of which most was re-matching 8 400 pairs"). Two later
changes moved the rest: the RaCo LightGlue+ matcher (`4c639aa`, 7 ms per pair against
pycolmap's 24) and batch triangulation in the rig-pose estimator (`605887b`; the
per-point `pycolmap.triangulate_multi_view_point` "costs about a second per rig frame").
Nobody re-measured. Today's full-RoboCap run (`data/bench/robocap_full_blob_vs_caspar_loops.md`):

| stage | blob (s) | colsfm, RaCo + CASPAR (s) |
|---|---:|---:|
| BoW vocabulary + index | 243.39 | (inside the loop stage) |
| loop association | 75.66 | 103.43 |
| pose graph, blob incl. verification | 386.61 | 0.57 |
| **loop work + optimisation** | **705.66** | **104.00** |

86 loop edges against the blob's 90; rig poses 159.06 mm / 1.236 deg from the blob's
(210.78 mm without loops), 277.79 mm from the input against the blob's 334.10.

## 4. Inside the 103 s (cProfile, full RoboCap, RaCo matcher, stage alone: 112.5 s)

| part | seconds | note |
|---|---:|---|
| vocabulary-tree retrieval index | 31.6 | `all_pairs_l1_score` 14.7 (dense N² intersection, one `np.ix_` add per word), `assign_words` 10.2, `train_vocabulary` 5.7 |
| LightGlue+ over 8274 new candidate pairs | ~47 | 4924 of 13 198 requested pairs were already matched and skipped; ~5.7 ms per pair, GPU-side, in threads cProfile does not see |
| square-covering NMS (`_cover_squares`) | 11.5 | pure Python `set` per pair, 15 000 calls |
| `pycolmap.verify_matches` | 10.5 | geometric verification of the same pairs |
| rig-resection measurement | ~12 | 1232 `estimate_and_refine_generalized_absolute_pose` (4.3 s) + 1187 `estimate_generalized_relative_pose` (4.1 s) + batch triangulation 0.6 s, over 876 verified rig pairs |
| candidate bookkeeping, reads | ~1 | 10 134 `read_matches`, 17 506 `record_pair` |

Two of these are `colsfm`'s own Python and are the obvious next targets: the NMS (11.5 s,
vectorisable or movable to the matcher's GPU pass) and the all-pairs L1 score (14.7 s; a
sparse `bow @ bow.T` with a min kernel, or scoring only the queried rows). Together they are
a quarter of the stage. The matcher and the verification are already the fast backends.

## 5. What loop closure does to the map on RoboCap

With loops the mapper sees 2.66 M observations (0.95 M blob, 1.39 M without loops), a mean
track length of 8.34 (4.65) and a mean reprojection error of **1.945 px** (blob 1.486,
`colsfm` without loops 1.520). The trajectory is the closest to the blob of any RoboCap run
(159 mm) but the per-point fit is the worst, and the run fails the harness's 1.10x
reprojection bound (1.635 px). Whether the extra observations are wrong merges or honest
long tracks on a fisheye rig with 1.5 px calibration error is the open question; it was not
visible while loop closure was off.

## 6. Galileo under the new default

`pixi run -e colsfm python -m colsfm run --input-dir data/r2b_galileo ...` with loop closure
on: 226 images, retrieval index 8.37 s, 5791 hits all inside the 10 s gap, 0 candidate pairs,
0 edges; 24.19 s total against 19.35 s before, 4/4 bounds against the blob
(`/tmp/colsfm_runs/galileo_default_loops_compare.md`: 225 registered, 1.333 px, 4.32 mm ATE,
0.60x runtime). The stage costs 8 s there and changes nothing, which is the honest price of
running what the blob runs.
