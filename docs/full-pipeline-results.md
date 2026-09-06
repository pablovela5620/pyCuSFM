# Full-pipeline runs: loop closure + extrinsic refinement, everything but localization

Asked for on 2026-09-05: "the full chart for each experiment ... loop + extrinsics refinement
so EVERYTHING except localization", then trimmed to "two reasonable experiments". Until this
batch, no KITTI or RoboCap run — blob or colsfm — had extrinsic refinement on, and the one
Galileo colsfm run that did had loop closure off. Everything below is new.

Chart: `data/bench/full/full-pipeline-runtime.png` (drawn by `tools/audit/full_pipeline_chart.py`
from each run's `runtime.csv`; spec `data/bench/full/chart_spec.json`). Reports and score
files: `data/bench/full/`. Run directories: `/tmp/colsfm_runs/full/<dataset>_<features>_<ba>/cusfm`.

## The two colsfm configurations

Both with `--loop-closure --optimize-extrinsics --ba-backend caspar` in `colsfm-caspar-fisheye`:

1. **blob engines + CASPAR** — `--features-backend tensorrt --matching-backend tensorrt`: the
   blob's own ALIKED and LightGlue TensorRT engines, COLMAP's GPU bundle adjuster with one
   Ceres polish, then the regularised extrinsic refinement (Ceres). The accuracy pick.
2. **RaCo + CASPAR** — `--features-backend raco --matching-backend raco`: RaCo-ALIKED at native
   resolution and LightGlue+, same mapper and refinement. The speed pick.

The blob was rerun with `--optimize_extrinsics --ba_frame_type vehicle_rig` on Galileo only
(36 s). KITTI and RoboCap keep the existing blob references, which ran loop closure but **not**
extrinsic refinement; those two blob rows are therefore one stage short, and the caption says so.
(First attempt at the Galileo blob rerun took the CLI's default `--min_inter_frame_distance 0.5`,
kept 56 of 226 images and finished in 13 s; the reference used 0.0 and so does the row below.)

## Seconds per stage bucket

| run | extraction | loop + pose graph | matching | mapping | extrinsics | export | total | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Galileo: blob, loops + extrinsics | 9.4 | 11.5 | 3.7 | 6.5 | in mapping | 1.4 | **32.4** | ATE 2.69 mm |
| Galileo: colsfm, blob engines + CASPAR | 6.0 | 8.3 | 2.1 | 4.0 | 2.5 | 0.3 | **23.2** | ATE 5.21 mm |
| Galileo: colsfm, RaCo + CASPAR | 2.6 | 8.5 | 2.0 | 4.0 | 2.5 | 0.3 | **20.0** | ATE 5.08 mm |
| KITTI 06: blob, loops, no extrinsics | 96.8 | 1306.8 | 42.0 | 143.1 | — | 7.4 | **1596.1** | Sim(3) ATE 1.328 m |
| KITTI 06: colsfm, blob engines + CASPAR | 57.0 | 49.7 | 26.8 | 62.3 | 16.3 | 2.6 | **214.7** | Sim(3) ATE 0.733 m |
| KITTI 06: colsfm, RaCo + CASPAR | 9.6 | 46.5 | 19.4 | 62.4 | 12.7 | 2.6 | **153.2** | Sim(3) ATE 0.933 m |
| RoboCap: blob, loops, no extrinsics | 183.9 | 705.7 | 58.1 | 178.7 | — | 7.1 | **1133.5** | 334 mm vs input |
| RoboCap: colsfm, blob engines + CASPAR | 116.7 | 106.1 | 40.8 | 104.4 | 118.2 | 6.4 | **492.6** | 245 mm vs input, 189 mm vs blob |
| RoboCap: colsfm, RaCo + CASPAR | 83.4 | 102.3 | 38.1 | 120.0 | 127.5 | 6.5 | **477.9** | 265 mm vs input, 168 mm vs blob |

The blob's extrinsic refinement is a flag on its single mapper pass, so on Galileo it is inside
the 6.5 s mapping figure. "loop + pose graph" for the blob is vocabulary + index + association +
`pose_graph_main`, which is mostly loop verification (`docs/loop-stage-cost.md` §2).

## What changed with the refinement on

**KITTI 06: nothing, for 13-16 s.** Sim(3) ATE 0.733 m against 0.732 m without refinement (blob
engines) and 0.933 against 0.937 (RaCo); the refinement ran 8-9 rounds and stopped on its own
tolerances. A two-camera rectified stereo rig has little extrinsic error to find.

**RoboCap: the refinement is now the most expensive stage, and it helps.** 118-128 s, the
20-round ceiling both times, on a 4528-image fisheye rig — more than mapping. Disagreement with
the input trajectory fell from 278 mm (loops, no refinement, RaCo) to 245 / 265 mm, better than
the blob's 334 mm, and the two runs land 189 / 168 mm from the blob's poses. Reprojection is
1.81 / 1.86 px (blob 1.49; loops without refinement 1.95). The 20-round cap is worth a look:
the stage may be converging slowly rather than done.

**Galileo: the blob's refinement wins, and it is the scale.** With `--optimize_extrinsics` the
blob goes from 5.00 mm ATE to **2.69 mm**, and its would-be Sim(3) scale against ground truth
goes from 0.978 to 1.004: its refinement re-scales the rig. Every colsfm configuration stays at
0.977-0.983 and 4.3-5.2 mm, refinement or not, CASPAR or Ceres, any feature backend:

| Galileo, loops + extrinsics on | ATE (mm) | scale vs GT | max extrinsic move |
|---|---:|---:|---|
| blob | 2.69 | 1.0039 | 9.34 mm / 0.90 deg |
| colsfm pycolmap + Ceres | 4.30 | 0.9829 | — |
| colsfm blob engines + Ceres | 5.03 | 0.9783 | — |
| colsfm RaCo + Ceres | 5.03 | 0.9784 | — |
| colsfm blob engines + CASPAR | 5.21 | 0.9773 | 2.64 mm / 0.27 deg |
| colsfm RaCo + CASPAR | 5.08 | 0.9781 | — |

(`tools/audit/extrinsics_audit.py`, `data/bench/full/ext_audit_*.json`.) The blob moves its
cameras up to 9.3 mm under the same 1 cm / 2 deg priors; colsfm's alternation — extrinsics
with poses and points frozen, then poses with extrinsics frozen — moves them 2.6 mm and never
re-scales the map. That is the open item this batch surfaced: the joint solve can trade rig
baseline against map scale, the alternation cannot. Every colsfm Galileo row fails the 1.10x
ATE bound against the *refined* blob (2.95 mm) while passing it against the unrefined one.

## Bounds against the blob

KITTI: registered and ATE pass, reprojection fails (0.534 / 0.572 px against a 0.500 ceiling)
as in every KITTI run. RoboCap: `data/bench/full/robocap_*.md`. Galileo: registered and
reprojection pass, ATE fails as above.
