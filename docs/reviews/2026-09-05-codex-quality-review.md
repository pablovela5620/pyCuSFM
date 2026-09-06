# Codex code-quality review of feat/pycolmap-open-pipeline (2026-09-05)

Model gpt-6-astra, reasoning effort xhigh, read-only sandbox, thermo-nuclear-code-quality-review rubric. Verdict: REQUEST CHANGES. Kept verbatim; the follow-up plan is tracked separately.

This branch needs a structural pass before merge. The main problem is duplicated ownership: execution, backend policy, serialization, and runtime state each have multiple owners. Several of those splits already cause observable defects.

I reviewed `main...HEAD` on `feat/pycolmap-open-pipeline`, confirmed 71 commits, and excluded the requested documentation, data, lockfile, and generated-artifact content.

1. **Use one pipeline runner; make the walkthrough an observer. — CAREFUL**

   [pipeline.py:1252](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/pipeline.py:1252) and [walkthrough.py:1763](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/walkthrough.py:1763) independently sequence the same stages, construct matching and mapping options, handle refinement, and export results. They also maintain separate clocks and timing contexts.

   The duplication has already diverged: the walkthrough constructs a smaller set of matching/BA options and never writes the pipeline summary. Its “same artifacts” test excludes that summary. Every new backend or stage therefore requires changes in several places, with no complete equivalence check.

   **Restructure:** keep one explicit, sequential runner. Let a synchronous observer consume typed stage results and produce the walkthrough recording. Centralize option resolution and run reporting. Keep visualization outside stage timing, and consume mutable reconstruction results before later stages change them.

   Delete single-field wrappers such as `FeatureExtractionStageResult(report)` and `ReconstructionStageResult(mapping)` where they add no contract. Return the existing report/result directly. This removes an entire execution path and a layer of pass-through objects.

2. **Introduce a coherent run lifecycle and publication boundary. — CAREFUL**

   [pipeline.py:1268](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/pipeline.py:1268) clears `runtime.csv` while leaving previous outputs, including `summary.json` and the sparse model. [database.py:85](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/database.py:85) deletes the previous database before constructing its replacement. Export and summary writes then update the destination incrementally.

   A failed rerun can therefore leave artifacts from different runs in one directory. [benchmark.py:186](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/benchmark.py:186) checks file presence, not common provenance or completion.

   Failure reporting is also wrong: [pipeline.py:480](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/pipeline.py:480) records timing and prints “finished” in `finally`, including when the stage raises. I reproduced this without writing files.

   **Restructure:** use a run-scoped staging workspace and explicit running/succeeded/failed state. Derive CSV and summary timing from one stage ledger. Publish completed artifacts through a defined commit boundary; readers must reject incomplete or mixed runs. Preserve the current output layout through that publication layer.

   Use database transactions for coherent writes, especially keypoints plus descriptors. Geometry deletion followed by verification also needs a recovery boundary; wrapping calls that open separate database connections in an outer transaction will not solve it.

3. **Delete the loop-closure discovery run. Plan required matches directly. — CAREFUL**

   [pipeline.py:983](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/pipeline.py:983) calls `find_loop_edges` with a dummy matcher that records requested pairs and returns empty matches. It then matches those pairs and calls `find_loop_edges` again.

   This makes pair discovery depend on executing the geometric verification path with deliberately false input. It repeats shortlisting and setup, discards the first result, and couples future verification changes to scheduling behavior.

   The same block treats `raw_match_count == 0` as “not matched,” so a completed zero-match result is indistinguishable from missing work.

   **Restructure:** produce a typed `LoopSearchPlan` containing shortlisted rig pairs, required image pairs, geometry/index data, and retrieval diagnostics. Then batch missing matches and verify that plan once.

   Preserve local-map neighbor pairs, stereo pairs, ordering, raw-match semantics, and exact gate thresholds. Represent match presence separately from match count. The dummy callback, probe constant, and first verification pass can disappear.

4. **Consolidate native feature extraction and engine caching. — CAREFUL**

   [features_trt.py:313](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/features_trt.py:313) and [features_raco.py:354](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/features_raco.py:354) duplicate score filtering, top-k selection, coordinate conversion, descriptor packing, and database writes. RaCo also imports common preparation machinery from the sibling backend module.

   [tools/raco_extract.py:322](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tools/raco_extract.py:322) adds another engine builder and execution implementation. Their contracts have diverged: the tool hashes the ONNX file, while [tensorrt_runtime.py:185](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/tensorrt_runtime.py:185) names cached engines from the filename, profile tag, TensorRT version, and GPU architecture. `resolve_engine` accepts an existing file without checking graph contents or the supplied profile. Replacing an ONNX graph at the same path can silently reuse the old engine.

   **Restructure:** share native extraction preparation, feature selection, storage, and runtime ownership. Use small model adapters for bindings, normalization, score meaning, batch limits, and engine selection. Return a typed feature batch at the tensor-binding boundary.

   Give the shared engine cache a complete build identity: graph digest, profiles, precision, build settings, runtime, and target architecture. Use unique temporary files and coordinated publication. Support trusted prebuilt blob engines explicitly.

   RaCo matching already delegates to the TensorRT matcher. Preserve that sharing and the intentional backend differences in score semantics and subsampling order.

5. **Resolve configuration into policies once; stop carrying unresolved modes through the pipeline. — CAREFUL**

   [PipelineOptions:229](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/pipeline.py:229) exposes 21 fields across unrelated stages. [__main__.py:52](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/__main__.py:52) gives metadata-only stages the entire set, including a required output directory they do not use.

   The resulting state space includes inert combinations: native TensorRT paths bypass the CPU device choice; `native_resolution=False` overrides the RaCo engine choice; both `match_cap_mode="off"` and a `None` cap disable limiting. CASPAR strings are parsed only after feature extraction, matching, loop closure, and pose-graph work.

   CASPAR resolution is particularly scattered. [mapping.py:713](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/mapping.py:713) collapses probe errors into unsupported-camera results; an empty set bypasses the camera guard; [mapping.py:1222](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/mapping.py:1222) later switches backend by mutating the caller’s options. Override casting also silently turns `solver_iter_max=2.9` into `2`; I confirmed that behavior.

   **Restructure:** retain existing CLI spellings and defaults as an input adapter, then resolve:
   - selection options;
   - backend-specific feature and matching settings;
   - an explicit match-limit policy;
   - a RaCo sizing/engine policy;
   - a BA execution plan, including any Ceres polish.

   Give CASPAR capability detection a typed outcome that distinguishes unavailable, unsupported, and probe failure. Keep it in a compatibility/backend module. Preserve deliberate fallback behavior, but record the effective backend and reason explicitly. Validate cheap inputs before creating or replacing artifacts.

6. **Make CUDA allocation ownership exception-safe. — CAREFUL**

   [tensorrt_runtime.py:357](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/tensorrt_runtime.py:357) frees an existing buffer before allocating its replacement. If allocation fails, the old pointer and capacity remain recorded.

   A read-only stub probe confirmed that a subsequent smaller request returns that freed pointer. `close()` can also attempt to free it again. [GpuPreprocessor:649](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/tensorrt_runtime.py:649) has partial-construction leaks if a later allocation fails.

   `DeviceTensor` carries only a pointer and shape. The same-stream, dtype, capacity, and lifetime requirements exist only in documentation, while device inputs bypass checks performed for host arrays.

   **Restructure:** use an owning buffer object with pointer, capacity, dtype, and device identity. Make resize leave a valid state on failure—either allocate before swapping, or invalidate the old allocation record before a potentially failing replacement. Register cleanup as each constructor resource is acquired.

   Make borrowed device views private or explicitly tied to their owner and stream. Add allocation-failure and partial-construction tests using a fake CUDA boundary.

7. **Give `FramesMeta` one authoritative representation. — CAREFUL**

   [frames_meta.py:145](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/frames_meta.py:145) stores both typed domain fields and a mutable protobuf message. [to_json():229](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/frames_meta.py:229) serializes only the message; update/filter methods also reconstruct from that message.

   These representations can disagree despite the frozen dataclass. I changed a camera width with `dataclasses.replace`: the typed object reported `640`, serialization remained unchanged, and filtering restored `1920`. The test helper already uses this replacement pattern at [test_raco_backend.py:374](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tests/colsfm/test_raco_backend.py:374).

   Different consumers can therefore observe different calibration from the same object.

   **Restructure:** either make the typed model authoritative and overlay its fields onto a preserved protobuf template during serialization, or expose a controlled immutable facade over a private message. Preserve unmodeled protobuf fields. Test edits followed by serialization, filtering, pose updates, and extrinsic updates—not only untouched round trips.

8. **Six new files exceed 1,000 lines. Decompose them along ownership boundaries. — CAREFUL**

   These are all additions relative to `main`, not pre-existing large files.

   | File | Lines | Concrete decomposition |
   |---|---:|---|
   | [colsfm/pipeline.py](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/pipeline.py:1) | 1,416 | Resolved configuration; run lifecycle/reporting; one runner. Move loop planning into the loop subsystem. |
   | [colsfm/mapping.py](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/mapping.py:1) | 1,526 | BA backend/capability adapter; correspondence loading; point filtering; mapping-round orchestration. |
   | [colsfm/extrinsic_refinement.py](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/extrinsic_refinement.py:1) | 1,216 | Observation indexing; residuals/Jacobians/priors; solve construction; alternating refinement driver. |
   | [colsfm/walkthrough.py](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/walkthrough.py:1) | 1,906 | Stage visualization; blueprint construction; CLI configuration. Delete its pipeline runner. |
   | [demo_rerun.py](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/demo_rerun.py:1) | 1,994 | Dataset adapters; metadata/model I/O; cuSFM invocation; visualization. Leave a small entry point. |
   | [tests/colsfm/test_mapping.py](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tests/colsfm/test_mapping.py:1) | 1,978 | Mapping behavior; BA policy/capabilities; filtering; extrinsics; separately selected performance/integration tests. |

   The size matters because these files mix reasons to change. Moving arbitrary sections into `utils.py` would preserve that problem.

   The other named examples are **below** the threshold: `loop_pose.py` 931, `features_raco.py` 504, and `tensorrt_runtime.py` 714 lines. `loop_pose.py` still has useful boundaries between local-map construction, observation assembly, and pose estimation.

9. **Make the package variants reproducible and remove the copied environment matrix. — CAREFUL**

   [packages/pycolmap-caspar-fisheye/recipe.yaml:65](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/packages/pycolmap-caspar-fisheye/recipe.yaml:65) builds `/home/pablo/0Dev/forks/colmap-caspar-fisheye`. Its declared revision does not pin that working tree. The checked-out local contents determine the package, so this branch does not contain enough information to reproduce it.

   The [package test at line 193](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/packages/pycolmap-caspar-fisheye/recipe.yaml:193) checks that the fisheye enum exists. That does not prove the CASPAR fisheye adapter exists.

   Meanwhile, [pixi.toml:231](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/pixi.toml:231) starts a manually synchronized set of four environment definitions. Parsed comparison confirmed that dependencies apart from `pycolmap`, PyPI dependencies, activation, PyPI options, and system requirements are identical.

   **Restructure:** pin accessible source plus committed patches, or a source archive with a digest. Share the recipe build procedure and dependency declarations; vary precision, source patch set, architecture, and build identity explicitly. Validate the actual fisheye adapter with a synthetic solve in the appropriate GPU validation job.

   Extract a common Pixi feature **without `pycolmap`**, then compose it with one provider feature per environment. Keep the separate solve groups and deliberate pins. Separate solves do not require four copied dependency lists.

10. **Move reusable audit logic into canonical modules; fix the demo’s independent parser. — CAREFUL**

   [tools/audit/caspar_polish.py:165](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tools/audit/caspar_polish.py:165) duplicates the mapper’s brief-report parser, including its misleading all-zero fallback. The Galileo and RoboCap baseline tools also duplicate calibrated rig transforms and rig configuration: [Galileo:233](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tools/audit/colmap_rig_baseline_galileo.py:233), [RoboCap:364](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tools/audit/colmap_rig_baseline_robocap.py:364).

   This makes experiments alternate implementations of production contracts. Fixes to gauges, transforms, or statistics must be found and repeated manually.

   The independent COLMAP parser has a concrete defect: [demo_rerun.py:1213](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/demo_rerun.py:1213) removes empty lines before selecting alternating pose lines. An image with an empty observation line shifts the pairing. A two-image in-memory fixture returned only the first image.

   **Restructure:** put solver-summary adaptation, calibrated rig construction, and reusable trajectory operations in focused library modules. Represent unavailable solver statistics explicitly, not as zero cost.

   Keep dataset selection and experiment settings in tools. Preserve distinct SE(3)/Sim(3) evaluation semantics. Extract a lightweight text-model reader for the demo so sharing does not force pycolmap’s dependency stack into its environment.

11. **Separate correctness evidence from timing, availability, and report generation. — SAFE**

   The suite contains valuable behavior tests: exported-model reloads, pose/extrinsic consistency, synthetic recovery, and finite-difference Jacobian checks. Keep those.

   The weak points are specific:

   - [test_raco_backend.py:91](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tests/colsfm/test_raco_backend.py:91) gates the whole module on GPU/model availability, including pure normalization and sizing tests.
   - [test_mapping.py:936](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tests/colsfm/test_mapping.py:936) tests caching through object identity and a 10 ms wall-clock guard. [test_pose_graph.py:666](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tests/colsfm/test_pose_graph.py:666) mixes numerical correctness with a one-second deadline. Retrieval tests add machine-dependent timing limits to ordinary test runs.
   - [test_walkthrough.py:151](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/tests/colsfm/test_walkthrough.py:151) checks entity paths and blueprint presence. Those checks do not prove the Viewer renders the intended result.
   - [colsfm/__init__.py:20](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/__init__.py:20) enables runtime type checking only through `PIXI_DEV_MODE`; the standard test task does not enable it.
   - [bench_cli.py:101](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/bench_cli.py:101) prints failed acceptance checks but exits successfully. Missing candidate timings also become a zero total at [benchmark.py:485](/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/colsfm/benchmark.py:485), which can pass a runtime-ratio bound.

   **Restructure:** run pure contracts without GPU/model gates; select integration and performance workloads explicitly. Test caching by counting expensive calls. Add failure/retry tests for artifact publication and CUDA ownership, an explicit type-checked test task, and a small Viewer screenshot smoke test.

   Preserve missing measurements as missing. Make requested acceptance checks produce an enforceable process result, with an explicit policy for unmeasurable bounds.

   The `bench_cli` / `benchmark` / `bench_report` split is sound. Keep CLI, metric computation, and rendering separate; repair their measurement and status contracts.

Validation performed: all 106 added Python files parsed, Ruff passed, CLI help was inspected, and read-only probes confirmed the metadata, allocation, stage-status, parser, and integer-coercion defects described above. I did not run the full numerical/GPU suite, rebuild packages, or validate Viewer pixels. No files were modified.

**Verdict: REQUEST CHANGES.**

Presumptive blockers are the duplicated pipeline executors, execute-to-discover loop closure, mixed-run artifact lifecycle, unresolved backend/fallback policy, unsafe CUDA ownership and cache identity, conflicting metadata representations, unreproducible fisheye recipe, and the validation gaps that would let those defects pass. The file decomposition should follow those ownership fixes.