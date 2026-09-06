# cuSFM protobuf schema

`cusfm_protos.fdset` is a `FileDescriptorSet` holding the 19 `.proto` files that
cuSFM's prebuilt binaries embed in their `protodesc_cold` ELF section. It is the
contract that makes writing cuSFM keyframes from Python legitimate rather than
guesswork: `tools/raco_extract.py` builds real message classes from it, so the
keyframes it writes are serialised by protobuf itself.

Verified by round trip: parsing a keyframe produced by cuSFM's own
`feature_extractor_main` and re-serialising it reproduces the file **byte for
byte** (1,083,342 bytes, sha256 prefix `9254cf795656d92e`).

It is vendored because it is 19 KB of schema and the alternative -- extracting it
from the binaries on every machine -- made the extractor and its contract tests
fail anywhere but the machine that first produced it.

Regenerate with `pixi run -e raco extract-cusfm-schema`, which reads the
descriptors back out of `pycusfm/x86_cuda13/bin/`. Upstream is frozen, so this
should only ever be needed after a `pycusfm` version bump -- and if the
regenerated set differs, the keyframe layout changed and
`tools/raco_extract.py` needs review.
