# Unmodified upstream reference used only for validation

Repository: `https://github.com/zikai1/ClusterReg`
Commit: `47b6ce98fb53534871da4e4bead739ac7d170091`
Path: `pypointsetreg/fuzzyclusterreg.py`
Git blob SHA-1: `0df3a88bb5d855674f3617fb04f2664289f9c2d6`
Size: 4114 bytes. `scripts/validate.py` verifies the Git blob hash before importing it.

Original implementation: Mingyang Zhao, Jingen Jiang and the ClusterReg contributors.
The repository declares AGPL-3.0. Its README additionally states learning/noncommercial
use and asks commercial users to contact the authors. Those upstream notices are
recorded here, not silently replaced by a permissive license. Consult the upstream
terms before redistribution or commercial integration; this package does not grant
additional rights to upstream material.

The validation harness fixes ONLY `_kmeans`'s returned landmark coordinates to the
C++ landmarks. The original kernel/eigendecomposition and registration function are
then executed unchanged. This removes random-initialization differences without
replacing the function under test. A Python trace hook reads iteration state; it
does not modify the original file or arithmetic. This is a same-landmark numerical
comparison, not a claim of identical outputs for independently randomized runs.

No official C++ runtime or MATLAB runtime is used in the reported comparisons.
