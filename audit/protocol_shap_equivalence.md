# E4 implementation equivalence check (saved before computation)

Compare the same fixed function under the same empirical background and feature permutations. This is a numerical check of linearity and dummy properties, not an attribution performance contest.

1. Enumerate every coalition independently for f(x)=2x0+x0*x1 (x2 unused), dimension 3. Compare exact Shapley weights to TESV with all 3! permutations on a fixed background. Verify the unused coordinate has zero attribution and efficiency telescopes.
2. Reuse E2 seed-42 T-learners for IHDP and Hillstrom. Ten test rows (the first ten of the saved random 50); B=50 train background rows, M=20 permutations per row, independently drawn with seed 20260928. Save background row indices and permutation matrices.
3. Compare TESV, an independent implementation of direct marginal-replacement Shapley of predict_cate, and a fully paired difference of Shapley attributions of the existing treatment/control regressors. Do not refit outcome models or change the object being explained.
4. Report maximum absolute and scaled relative differences and efficiency residuals. Check absolute tolerance 1e-10 plus relative tolerance 1e-8; scaled relative denominator max(1,|reference attribution|). No statistical significance test. A mismatch triggers implementation investigation rather than a change of random seed.
