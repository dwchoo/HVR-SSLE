# SSIM Similarity Matrices (Train ↔ Test)

This folder contains SSIM-based similarity data between **train** and **test** images drawn from the LOL-v1 and LOL-v2 datasets.  
The similarity scores were computed with SSIM and then **thresholded at 0.9**: any SSIM < 0.9 was set to **0.0** to make the heatmap and downstream inspection clearer.

In the paper’s description of the sparse test–train similarity heatmap (see Eqs. 24–25), only pairs with **SSIM(testᵢ, trainⱼ) ≥ 0.90** are visualized to highlight near-identical structures. The resulting strong diagonal and off-diagonal clusters (red blocks) indicate substantial scene duplication within LOL-v1 and significant scene overlap between the LOL-v1 and LOL-v2 benchmarks. This implies that many test scenes are also present in training data, revealing notable train–test overlap.

## Summary (from the files in this folder)

- **Matrix shape:** 215 test images x 2074 train images
- **Test split composition:** LOL-v2 (200), LOLv1 (15)
- **Train split composition:** LOL-v2 (1589), LOLv1 (485)
- **Thresholded matches (SSIM >= 0.9):** 279 pairs total; 106/215 test images have at least one match
- **Non-zero SSIM range in matched matrix:** 0.928643 to 1.0
- **Match breakdown (test -> train, SSIM >= 0.9):** LOL-v2->LOLv1 219; LOLv1->LOL-v2 36; LOLv1->LOLv1 24; LOL-v2->LOL-v2 0

## Files

### `similarity_matrix_ssim_test_to_train.csv`
Full SSIM similarity matrix (before thresholding).
- **Rows (y-axis):** test images (first column is the row label)
- **Columns (x-axis):** train images (header row)
- **Cell values:** SSIM score between the test image and the train image

### `similarity_matrix_ssim_test_to_train_matched.csv`
Thresholded SSIM matrix used for the heatmap.
- Same shape and layout as the full matrix
- **SSIM < 0.9 → 0.0**

### `test_to_train_matches_long.csv`
Long (tidy) format derived from the thresholded matrix.
- **Columns:**
  - `test`: test image path
  - `train`: train image path
  - `score`: SSIM score (only scores > 0.0 kept)
  - `rank`: rank of the match within each test image (highest score = 1)
- Useful for filtering, sorting, or joining.

### `test_to_train_matches_grouped.csv`
Grouped (list) format derived from the thresholded matrix.
- **Columns:**
  - `test`: test image path
  - `train_list`: `;`-separated list of matching train images
  - `score_list`: `;`-separated list of SSIM scores aligned with `train_list`
- Convenient for quick per-test inspection.

### `train-test heatmap.pdf`
Heatmap visualization of the thresholded matrix (`similarity_matrix_ssim_test_to_train_matched.csv`).

## Notes
- All SSIM values shown in the derived files come from the **thresholded** matrix (`SSIM >= 0.9`).
- Paths are stored with dataset prefixes, e.g. `train|LOLv1/...` or `test|LOLv1/...`, to preserve source information.
- This analysis is best described as **train ↔ test** similarity across LOL-v1 and LOL-v2, rather than LOL-v1 ↔ LOL-v2 alone.
