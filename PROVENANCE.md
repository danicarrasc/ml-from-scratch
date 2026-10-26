# Provenance and changes

Implementations completed by Daniel Carrasco Jurado in Machine Learning coursework at ICAI. Templates, interfaces and original tests come from ICAI-IMAT-ML teaching materials. Source repositories may require organization access.

| Published module | Original source |
|---|---|
| `linear_regression.py` | `p2-3-linear-regression-danicarrasc/src/lab_2_3_LinearRegression.py` |
| `gradient_descent.py` | `p2-4-linear-regression-ii-danicarrasc/src/Lab_2_4_LR2.py` |
| `logistic_regression.py` | `p2-5-logistic-regression-danicarrasc/src/Lab2_5_LogisticRegression_and_regularization.py` |
| `cross_validation.py` | `p2-6-cross-validation-danicarrasc/src/Lab2_6_CV.py` |
| `knn.py` | `p2-2-knn-danicarrasc/src/Lab_2_2_kNN.py` |
| `lda.py` | `p2-7-lda-danicarrasc/src/LDA_model.py` |

The first four modules were verified byte-for-byte against the original files. Their algorithms remain unchanged. Original tests are included with adapted imports; the Anscombe test has a `network` marker.

## Targeted corrections to kNN

- Vote over training labels instead of neighbor row indices.
- Count frequencies against sorted class labels, supporting strings and nonconsecutive integers.
- Reject `k` larger than the training sample count; use stable sorting for equal distances.
- Remove an unused commented probability implementation.
- In the ROC helper, use positive label 1 after mapping the original labels to binary values.

## Targeted corrections to LDA

- Index projected class means by class position, not by the label value.
- Shift distance scores before exponentiation to avoid all-zero softmax denominators for distant queries.
- Validate component counts and recompute the default dimension on each fit.
- Clarify that normalized distance scores are not calibrated LDA posteriors, and correct the misleading symmetry comment.

The original scatter-matrix construction, eigenproblem and nearest-centroid prediction design are preserved. No claim of equivalence to the probabilistic LDA classifier is made.

## Added portfolio material

English documentation, a synthetic-data demo, environment configuration, CI and additional reference/regression tests were prepared with AI assistance. Original coursework folders were left unchanged. The classroom notebooks were not republished.

## Academic source repositories

- https://github.com/ICAI-IMAT-ML/p2-2-knn-danicarrasc
- https://github.com/ICAI-IMAT-ML/p2-3-linear-regression-danicarrasc
- https://github.com/ICAI-IMAT-ML/p2-4-linear-regression-ii-danicarrasc
- https://github.com/ICAI-IMAT-ML/p2-5-logistic-regression-danicarrasc
- https://github.com/ICAI-IMAT-ML/p2-6-cross-validation-danicarrasc
- https://github.com/ICAI-IMAT-ML/p2-7-lda-danicarrasc
