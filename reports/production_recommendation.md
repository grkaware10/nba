# Production Model Recommendation

Best holdout performer: random_forest with knn imputation.
- ROC-AUC: 0.9667
- PR-AUC: 0.9818
- Accuracy: 0.7692
- F1: 0.8696
- Validation: time_aware_holdout
- Calibration: sigmoid

## Why This Setup
- KNN imputation won this run because neighboring policy profiles reconstructed missing numeric values more effectively than global summaries.
- KNN imputation can perform well but is slower and less stable at scale for online retraining.
- IQR capping preserves sample size, which is better for recommendation coverage than dropping outlier rows.
- Yeo-Johnson is preferred over Box-Cox because the engineered feature set includes zero-heavy distributions.
- Tree ensembles outperform the logistic baseline by capturing non-linear interactions between offer type, claims, renewal timing, and channel preference.
- Time-aware holdout evaluation reduces leakage from future observations into the training slice.
- Sigmoid calibration makes ranked recommendation scores more reliable for thresholding and downstream channel decisions.

## Why Other Options Were Not Chosen
- Logistic regression remains the interpretability baseline but scored ROC-AUC 0.8667, below the top ensemble.
- A single decision tree is easier to explain but more variance-prone and less stable on noisy, partially imputed training data.
- The closest runner-up was extra_trees with mean imputation, but it trailed on ROC-AUC or PR-AUC.
- HistGradientBoosting remains a strong fallback when latency is tighter than interpretability and feature importance review requirements.

## Production Guidance
- Use the top ensemble for batch or API scoring with periodic recalibration checks on the latest observation window.
- Keep the logistic baseline as a challenger for governance and interpretability review.
- Re-run the benchmark whenever the schema or eligibility policy changes.