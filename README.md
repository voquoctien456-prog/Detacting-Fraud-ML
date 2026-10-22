# Detecting Abnormal Trading Sessions ("Pump" Activity) in Vietnamese Bank Stocks

Course project for Applied Algorithms in Finance and Business. The goal is a screening tool
that flags trading sessions on Vietnamese bank stocks that look like they might involve
coordinated price manipulation, to help narrow down what a market surveillance team should
actually look at.

## 1. Why this problem

Manipulated sessions hurt the market in a few concrete ways: retail investors buy in at an
artificially pushed-up price and lose money when it corrects, price stops reflecting real
information which makes capital allocation worse across the market, and regulators have to
watch thousands of sessions a day with a small team. This project is meant as a screening
tool, not a final verdict - it ranks sessions by how suspicious they look so a human can
decide what to check first.

## 2. Data

| | |
|---|---|
| Source | Yahoo Finance (`yfinance`) |
| Coverage | 15 listed Vietnamese banks: TCB, STB, MBB, ACB, VPB, SHB, EIB, LPB, VIB, CTG, HDB, VCB, BID, SSB, NAB |
| Period | 01/02/2021 to today (pulls up to the latest available session automatically) |
| Sessions | 15,914 |
| Anomalous sessions | 363 (2.28%) |

### Label (`Is_Anomaly`)
```
Is_Anomaly = 1  if  Spread > 2 x Price_Volat (60-session rolling std)  AND  V_Surge > 2.5
```
- `Spread` = (High - Low) / Low for that session
- `Price_Volat` = rolling std of `P_Change` over the previous 60 sessions (shifted by 1 day, no look-ahead)
- `V_Surge` = session volume divided by the trailing 20-session average volume (also shifted)

### Predictive features (X) - kept separate from the labeling formula
| Feature | What it captures |
|---|---|
| `P_Change` | daily % change in closing price |
| `Rel_P_Change` | `P_Change` minus the same-day average across all 15 banks (removes sector-wide moves) |
| `Vol_Change` | daily % change in volume |
| `Vol_Change_MA5` | 5-session rolling average of `Vol_Change` (catches gradual accumulation, not just single-day spikes) |

`Spread` and `V_Surge` are used only to build the label. They are **not** in X - if they were,
the model would just be re-learning the labeling formula instead of predicting anything new.
That was actually a bug in an earlier version of this project (see Section 4).

## 3. Notebook structure (`ML_final_submission.ipynb`)

| Block | What it does | Main output |
|---|---|---|
| 1 | Pull data from Yahoo Finance, build causal (shifted) features, apply the labeling rule | `Stock_Real_Anomaly_Data.xlsx` |
| 2-3 | Time-based 80/20 split per bank, train and compare 7 algorithms | `Models_Performance_Comparison.xlsx` |
| 4 | EDA dashboard (F1 ranking, correlation matrix, feature importance, labeling scatter) | `Final_Dashboard_EDA.png` |
| 5 | GridSearchCV for XGBoost (SMOTE inside the pipeline, `TimeSeriesSplit`), threshold tuning, SHAP, Isolation Forest cross-check | `SHAP_Explainable_AI.png`, `Confusion_Matrix_Tuned.png` |
| 6 | Walk-forward validation by quarter, ablation test for `Rel_P_Change`, top-5 case study | `Walk_Forward_Validation.xlsx/.png`, `Top5_Flagged_Sessions.xlsx` |

Run Block 1 through 6 in order, in the same kernel - later blocks reuse variables from earlier
ones (`df_clean`, `X_train`, `test_df`, `y_prob_test`, `FEATURE_COLS`, etc.), so don't restart
the kernel halfway through.

## 4. Bugs found and fixed along the way

| # | Problem (first version) | Effect | Fix |
|---|---|---|---|
| 1 | `Spread`/`V_Surge` used both to build the label and as model inputs | Model just re-learned the labeling rule, ROC-AUC ~99.9% but meaningless | Removed both from X, kept them only for building Y |
| 2 | Rolling stats computed over the full history with no shift | Future information leaked into the labeling threshold | All rolling stats now use `.shift(1)` first |
| 3 | Random `train_test_split` on time series data | Adjacent, highly correlated sessions ended up split across train and test | 80/20 split by date, done separately per bank |
| 4 | SMOTE run before `GridSearchCV` | Synthetic samples leaked across CV folds, inflating CV scores | SMOTE placed inside an `imblearn.Pipeline`, run separately per fold |
| 5 | Default shuffled `cv=3` on time series data | Folds didn't respect chronological order | Switched to `TimeSeriesSplit(n_splits=3)` |
| 6 | Fixed 0.5 classification threshold | Not tuned for a heavily imbalanced problem | Threshold chosen from the precision-recall curve to maximize F1 |
| 7 | "Complete quarter" filter used a fixed row-count cutoff (`>=55`) | The still-open current quarter slipped through and added noise to the last point in the walk-forward chart | Now excludes the current calendar quarter directly, via `pd.Timestamp.today().to_period('Q')` |

## 5. Main results (actual run)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 71.43% | 25.86% | **37.97%** | 94.41% | 44.21% |
| **XGBoost** | 22.78% | 70.69% | 34.45% | **95.68%** | **46.01%** |
| Decision Tree | 13.55% | 86.21% | 23.42% | 92.69% | 33.17% |
| Logistic Regression | 12.10% | 72.41% | 20.74% | 81.35% | 30.15% |
| SVM | 10.06% | 86.21% | 18.02% | 91.71% | 30.47% |
| KNN | 25.00% | 3.45% | 6.06% | 72.04% | 11.29% |
| Naive Bayes | 1.82% | 100.00% | 3.57% | 88.79% | 14.40% |

**SHAP:** `Vol_Change` is by far the most important feature (roughly 3x the others) - unusual
liquidity, not the size of the price move itself, looks like the earliest signal of manipulation.

**Walk-forward (18 complete quarters, 2022Q1-2026Q2):** F1 ranges from 18.5% to 52%, PR-AUC from
14.6% to 69.2%. Performance is not fully stable over time - this would need ongoing monitoring
and retraining in any real deployment.

**Worth noting:** the LPB session on 13/08/2025 got a 99.57% model probability but was NOT
flagged by the original rule (its `V_Surge` was 2.11, just under the 2.5 cutoff). That's a
concrete example of the model catching something the fixed rule missed.

## 6. Known limitations

- The ground-truth label is a hand-written rule, not a confirmed list of manipulation cases
  from regulators.
- Only price and volume features are used - no order-flow, foreign-ownership, or account-level
  data, which would likely catch more sophisticated manipulation patterns.
- Absolute performance (F1 around 34-38%) is reasonable for a screening tool but not accurate
  enough to fully automate any enforcement decision.
- Performance swings noticeably across quarters, so this is closer to a proof of concept than
  something ready to deploy as-is.

## 7. Output files

```
Stock_Real_Anomaly_Data.xlsx        # processed dataset, X + Y
Models_Performance_Comparison.xlsx  # 7-model comparison table
Final_Dashboard_EDA.png             # 4-panel EDA dashboard
SHAP_Explainable_AI.png             # SHAP summary plot
Confusion_Matrix_Tuned.png          # confusion matrix at the tuned threshold
Walk_Forward_Validation.xlsx/.png   # quarterly walk-forward results
Top5_Flagged_Sessions.xlsx          # 5 highest-probability sessions (case study)
```

## 8. How to run it

1. Needs internet access for `yfinance` in Block 1.
2. Install: `pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn openpyxl yfinance`
3. Run Block 1 through 6 in order, same kernel.
4. Output files are written to the current working directory.
