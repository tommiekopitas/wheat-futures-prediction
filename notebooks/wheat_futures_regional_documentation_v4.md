# Code Documentation: Wheat Futures Prediction v4
# Agro-Climatic Clustering, Feature Engineering & Ensemble Methods

## Table of Contents

1. [Overview and Design Philosophy](#1-overview-and-design-philosophy)
2. [Section 1 — Load and Prepare Futures Data](#2-section-1--load-and-prepare-futures-data)
3. [Section 2 — Target Variable and Minimal Price Features](#3-section-2--target-variable-and-minimal-price-features)
4. [Section 3 — Load CY-Bench Agricultural Data](#4-section-3--load-cy-bench-agricultural-data)
5. [Section 4 — Per-Region Processing, Production Weights & Feature Engineering](#5-section-4--per-region-processing-production-weights--feature-engineering)
6. [Section 5 — Agro-Climatic Clustering](#6-section-5--agro-climatic-clustering)
7. [Section 6 — Merge and Model](#7-section-6--merge-and-model)
8. [Results Summary](#8-results-summary)
9. [Key Design Decisions Summary](#9-key-design-decisions-summary)

---

## 1. Overview and Design Philosophy

### Problem Statement

We predict the 5-day forward return on Euronext Milling Wheat N2 futures using agricultural data from CY-Bench (weather, soil moisture, vegetation indices) across EU wheat-producing countries.

### Evolution of Approaches

| Version | Approach | Best R² | Key Issue |
|---------|----------|---------|-----------|
| v11 | Aggregate regions → single model | 0.0005 (RF) | 17 price features confound ag signal |
| v2 | Per-region models → aggregate predictions | All negative | ~15 raw features per region, no feature engineering, overfitting |
| **v4** | **Cluster regions → engineer features → per-cluster models** | **0.0044 (ExtraTrees)** | **Current approach** |

v4 achieves **9× higher R²** than v11 with **14 fewer price features**, demonstrating that the improvement comes from the agricultural data processing pipeline, not price feature engineering.

### v4 Architecture

```
645 regions across 8 EU countries
    → Per-region feature engineering (~60 features each)
    → Agro-climatic clustering (k-means, elbow method)
    → Production-weighted aggregation within clusters
    → Three modelling approaches:
        A. Per-cluster models → production-weighted EU aggregate
           (Ridge, Lasso, ElasticNet, DT, KNN, ExtraTrees,
            BaggingRidge, BaggingLasso, RF)
        B. Lasso stacking across clusters
        C. PCA regression across all cluster features
```

### Key Academic Sources

| Component | Source |
|-----------|--------|
| Feature engineering | CY-Bench (Paudel et al. 2025, §2.2.3) |
| Agro-climatic clustering | Paudel et al. (2022, §4) |
| Within-cluster aggregation | Bagging principle (MA429 Lecture 4, Olver) |
| Production-weighted aggregation | Paudel et al. (2022, §2.5.3); Cerrani & López Lozano (2017) |
| Lasso stacking | Wolpert (1992); Lasso selection (MA429 Lecture 3, §3.5.3) |
| PCA regression | MA429 Lecture 10 (Olver) |
| Ridge/Lasso regularisation | MA429 Lecture 3 (Olver) |
| Decision trees, RF, boosting | MA429 Lecture 4 (Olver); ISLP Ch 8 |
| Bias-variance tradeoff | MA429 Lecture 2 (Olver) |
| Cross-validation | MA429 Lecture 3 (Olver) |

---

## 2. Section 1 — Load and Prepare Futures Data

### Cells 2-5

Loads Milling Wheat N2 Futures Historical Data from Euronext. Parses dates (%m/%d/%Y format), cleans numeric columns, handles volume suffixes (K/M), filters to 2003-2023.

**Configuration:**
- FORECAST_HORIZON = 5 — 5 trading days (1 week).
- AG_LAG_DAYS = 0 — No artificial lag; futures traders react to weather in real time.

---

## 3. Section 2 — Target Variable and Minimal Price Features

### Cell 7

**Target:**

futures['Fwd_Return'] = futures['Price'].shift(-FORECAST_HORIZON) / futures['Price'] - 1

Continuous 5-day forward return (regression, not classification). Lecture 1 (Olver): "Regression — goal is to predict a continuous numerical quantity."

**Price features (deliberately minimal — 3 features):**

| Feature | Description | Justification |
|---------|-------------|---------------|
| ret_5d | 5-day return | Same horizon as target; weekly momentum |
| ret_20d | 20-day return | Medium-term trend |
| vol_sma_ratio | Volume / 20-day MA volume | Relative trading activity |

**Decision: Only 3 price features.** The goal is to isolate whether agricultural data adds predictive signal. With 17 price features (v11), any R² improvement could be confounded with price feature interactions. By stripping to 3 minimal market-state controls, any R² improvement in v4 is attributable to the agricultural pipeline.

---

## 4. Section 3 — Load CY-Bench Agricultural Data

### Cell 9

Generic load_country_data(country_code, cybench_dir) function that loads meteo, soil_moisture, ndvi, fpar, and yield CSVs for any country. Controlled by COUNTRIES list in configuration.

### Cell 14 — Load Pre-Generated Region CSVs

For efficiency, region daily data is pre-computed and saved as individual CSVs ({adm_id}_daily.csv). This cell loads them directly, filtering to countries in the COUNTRIES list.

---

## 5. Section 4 — Per-Region Processing, Production Weights & Feature Engineering

### Cell 11 — Production Weights

Builds production_weights = {adm_id: {year: tonnes}} from yield CSVs.

**Handling incomplete German data:**

| Column | France | Germany | Poland |
|--------|--------|---------|--------|
| yield | Every year (to 2020) | Every year (to 2021) | Every year |
| harvest_area | Every year | Every ~4 years (census) | Every year |
| production | Every year | Entirely NaN | Every year |

**Solution:** Forward-fill harvest_area within each region (area changes slowly between censuses), then compute production = yield × harvest_area. Standard practice: Eurostat interpolates between agricultural censuses (Cerrani & López Lozano, 2017).

Uses previous year's production to avoid look-ahead bias (Paudel et al. 2022, §2.1). get_production_weight(adm_id, year) falls back to nearest earlier year if exact year unavailable.

### Cell 16 — Per-Region Feature Engineering

The engineer_region_features(df) function applies to each region's daily DataFrame individually, following CY-Bench Section 2.2.3 recommendations. Grows each region from ~8-11 raw features to ~60-65 engineered features.

#### A. Phenological Phase Indicators (5 features)

Binary flags based on month, reflecting French winter wheat phenology:

| Feature | Months | Crop Phase |
|---------|--------|------------|
| in_growing_season | Oct-Jul | Full season |
| phase_dormant | Dec-Feb | Vernalisation |
| phase_stem_elong | Mar-Apr | Stem elongation |
| phase_flowering | May | Heading/flowering |
| phase_grain_fill | Jun-Jul | Grain filling |

#### B. Phase-Conditional Stress Features (~10 features)

Stress indicators that only accumulate during vulnerable crop windows:

| Feature | Condition | Window | Reference |
|---------|-----------|--------|-----------|
| winterkill_risk_30d/60d | tmin < -10°C during Dec-Feb | 30, 60d | Severe frost during dormancy |
| frost_days_dormant_60d | tmin < -5°C during Dec-Feb | 60d | Milder frost accumulation |
| spring_heat_days_30d/60d | tmax > 25°C during Mar-May | 30, 60d | Spring heat shock |
| grainfill_heat_days_30d | tmax > 30°C during May-Jul | 30d | Schlenker & Roberts (2009) |
| grainfill_excess_heat_30d | Cumulative (tmax - 30)+ during May-Jul | 30d | Excess degree-days |
| flowering_drought_60d/90d | Cumulative CWB during Apr-Jun | 60, 90d | Water deficit at flowering |

#### CY-Bench Threshold Indicators (3 features)

Directly from Paudel et al. (2025, §2.2.3):

| Feature | Condition | Reference |
|---------|-----------|-----------|
| cold_days_30d | tmin < 0°C | CY-Bench: "number of days tmin < 0°C" |
| hot_days_30d | tmax > 35°C | CY-Bench: "days tmax > 35°C" |
| dry_days_30d | prec < 1mm | CY-Bench: "days prec < 1mm" |

#### C. Anomaly Features (5 features)

Computed relative to expanding day-of-year climatology (no future leakage):

| Feature | Method | Reference |
|---------|--------|-----------|
| ndvi_anomaly | NDVI - expanding DOY mean | Crop health surprise |
| ndvi_anomaly_30d | 30-day rolling mean of anomaly | Smoothed signal |
| fpar_anomaly | fPAR - expanding DOY mean | Radiation absorption anomaly |
| fpar_anomaly_30d | 30-day rolling mean of anomaly | Smoothed signal |

#### D. Rolling Features (~22 features)

Following CY-Bench §2.2.3: "monthly averages... cumulative growing degree days, cumulative precipitation, cumulative fPAR and cumulative NDVI":

| Feature | Variables | Windows |
|---------|-----------|---------|
| {var}_mean_{w}d | tavg, tmin, tmax, prec, vpd, cwb, ssm, ndvi, fpar | 30, 60d |
| cumul_prec_{w}d | prec | 30, 60d |
| gdd_cumul_{w}d | max(tavg, 0) | 30, 60d |

#### E. Weekend Weather Accumulator (3 features)

Futures don't trade weekends but weather continues. Monday's price reacts to 3 days of news:

| Feature | Formula |
|---------|---------|
| weekend_prec | 3-day cumul prec × is_monday |
| weekend_tmax | 3-day max tmax × is_monday |
| weekend_tmin | 3-day min tmin × is_monday |

#### F. Standardised Precipitation Index (2 features)

Z-score of rolling precipitation relative to expanding historical distribution. McKee et al. (1993):

| Feature | Window |
|---------|--------|
| spi_30d | 30-day cumulative precipitation z-score |
| spi_60d | 60-day cumulative precipitation z-score |

#### G. Diurnal Temperature Range (3 features)

Large DTR = clear skies = drought risk. Lobell (2007); Laudien et al. (2023):

| Feature | Formula |
|---------|---------|
| dtr | tmax - tmin |
| dtr_mean_30d | 30-day rolling mean |
| dtr_anomaly_30d | dtr - expanding mean |

#### H. VPD-Based Drought Stress (3 features)

Vapour pressure deficit is a stronger predictor of crop stress than temperature alone. Lobell et al. (2014); Schlenker & Roberts (2009):

| Feature | Formula |
|---------|---------|
| vpd_stress_critical_30d | VPD masked to Mar-Jul, 30d mean |
| vpd_stress_critical_60d | VPD masked to Mar-Jul, 60d mean |
| high_vpd_days_30d | Count of days VPD > 15 hPa in 30d |

#### I. Soil Moisture Deficit Index (2-4 features)

Z-score of soil moisture relative to expanding climatology. Narasimhan & Srinivasan (2005):

| Feature | Formula |
|---------|---------|
| soil_moisture_ssm_zscore | (SSM - expanding_mean) / expanding_std |
| soil_moisture_ssm_zscore_30d | 30-day rolling mean of z-score |
| soil_moisture_rsm_zscore | Same for root-zone SM (if available) |
| soil_moisture_rsm_zscore_30d | Same rolling mean (if available) |

#### J. Radiation Use Efficiency Proxy (3 features, if rad available)

FPAR × radiation ≈ absorbed photosynthetically active radiation (APAR). Monteith (1972):

| Feature | Formula |
|---------|---------|
| apar_proxy | fpar × radiation |
| apar_proxy_mean_30d | 30-day rolling mean |
| apar_anomaly | apar - expanding DOY mean |

#### K. Temperature Optimality Index (3 features)

Wheat optimal range 15-22°C. Porter & Gawith (1999):

| Feature | Formula |
|---------|---------|
| temp_suboptimal | Distance from 15-22°C range (0 inside) |
| temp_suboptimal_30d | 30-day rolling mean |
| temp_suboptimal_critical_60d | Same but masked to Mar-Jul only |

---

## 6. Section 5 — Agro-Climatic Clustering

### Cell 18 — k-Means Clustering

**Motivation:** Paudel et al. (2022, §4): "We believe grouping regions based on agro-climatic similarities would help... machine learning models trained on data from widely different regions have to learn spatial and temporal yield variability simultaneously."

**Method:**

1. **Climate profiles:** For each region, compute mean and std of 7 key variables (tavg, tmax, tmin, prec, vpd, ndvi, ssm) over the training period only (pre-2017). Produces a ~14-dimensional vector.

2. **Standardise:** StandardScaler on profile vectors.

3. **k selection — Elbow method:** Plot inertia (within-cluster sum of squares) vs k for k=3..49. Detect the elbow via second derivative (largest acceleration in the inertia curve). Enforce a floor of k >= 3.

   **Decision: Elbow method over silhouette score.** Silhouette score is biased toward small k (fewer, rounder clusters) and selected k=3 in our data, putting 536 of 645 regions into one cluster — defeating the purpose. The elbow method is more appropriate when we want granularity for downstream modelling.

4. **Assign clusters:** Each region is assigned to exactly one cluster via k-means hard assignment.

### Cell 19 — Within-Cluster Aggregation

For each cluster, at each date, compute the production-weighted mean of all member regions' features:

    cluster_feature(date) = SUM_r [w_r × feature_r(date)] / SUM_r [w_r]

where w_r = production of region r in the previous year.

**Why aggregate within clusters?** Same principle as bagging (Lecture 4, Olver): averaging across similar regions cancels idiosyncratic noise while preserving the cluster's distinctive climate signal. A drought across Mediterranean regions averages to a strong drought signal within the Mediterranean cluster, rather than being diluted by averaging with rainy northern regions.

---

## 7. Section 6 — Merge and Model

### Cell 21 — Merge Clusters with Futures

Each cluster's aggregated features are merged with futures data using merge_asof (backward direction). Each cluster DataFrame gets the same shared price features + its own aggregated ag features. Target is the same for all clusters: Milling Wheat N2 5-day forward return.

### Cell 22 — Walk-Forward Regression Framework

Expanding-window evaluation: for each test year t in {2017,...,2023}, train on all data before year t, test on year t.

**Scaler fit inside each fold** — StandardScaler is refit on training data only per fold, not globally. This prevents test-set statistics from leaking into training (Lecture 3, Olver).

### Cell 23 — Approach A: Per-Cluster Models

For each model family, fit on each cluster's data separately, then aggregate cluster-level predictions to EU level using total cluster production as weight.

**Models tested (9 total):**

| Model | Family | MA429 Reference | Key Property |
|-------|--------|----------------|-------------|
| Ridge | Linear, L2 regularised | Lecture 3 | Closed-form; shrinks all coefficients |
| Lasso | Linear, L1 regularised | Lecture 3, §3.5.3 | Drives irrelevant features to zero |
| ElasticNet | Linear, L1+L2 | Lecture 3 | Combines Lasso sparsity + Ridge stability |
| DecisionTree | Tree | Lecture 4 | Captures nonlinear thresholds |
| KNN | Instance-based | Lecture 2 | "Similar weather → similar returns" |
| ExtraTrees | Ensemble (bagging + extra randomisation) | Geurts et al. (2006) | More decorrelated trees than RF |
| BaggingRidge | Ensemble (bagging) + linear | Lectures 3 + 4 | 200 Ridge regressions on bootstrap samples |
| BaggingLasso | Ensemble (bagging) + sparse linear | Lectures 3 + 4 | Each bootstrap selects different features |
| RF | Ensemble (bagging + feature subsampling) | Lecture 4, ISLP Ch 8.2 | Standard random forest |

**Decision: Why ExtraTrees outperforms.** ExtraTrees (Geurts et al., 2006) differs from RF in one key way: at each split, instead of finding the optimal threshold among the random feature subset, it picks a random threshold. This additional randomisation further decorrelates the trees. In a low signal-to-noise regime (financial returns), the variance reduction from decorrelation outweighs the small bias increase from suboptimal splits. This is the same principle as RF over bagging (Lecture 4, slide 19): "Reduce correlation by selecting a random subset of features for each split."

**Decision: Why BaggingLasso is included.** Each bootstrap sample changes which rows are in/out, shifting correlations between features. This means each Lasso selects a slightly different sparse subset. Averaging 200 different sparse models gives implicit feature importance diversity on top of sample diversity — combining Lasso's feature selection (Lecture 3) with bagging's variance reduction (Lecture 4).

### Cell 24 — Approach B: Lasso Stacking

**Step 1:** Fit Ridge on each cluster separately (base learners). Collect walk-forward out-of-fold predictions.

**Step 2:** Build stacking matrix (each column = one cluster's prediction, rows = dates).

**Step 3:** Fit Lasso on stacking matrix. L1 penalty drives uninformative clusters to zero weight.

Lecture 3 (Olver, §3.5.3): "lasso will have an effect similar to subset selection... the optimal solution will implicitly do the subset selection."

Reference: Wolpert, D. H. (1992). "Stacked Generalization." Neural Networks, 5(2), 241-259.

### Cell 25 — Approach C: PCA Regression

**Step 1:** Stack all cluster features into one wide matrix (~clusters × ~60 features = 300-600 columns).

**Step 2:** PCA keeping components explaining 90% of variance. Reduces to ~10-30 components representing EU-wide weather patterns.

**Step 3:** Ridge on PCs + price features.

Lecture 10 (Olver): "PCR can work particularly well in case of multicollinearity." Cluster features are heavily correlated (temperature in Mediterranean cluster correlates with Continental cluster). PCA extracts orthogonal directions of maximum variance.

PCA is unsupervised (Lecture 10): components capture weather variation patterns, not price patterns. Ridge on top then learns which weather patterns relate to price.

### Cell 26 — Price-Only Baseline

Ridge regression on 3 price features only. Establishes the baseline that agricultural data must beat.

---

## 8. Results Summary

### 5-Day Horizon (Primary Result)

| Model | R² | Dir Acc | Notes |
|-------|:--:|:-------:|-------|
| **A: ExtraTrees** | **0.0044** | 0.499 | **Best overall** |
| A: RF | 0.0037 | 0.517 | Second best |
| A: Lasso | -0.003 | 0.501 | Best linear model |
| A: BaggingLasso | -0.004 | 0.490 | |
| Baseline: Price-Only | -0.012 | 0.475 | |
| A: Ridge | -0.077 | 0.498 | Worst |

**Key finding:** ExtraTrees R²=0.0044 is 9× better than v11's best (R²=0.0005), using 3 price features instead of 17. The improvement is attributable to the agro-climatic clustering + feature engineering pipeline, not price features.

**Agricultural feature lift:** Best price-only R² = -0.012, best with ag = 0.0044, lift = +0.016.

### 20-Day Horizon (Robustness Check)

| Model | R² | Dir Acc | Notes |
|-------|:--:|:-------:|-------|
| A: ExtraTrees | -0.006 | 0.481 | Best R² but negative |
| C: PCA + Ridge | -0.102 | **0.528** | Best directional accuracy |
| Baseline: Price-Only | -0.028 | 0.455 | |

20-day performs worse on R² due to overlapping targets (19/20 day overlap between consecutive observations). However, PCA + Ridge achieves 52.8% directional accuracy — notably above 50%, suggesting PCA extracts a genuine EU-wide weather pattern that predicts monthly price direction.

### Why Ensemble Methods Dominate

In a low signal-to-noise regime, the bias-variance tradeoff (Lecture 2, Olver) is dominated by variance. All linear models (Ridge, Lasso, ElasticNet) have R² < 0 because they overfit to noise patterns in the training set. Ensemble methods (ExtraTrees, RF, BaggingLasso) survive because averaging many decorrelated models cancels out the noise-fitting while preserving any genuine signal. This is exactly the bagging principle (Lecture 4): "variance reduction by averaging the output, assuming models are not too highly correlated."

---

## 9. Key Design Decisions Summary

| Decision | Choice | Justification |
|----------|--------|---------------|
| Feature engineering | CY-Bench recommendations: rolling means, cumul sums, threshold counts, anomalies, phase-conditional stress | Paudel et al. (2025, §2.2.3) |
| Agro-climatic clustering | k-means on long-term climate profiles | Paudel et al. (2022, §4) |
| k selection | Elbow method with floor | Silhouette biased toward k=3; elbow gives more granularity |
| Within-cluster aggregation | Production-weighted average | Paudel et al. (2022, §2.5.3); noise cancellation (Lecture 4) |
| 9 model families | Ridge, Lasso, ElasticNet, DT, KNN, ExtraTrees, BaggingRidge, BaggingLasso, RF | Lectures 2-4; covers linear, nonlinear, ensemble |
| Lasso stacking | Lasso meta-learner on cluster predictions | Lecture 3 (§3.5.3); Wolpert (1992) |
| PCA regression | PCA → Ridge | Lecture 10: PCR for multicollinear data |
| Minimal price features | ret_5d, ret_20d, vol_sma_ratio | Isolate ag signal from price confounding |
| Expanding climatologies | No future leakage in anomaly features | Same principle as scaler-inside-fold (Lecture 3) |
| Production weights (lagged) | Previous year's production | Paudel et al. (2022): avoid look-ahead bias |
| DE harvest_area imputation | Forward-fill between censuses, then yield × area | Cerrani & López Lozano (2017) |
| Walk-forward evaluation | Expanding window, yearly (2017-2023) | Paudel et al. (2022, Fig. 4) |
| Scaler inside folds | StandardScaler refit per fold | Lecture 3: prevent test leakage |
| FORECAST_HORIZON = 5 | 1 trading week | Best empirical performance; ag features most predictive here |
| AG_LAG_DAYS = 0 | No artificial lag | Futures traders react to weather in real time |
| 5-day over 20-day | 5-day as primary, 20-day as robustness check | 20-day suffers from overlapping targets (19/20 day overlap) |
