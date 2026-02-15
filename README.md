# M5 Forecasting: Retail Demand Prediction

An end-to-end pipeline for the Kaggle M5 Forecasting competition, predicting 28-day unit sales for Walmart across hierarchical levels (SKU, store, state).

The solution prioritizes memory efficiency and training throughput, utilizing **Polars** for vectorized feature engineering and **LightGBM** for gradient boosting on sparse data.

### Performance
- **Private Score:** 0.54547
- **Public Score:** 0.63360

---

## Architecture & Design Decisions

### 1. Data Pipeline (Polars vs. Pandas)
Given the dataset size (~60M rows after unpivoting), standard Pandas workflows hit OOM bottlenecks during lag generation.
- Migrated the ETL pipeline to **Polars** to leverage the Lazy API and parallel execution.
- Enforced strict type downcasting (`Int16`, `Float32`, `Categorical`) to reduce memory footprint by ~70% without information loss.
- Implemented a chunked training strategy (processing data by store/week) to maintain a low memory ceiling during the feature engineering phase.

### 2. Modeling Strategy
Retail data is inherently intermittent. Standard RMSE minimization biases predictions toward zero for slow-moving items.
- **Objective Function:** Used **Tweedie loss** (`variance_power=1.1`) to model the compound Poisson-Gamma distribution, effectively handling the zero-inflated target variable.
- **Granularity:** Trained 10 independent models (one per store). This offers a better trade-off than a single global model (which generalizes too broadly) or item-level models (which lack sufficient signal).
- **Validation:** Utilized a time-series split (rolling origin) matching the competition's 28-day horizon to prevent leakage.

### 3. Feature Engineering
Spectral analysis of the sales history indicated strong weekly seasonality over annual trends.
- **Seasonality:** Constructed modulo-7 lags (7, 14, 28 days) to capture the weekly heartbeat.
- **Trends:** Rolling means and standard deviations (windows: 7, 30, 60, 140) with a 7-day shift to ensure causal consistency.
- **Context:** Integrated price momentum (current vs. historical avg) and SNAP (food stamp) event flags, which significantly impact variance in specific states (CA/TX/WI).
