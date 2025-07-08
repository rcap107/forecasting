
## h=1 model design and evaluation

- Use dataframes / skrub to fetch and align time-structured data source to
  build exogeneous features that are available for the forecast horizon of
  choice at the time of prediction.
- Use skrub expressions to be able to do model selection on the pipeline steps:
    - lag variables included or not and lag amount
        - Comment on "system induced lag": time of prediction in deployment
          setting: most recent values might be missing in the system even if
          they show in historical data: in practice this means that we should
          create lag feature with a minimum lag of a few hours to leave time
          for recent measurements to reach the ML prediction system.

        - iceberg / deltalake can explicitly record system lag info in
          historical data.
        - **TODO** refactor to used deferred to get a more interpretable
    - windowing aggregates included or not and window size
    - weather features granularity
    - calendar features
    - holiday feature
    - use a skrub choice tree + builtin random search
- Train a t+1 prediction model and evaluate it:
    - Time-aware cross-validation.
    - MSE/R2, Tweedie deviance ou MAPE.
    - Lorenz curve
    - reliability diagram
    - **TODO** residuals vs predicted
    - Binned residual analysis (**TODO**: refactor common code and add perfect line + always blue) 
    - models:
        - HGBDR
        - **TODO** Exercise: pipeline with missing value support: SimpleImputer with
          indicator, Spline, Nystroem, RidgeCV or TableVectorizer
            - hyper tuning + per analysis of the CV results of the best model.

## Multiple h models

- Train a family of t+h direct models and evaluate them:
    - Plot predictions at different time points.
    - Compute per-horizon metrics + metrics integrated over all horizons of interests
    - Show results as bar plots by h (one for R2, one for MAPE), compute mean
      with (min-max error bars).
- Consider models that are natively multioutput: `RandomForestRegressor` (with
  `min_samples_leaf` to 30 min) or XGBoost multioutput vector leafs.
- Alternatives to a family of t+h direct models:
    - Recursive modeling: show limitations on synthetic data (show with mlforecast, darts or sktime)
    - Use vector output models with concatenated future covariates.
    - Pass h as an extra features and generate expanded datasets for many h values and concatenated future covariates?

## Uncertainty quantification

- Quantify uncertainty in predictions with quantile regressors and evaluate them:
    - Study pinball loss, coverage / width of uncertainty regressors + reliability diagrams + Lorenz curve
- Study if conformal predictions can improve upon this (optional)
    - Show limitation of split conformal predictions:
    - Show CQR.
    - Non-exchangeable conformal prediction.
- Regression as probabilistic classification reduction.
    - https://github.com/ogrisel/notebooks/blob/3a3d2321d4b81d0f089fd13aef96fd27745b505f/quantile_regression_as_classification.ipynb
    - https://github.com/ogrisel/euroscipy-2022-time-series/blob/main/plot_time_series_feature_engineering.ipynb
    - Auto-regressive sampling to sample from the joint future distribution.

## Other ideas

- TabICL or TabPFN on calendar + exogenous features (without lag features).
- Dealing with drifts and trends via multiplicative preprocessing of the target.
- Making models with lagged features robust to random missing values by injecting missing data at training time (possibly by feature blocks).
- Using sample weights to deal with contiguous data quality problems.


## Exercises

- Exercise: pipeline with missing value support: SimpleImputer with
  indicator, Spline, Nystroem, RidgeCV or TableVectorizer
    - hyper tuning + per analysis of the CV results of the best model.

- Adapt the main skrub pipeline to treat weather data as past covariates
  instead of future covariates.

- Exercise: show how to use subsampling.

- Exercise: custom splitter with metadata routing on datetime info: year-based splitting with year passed as feature.
    - clean implementation would require making `SkrubPipeline` implement the `get_metadata_routing` method like the `sklearn.pipeline.Pipeline` does.

- Exercise: Use a `sklearn.ensemble.RandomForestRegressor` to handle multioutput
  horizon forecasts and show that it handles out of the box the multioutput
  problem and thus is faster than using a `sklearn.multioutput.MultiOutputRegressor`.
