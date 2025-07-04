- Use dataframes / skrub to fetch and align time-structured data source to
  build exogeneous features that are available for the forecast horizon of
  choice at the time of prediction.
- Use skrub expressions to be able to do model selection on the pipeline steps:
    - lag variables included or not and lag amount
    - windowing aggregates included or not and window size
    - weather features granularity
    - calendar features
    - holiday features
    - use a skrub choice tree + builtin random search
    - optuna without skrub but cannot do nested cross-validation.
- Train a t+1 prediction model and evaluate it:
    - MSE/R2, Tweedie deviance ou MAPE.
    - reliability diagram
    - Lorenz curve
    - Time-aware cross-validation.
    - Time of prediction in deployment setting != every possible times
    - models:
        - HGBDR
        - pipeline with missing value support: SimpleImputer with indicator, Spline, Nystroem, RidgeCV
- Train a family of t+h direct models and evaluate them:
    - Plot predictions at different time points.
    - Compute per-horizon metrics + metrics integrated over all horizons of interests
- Alternatives to a family of t+h direct models:
    - recursive modeling: show limitations on synthetic data (show with mlforecast, darts and sktime)
    - pass h as an extra features and generate expanded datasets for many h values (and nans in lag features).
- Quantify uncertainty in predictions with quantile regressors and evaluate them:
    - Study pinball loss, coverage / width of uncertainty regressors + reliability diagrams + Lorenz curve
    - Study if conformal predictions can improve upon this (optional)
        - Show limitation of split conformal predictions: 
        - Show CQR.
        - Non-exchangeable conformal prediction.

- Making models with lagged features robust to random missing values.
- TabICL or TabPFN on calendar + exogenous features (without lag features).

- Use skore at some points? Maybe not for this iteration.
