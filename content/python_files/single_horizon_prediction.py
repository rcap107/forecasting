# %% [markdown]
#
# # Single horizon predictive modeling
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite). We need the development version of skrub to be able to
# use the skrub expressions.

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl
# %pip install -q altair holidays plotly nbformat

# %%
import datetime
import warnings

import skrub
import polars as pl
import altair
import cloudpickle
import numpy as np
from plotly.io import write_json, read_json  # noqa: F401

from tutorial_helpers import (
    binned_coverage,
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
    plot_binned_residuals,
    plot_horizon_forecast,
    collect_cv_predictions,
)


# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# %%
with open("feature_engineering_pipeline.pkl", "rb") as f:
    feature_engineering_pipeline = cloudpickle.load(f)


features = feature_engineering_pipeline["features"]
targets = feature_engineering_pipeline["targets"]
prediction_time = feature_engineering_pipeline["prediction_time"]
horizons = feature_engineering_pipeline["horizons"]
target_column_name_pattern = feature_engineering_pipeline["target_column_name_pattern"]

# %% [markdown]
#
# # Single horizon predictive modeling
#
# For now, let's focus on the last horizon (24 hours) to train a model
# predicting the electricity load at the next 24 hours.
# %%
horizon_of_interest = horizons[-1]  # Focus on the 24-hour horizon
target_column_name = target_column_name_pattern.format(horizon=horizon_of_interest)
predicted_target_column_name = "predicted_" + target_column_name
target = targets[target_column_name].skb.mark_as_y()
target

# %% [markdown]
#
# Let's define our first single output prediction pipeline. This pipeline
# chains our previous feature engineering steps with a `skrub.DropCols` step to
# drop some columns that we do not want to use as features, and a
# `HistGradientBoostingRegressor` model from scikit-learn.
#
# The `skrub.choose_from`, `skrub.choose_float`, and `skrub.choose_int`
# functions are used to define hyperparameters that can be tuned via
# cross-validated randomized search.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
import skrub.selectors as s


features_with_dropped_cols = features.skb.apply(
    skrub.DropCols(
        cols=skrub.choose_from(
            {
                "none": s.glob(""),  # No column has an empty name.
                "load": s.glob("load_*"),
                "rolling_load": s.glob("load_mw_rolling_*"),
                "weather": s.glob("weather_*"),
                "temperature": s.glob("weather_temperature_*"),
                "moisture": s.glob("weather_moisture_*"),
                "cloud_cover": s.glob("weather_cloud_cover_*"),
                "calendar": s.glob("cal_*"),
                "holiday": s.glob("cal_is_holiday*"),
                "future_1h": s.glob("*_future_1h"),
                "future_24h": s.glob("*_future_24h"),
                "non_paris_weather": s.glob("weather_*") & ~s.glob("weather_*_paris_*"),
            },
            name="dropped_cols",
        )
    )
)

hgbr_predictions = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(
        random_state=0,
        loss=skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss"),
        learning_rate=skrub.choose_float(
            0.01, 1, default=0.1, log=True, name="learning_rate"
        ),
        max_leaf_nodes=skrub.choose_int(
            3, 300, default=30, log=True, name="max_leaf_nodes"
        ),
    ),
    y=target,
)
hgbr_predictions

# %% [markdown]
#
# The `predictions` expression captures the whole expression graph that
# includes the feature engineering steps, the target variable, and the model
# training step.
#
# In particular, the input data keys for the full pipeline can be
# inspected as follows:

# %%
hgbr_predictions.skb.get_data().keys()

# %% [markdown]
#
# Furthermore, the hyper-parameters of the full pipeline can be retrieved as
# follows:

# %%
hgbr_pipeline = hgbr_predictions.skb.get_pipeline()
hgbr_pipeline.describe_params()

# %% [markdown]
#
# When running this notebook locally, you can also interactively inspect all
# the steps of the DAG using the following (once uncommented):

# %%
# predictions.skb.full_report()

# %% [markdown]
#
# Since we passed input values to all the upstream `skrub` variables, `skrub`
# automatically evaluates the whole expression graph graph (train and predict
# on the same data) so that we can interactively check that everything will
# work as expected.
#
# Let's use altair to visualize the predictions against the target values for
# the last 24 hours of the prediction time range used to train the model. This
# allows us can (over)fit the data with the features at hand.

# %%
altair.Chart(
    pl.concat(
        [
            targets.skb.preview(),
            hgbr_predictions.rename(
                {target_column_name: predicted_target_column_name}
            ).skb.preview(),
        ],
        how="horizontal",
    ).tail(24 * 7)
).transform_fold(
    [target_column_name, predicted_target_column_name],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()

# %% [markdown]
#
# ## Assessing the model performance via cross-validation
#
# Being able to fit the training data is not enough. We need to assess the
# ability of the training pipeline to learn a predictive model that can
# generalize to unseen data.
#
# Furthermore, we want to assess the uncertainty of this estimate of the
# generalization performance via time-based cross-validation, also known as
# backtesting.
#
# scikit-learn provides a `TimeSeriesSplit` splitter providing a convenient way to
# split temporal data: in the different folds, the training data always precedes the
# test data. It implies that the size of the training data is getting larger as the
# fold index increases. The scikit-learn utility allows to define a couple of
# parameters to control the size of the training and test data and as well as a gap
# between the training and test data to potentially avoid leakage if our model relies
# on lagged features.
#
# In the example below, we define that the training data should be at most 2 years
# worth of data and the test data should be 24 weeks long. We also define a gap of
# 1 week between the training.
#
# Let's check those statistics by iterating over the different folds provided by the
# splitter.

# %%
from sklearn.model_selection import TimeSeriesSplit


max_train_size = 2 * 52 * 24 * 7  # max ~2 years of training data
test_size = 24 * 7 * 24  # 24 weeks of test data
gap = 7 * 24  # 1 week gap between train and test sets
ts_cv_5 = TimeSeriesSplit(
    n_splits=5, max_train_size=max_train_size, test_size=test_size, gap=gap
)

for fold_idx, (train_idx, test_idx) in enumerate(
    ts_cv_5.split(prediction_time.skb.eval())
):
    print(f"CV iteration #{fold_idx}")
    train_datetimes = prediction_time.skb.eval()[train_idx]
    test_datetimes = prediction_time.skb.eval()[test_idx]
    print(
        f"Train: {train_datetimes.shape[0]} rows, "
        f"Test: {test_datetimes.shape[0]} rows"
    )
    print(f"Train time range: {train_datetimes[0, 0]} to " f"{train_datetimes[-1, 0]} ")
    print(f"Test time range: {test_datetimes[0, 0]} to " f"{test_datetimes[-1, 0]} ")
    print()

# %% [markdown]
#
# Once the cross-validation strategy is defined, we pass it to the
# `cross_validate` function provided by `skrub` to compute the cross-validated
# scores. Here, we compute the mean absolute percentage error that is easily
# interpretable and customary for regression tasks with a strictly positive
# target variable such as electricity load forecasting.
#
# We can also look at the R2 score and the Poisson and Gamma deviance which are
# all strictly proper scoring rules for estimation of E[y|X]: in the large
# sample limit, minimizers of those metrics all identify the conditional
# expectation of the target variable given the features for strictly positive
# target variables. All those metrics follow the higher is better convention,
# 1.0 is the maximum reachable score and 0.0 is the score of a model that
# predicts the mean of the target variable for all observations, irrespective
# of the features.
#
# No that in general, a deviance score of 1.0 is not reachable since it
# corresponds to a model that always predicts the target value exactly
# for all observations. In practice, because there is always a fraction of the
# variability in the target variable that is not explained by the information
# available to construct the features.

# %%
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, get_scorer
from sklearn.metrics import d2_tweedie_score


hgbr_cv_results = hgbr_predictions.skb.cross_validate(
    cv=ts_cv_5,
    scoring={
        "mape": make_scorer(mean_absolute_percentage_error),
        "r2": get_scorer("r2"),
        "d2_poisson": make_scorer(d2_tweedie_score, power=1.0),
        "d2_gamma": make_scorer(d2_tweedie_score, power=2.0),
    },
    return_train_score=True,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
hgbr_cv_results.round(3)

# %% [markdown]
#
# Those results show very good performance of the model: less than 3% of mean
# absolute percentage error (MAPE) on the test folds. Similarly, all the
# deviance scores are close to 1.0.
# 
# We observe a bit of variability in the scores across the different folds: in
# particular the test performance on the first fold seems to be worse than the
# other folds. This is likely due to the fact that the first fold contains
# training data from 2021 and 2022 and the test data mostly from 2023.
# 
# The invasion in Ukraine and a sharp drop in nuclear electricity production
# due to safety problems strongly impacted the distribution of the electricity
# prices in 2022, with unprecedented high prices, which can in turn cause a
# shift in the electricity load demand. This could explain a higher than usual
# distribution shift between the train and test folds of the first CV
# iteration.
#
# We can further refine the analysis of the performance of our model by
# collecting the predictions on each cross-validation split.


# %%
hgbr_cv_predictions = collect_cv_predictions(
    hgbr_cv_results["pipeline"], ts_cv_5, hgbr_predictions, prediction_time
)
hgbr_cv_predictions[0]

# %% [markdown]
#
# The first curve is called the Lorenz curve. It shows on the x-axis the fraction of
# observations sorted by predicted values and on the y-axis the cumulative observed
# load proportion.

# %%
plot_lorenz_curve(hgbr_cv_predictions).interactive()

# %% [markdown]
#
# The diagonal on the plot corresponds to a model predicting a constant value that is
# therefore not an informative model. The oracle model corresponds to the "perfect"
# model that would provide the an output identical to the observed values. Thus, the
# ranking of such hypothetical model is the best possible ranking. However, you should
# note that the oracle model is not the line passing through the right-hand corner of
# the plot. Instead, this curvature is defined by the distribution of the observations.
# Indeed, more the observations are composed of small values and a couple of large
# values, the more the oracle model is closer to the right-hand corner of the plot.
#
# A true model is navigating between the diagonal and the oracle model. The area between
# the diagonal and the Lorenz curve of a model is called the Gini index.
#
# For our model, we observe that each oracle model is not far from the diagonal. It
# means that the observed values do not contain a couple of large values with high
# variability. Therefore, it informs us that the complexity of our problem at hand is
# not too high. Looking at the Lorenz curve of each model, we observe that it is quite
# close to the oracle model. Therefore, the gradient boosting regressor is
# discriminative for our task.
#
# Then, we have a look at the reliability diagram. This diagram shows on the x-axis the
# mean predicted load and on the y-axis the mean observed load.

# %%
plot_reliability_diagram(hgbr_cv_predictions).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %% [markdown]
#
# The diagonal on the reliability diagram corresponds to the best possible model: for
# a level of predicted load that fall into a bin, then the mean observed load is also
# in the same bin. If the line is above the diagonal, it means that our model is
# predicted a value too low in comparison to the observed values. If the line is below
# the diagonal, it means that our model is predicted a value too high in comparison to
# the observed values.
#
# For our cross-validated model, we observe that each reliability curve is close to the
# diagonal. We only observe a mis-calibration for the extremum values.

# %%
plot_residuals_vs_predicted(hgbr_cv_predictions).interactive().properties(
    title="Residuals vs Predicted Values from cross-validation predictions"
)

# %%
plot_binned_residuals(hgbr_cv_predictions, by="hour").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

# %%
plot_binned_residuals(hgbr_cv_predictions, by="month").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

# %%
ts_cv_2 = TimeSeriesSplit(
    n_splits=2, test_size=test_size, max_train_size=max_train_size, gap=24
)
# randomized_search_hgbr = hgbr_predictions.skb.get_randomized_search(
#     cv=ts_cv_2,
#     scoring="r2",
#     n_iter=100,
#     fitted=True,
#     verbose=1,
#     n_jobs=-1,
# )
# # %%
# randomized_search_hgbr.results_.round(3)

# %%
# fig = randomized_search_hgbr.plot_results().update_layout(margin=dict(l=200))
# write_json(fig, "parallel_coordinates_hgbr.json")

# %%
fig = read_json("parallel_coordinates_hgbr.json")
fig.update_layout(margin=dict(l=200))

# %%
# nested_cv_results = skrub.cross_validate(
#     environment=predictions.skb.get_data(),
#     pipeline=randomized_search,
#     cv=ts_cv_5,
#     scoring={
#         "r2": get_scorer("r2"),
#         "mape": make_scorer(mean_absolute_percentage_error),
#     },
#     n_jobs=-1,
#     return_pipeline=True,
# ).round(3)
# nested_cv_results

# %%
# for outer_fold_idx in range(len(nested_cv_results)):
#     print(
#         nested_cv_results.loc[outer_fold_idx, "pipeline"]
#         .results_.loc[0]
#         .round(3)
#         .to_dict()
#     )

# %% [markdown]
#
# ### Exercise: non-linear feature engineering coupled with linear predictive model
#
# Now, it is your turn to make a predictive model. Towards this end, we request you
# to preprocess the input features with non-linear feature engineering:
#
# - the first step is to impute the missing values using a `SimpleImputer`. Make sure
#   to include the indicator of missing values in the feature set (i.e. look at the
#   `add_indicator` parameter);
# - use a `SplineTransformer` to create non-linear features. Use the default parameters
#   but make sure to set `sparse_output=True` since it subsequent processing will be
#   faster and more memory efficient with such data structure;
# - use a `VarianceThreshold` to remove features with potential constant features;
# - use a `SelectKBest` to select the most informative features. Set `k` to be chosen
#   from a log-uniform distribution between 100 and 1,000 (i.e. use `skrub.choose_int`);
# - use a `Nystroem` to approximate an RBF kernel. Set `n_components` to be chosen
#   from a log-uniform distribution between 10 and 200 (i.e. use `skrub.choose_int`).
# - finally, use a `Ridge` as the final predictive model. Set `alpha` to be
#   chosen from a log-uniform distribution between 1e-6 and 1e3 (i.e. use
#   `skrub.choose_float`).
#
# Use a scikit-learn `Pipeline` using `make_pipeline` to chain the steps together.
#
# Once the predictive model is defined, apply it on the `feature_with_dropped_cols`
# expression. Do not forget to define that `target` is the `y` variable.


# %%
# Here we provide all the imports for creating the predictive model.
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
predictions_ridge = features_with_dropped_cols.skb.apply(
    make_pipeline(
        SimpleImputer(add_indicator=True),
        SplineTransformer(sparse_output=True),
        VarianceThreshold(threshold=1e-6),
        SelectKBest(
            k=skrub.choose_int(100, 1_000, log=True, name="n_selected_splines")
        ),
        Nystroem(
            n_components=skrub.choose_int(
                10, 200, log=True, name="n_components", default=150
            )
        ),
        Ridge(
            alpha=skrub.choose_float(1e-6, 1e3, log=True, name="alpha", default=1e-2)
        ),
    ),
    y=target,
)
predictions_ridge

# %% [markdown]
#
# Now that you defined the predictive model, let's make a similar analysis than earlier.
# First, let's make a sanity check that plot forecast of our model on a subset of the
# training data to make a sanity check.

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
altair.Chart(
    pl.concat(
        [
            targets.skb.preview(),
            predictions_ridge.rename(
                {target_column_name: predicted_target_column_name}
            ).skb.preview(),
        ],
        how="horizontal",
    ).tail(24 * 7)
).transform_fold(
    [target_column_name, predicted_target_column_name],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()


# %% [markdown]
#
# Now, let's evaluate the performance of the model using cross-validation. Use the
# time-based cross-validation splitter `ts_cv_5` defined earlier. Make sure to compute
# the R2 score and the mean absolute percentage error. Return the training scores as
# well as the fitted pipeline such that we can make additional analysis.
#
# Does this model perform better or worse than the previous model?
# Is it underfitting or overfitting?

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
cv_results_ridge = predictions_ridge.skb.cross_validate(
    cv=ts_cv_5,
    scoring={
        "r2": get_scorer("r2"),
        "mape": make_scorer(mean_absolute_percentage_error),
    },
    return_train_score=True,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)

# %% [markdown]
#
# Compute all cross-validated predictions to plot the Lorenz curve and the
# reliability diagram for this pipeline.
#
# To do so, you can use the function `collect_cv_predictions` to collect the
# predictions and then call the `plot_lorenz_curve` and
# `plot_reliability_diagram` functions to plot the results.

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
cv_predictions_ridge = collect_cv_predictions(
    cv_results_ridge["pipeline"], ts_cv_5, predictions_ridge, prediction_time
)

# %%
plot_lorenz_curve(cv_predictions_ridge).interactive()

# %%
plot_reliability_diagram(cv_predictions_ridge).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %% [markdown]
#
# Now, let's perform a randomized search on the hyper-parameters of the model. The code
# to perform the search is shown below. Since it will be pretty computationally
# expensive, we are reloading the results of the parallel coordinates plot.

# %%
# randomized_search_ridge = predictions_ridge.skb.get_randomized_search(
#     cv=ts_cv_2,
#     scoring="r2",
#     n_iter=100,
#     fitted=True,
#     verbose=1,
#     n_jobs=-1,
# )

# %%
# fig = randomized_search_ridge.plot_results().update_layout(margin=dict(l=200))
# write_json(fig, "parallel_coordinates_ridge.json")

# %%
fig = read_json("parallel_coordinates_ridge.json")
fig.update_layout(margin=dict(l=200))

# %% [markdown]
#
# We observe that the default values of the hyper-parameters are in the optimal
# region explored by the randomized search. This is a good sign that the model
# is well-specified and that the hyper-parameters are not too sensitive to
# small changes of those values.
#
# We could further assess the stability of those optimal hyper-parameters by
# running a nested cross-validation, where we would perform a randomized search
# for each fold of the outer cross-validation loop as below but this is
# computationally expensive.

# %%
# nested_cv_results_ridge = skrub.cross_validate(
#     environment=predictions_ridge.skb.get_data(),
#     pipeline=randomized_search_ridge,
#     cv=ts_cv_5,
#     scoring={
#         "r2": get_scorer("r2"),
#         "mape": make_scorer(mean_absolute_percentage_error),
#     },
#     n_jobs=-1,
#     return_pipeline=True,
# ).round(3)

# %%
# nested_cv_results_ridge.round(3)

# %% [markdown]
#
# ## Predicting multiple horizons with a multi-output model
#
# Usually, it is really common to predict values for multiple horizons at once. The most
# naive approach is to train as many models as there are horizons. To achieve this,
# scikit-learn provides a meta-estimator called `MultiOutputRegressor` that can be used
# to train a single model that predicts multiple horizons at once.
#
# In short, we only need to provide multiple targets where each column corresponds to
# an horizon and this meta-estimator will train an independent model for each column.
# However, we could expect that the quality of the forecast might degrade as the horizon
# increases.
#
# Let's train a gradient boosting regressor for each horizon.

# %%
from sklearn.multioutput import MultiOutputRegressor

multioutput_predictions = features_with_dropped_cols.skb.apply(
    MultiOutputRegressor(
        estimator=HistGradientBoostingRegressor(random_state=0), n_jobs=-1
    ),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
)

# %% [markdown]
#
# Now, let's just rename the columns for the predictions to make it easier to plot
# the horizon forecast.

# %%
target_column_names = [target_column_name_pattern.format(horizon=h) for h in horizons]
predicted_target_column_names = [
    f"predicted_{target_column_name}" for target_column_name in target_column_names
]
named_predictions = multioutput_predictions.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %% [markdown]
#
# Let's plot the horizon forecast on a training data to check the validity of the
# output.

# %%
plot_at_time = datetime.datetime(2021, 4, 19, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions,
    plot_at_time,
    target_column_name_pattern,
).skb.preview()

# %% [markdown]
#
# On this curve, the red line corresponds to the observed values past to the the date
# for which we would like to forecast. The orange line corresponds to the observed
# values for the next 24 hours and the blue line corresponds to the predicted values
# for the next 24 hours.
#
# Since we are using a strong model and very few training data to check the validity
# we observe that our model perfectly fits the training data.
#
# So, we are now ready to assess the performance of this multi-output model and we need
# to cross-validate it. Since we do not want to aggregate the metrics for the different
# horizons, we need to create a scikit-learn scorer in which we set
# `multioutput="raw_values"` to get the scores for each horizon.
#
# Passing this scorer to the `cross_validate` function returns all horizons scores.

# %%
from sklearn.metrics import r2_score


def multioutput_scorer(regressor, X, y, score_func, score_name):
    y_pred = regressor.predict(X)
    return {
        f"{score_name}_horizon_{h}h": score
        for h, score in enumerate(
            score_func(y, y_pred, multioutput="raw_values"), start=1
        )
    }


def scoring(regressor, X, y):
    return {
        **multioutput_scorer(regressor, X, y, mean_absolute_percentage_error, "mape"),
        **multioutput_scorer(regressor, X, y, r2_score, "r2"),
    }


multioutput_cv_results = multioutput_predictions.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
)

# %% [markdown]
#
# One thing that we observe is that training such multi-output model is expensive. It is
# expected since each horizon involves a different model and thus a training.

# %%
multioutput_cv_results.round(3)

# %% [markdown]
#
# Instead of reading the results in the table, we can plot the scores depending on the
# type of data and the metric.

# %%
import itertools
from IPython.display import display

for metric_name, dataset_type in itertools.product(["mape", "r2"], ["train", "test"]):
    columns = multioutput_cv_results.columns[
        multioutput_cv_results.columns.str.startswith(f"{dataset_type}_{metric_name}")
    ]
    data_to_plot = multioutput_cv_results[columns]
    data_to_plot.columns = [
        col.replace(f"{dataset_type}_", "")
        .replace(f"{metric_name}_", "")
        .replace("_", " ")
        for col in columns
    ]

    data_long = data_to_plot.melt(var_name="horizon", value_name="score")
    chart = (
        altair.Chart(
            data_long,
            title=f"{dataset_type.title()} {metric_name.upper()} scores by horizon",
        )
        .mark_boxplot(extent="min-max")
        .encode(
            x=altair.X(
                "horizon:N",
                title="Horizon",
                sort=altair.Sort(
                    [f"horizon {h}h" for h in range(1, data_to_plot.shape[1])]
                ),
            ),
            y=altair.Y("score:Q", title=f"{metric_name.upper()} Score"),
            color=altair.Color("horizon:N", legend=None),
        )
    )

    display(chart)

# %% [markdown]
#
# An interesting and unexpected observation is that the MAPE error on the test
# data is first increases and then decreases once past the horizon 18h. We
# would not necessarily expect this behaviour.
#
# ## Native multi-output handling using `RandomForestRegressor`
#
# In the previous section, we showed how to wrap a `HistGradientBoostingRegressor`
# in a `MultiOutputRegressor` to predict multiple horizons. With such a strategy, it
# means that we trained independent `HistGradientBoostingRegressor`, one for each
# horizon.
#
# `RandomForestRegressor` natively supports multi-output regression: instead of
# independently training a model per horizon, it will train a joint model that
# predicts all horizons at once.
#
# Repeat the previous analysis using a `RandomForestRegressor`. Fix the parameter
# `min_samples_leaf` to 5.
#
# Once you created the model, plot the horizon forecast for a given date and time.
# In addition, compute the cross-validated predictions and plot the R2 and MAPE
# scores for each horizon.
#
# Does this model perform better or worse than the previous model?

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
multioutput_predictions_rf = features_with_dropped_cols.skb.apply(
    RandomForestRegressor(min_samples_leaf=5, random_state=0, n_jobs=-1),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
)

# %%
named_predictions_rf = multioutput_predictions_rf.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %%
plot_at_time = datetime.datetime(2021, 4, 24, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions_rf,
    plot_at_time,
    target_column_name_pattern,
).skb.preview()

# %%
multioutput_cv_results_rf = multioutput_predictions_rf.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
)

# %%
multioutput_cv_results_rf.round(3)

# %%
import itertools
from IPython.display import display

for metric_name, dataset_type in itertools.product(["mape", "r2"], ["train", "test"]):
    columns = multioutput_cv_results_rf.columns[
        multioutput_cv_results_rf.columns.str.startswith(
            f"{dataset_type}_{metric_name}"
        )
    ]
    data_to_plot = multioutput_cv_results_rf[columns]
    data_to_plot.columns = [
        col.replace(f"{dataset_type}_", "")
        .replace(f"{metric_name}_", "")
        .replace("_", " ")
        for col in columns
    ]

    data_long = data_to_plot.melt(var_name="horizon", value_name="score")
    chart = (
        altair.Chart(
            data_long,
            title=f"{dataset_type.title()} {metric_name.upper()} Scores by Horizon",
        )
        .mark_boxplot(extent="min-max")
        .encode(
            x=altair.X(
                "horizon:N",
                title="Horizon",
                sort=altair.Sort(
                    [f"horizon {h}h" for h in range(1, data_to_plot.shape[1])]
                ),
            ),
            y=altair.Y("score:Q", title=f"{metric_name.upper()} Score"),
            color=altair.Color("horizon:N", legend=None),
        )
    )

    display(chart)

# %% [markdown]
#
# We observe that the performance of the `RandomForestRegressor` is not better in terms
# of scores or computational cost. The trend of the scores along the horizon is also
# different from the `HistGradientBoostingRegressor`: the scores worsen as the horizon
# increases.
#
# ## Uncertainty quantification using quantile regression
#
# ### Define the quantile regressors
#
# In this section, we show how one can use a gradient boosting but modify the loss
# function to predict different quantiles and thus obtain an uncertainty quantification
# of the predictions.
#
# In terms of evaluation, we reuse the R2 and MAPE scores. However, they are not helpful
# to assess the reliability of quantile models. For this purpose, we use a derivate of
# the metric minimize by those models: the pinball loss. We use the D2 score that is
# easier to interpret since the best possible score is bounded by 1 and a score of 0
# corresponds to constant predictions at the target quantile.

# %%
from sklearn.metrics import d2_pinball_score

scoring = {
    "r2": get_scorer("r2"),
    "mape": make_scorer(mean_absolute_percentage_error),
    "d2_pinball_05": make_scorer(d2_pinball_score, alpha=0.05),
    "d2_pinball_50": make_scorer(d2_pinball_score, alpha=0.50),
    "d2_pinball_95": make_scorer(d2_pinball_score, alpha=0.95),
}

# %% [markdown]
#
# We know define three different models:
#
# - a model predicting the 5th percentile of the load
# - a model predicting the median of the load
# - a model predicting the 95th percentile of the load

# %%
common_params = dict(
    loss="quantile", learning_rate=0.1, max_leaf_nodes=100, random_state=0
)
predictions_hgbr_05 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.05),
    y=target,
)
predictions_hgbr_50 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.5),
    y=target,
)
predictions_hgbr_95 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.95),
    y=target,
)

# %% [markdown]
#
# Now, let's make a plot of the predictions for each model and thus we need to gather
# all the predictions in a single dataframe.

# %%
results = pl.concat(
    [
        targets.skb.select(cols=["prediction_time", target_column_name]).skb.preview(),
        predictions_hgbr_05.rename({target_column_name: "quantile_05"}).skb.preview(),
        predictions_hgbr_50.rename({target_column_name: "median"}).skb.preview(),
        predictions_hgbr_95.rename({target_column_name: "quantile_95"}).skb.preview(),
    ],
    how="horizontal",
).tail(24 * 7)

# %% [markdown]
#
# Now, we plot the observed values and the predicted median with a line. In addition,
# we plot the 5th and 95th percentiles as a shaded area. It means that between those
# two bounds, we expect to find 90% of the observed values.
#
# We plot this information on a portion of the training data to observe the uncertainty
# quantification of the predictions.

# %%
median_chart = (
    altair.Chart(results)
    .transform_fold([target_column_name, "median"])
    .mark_line(tooltip=True)
    .encode(x="prediction_time:T", y="value:Q", color="key:N")
)

# Add a column for the band legend
results_with_band = results.with_columns(pl.lit("90% interval").alias("band_type"))

quantile_band_chart = (
    altair.Chart(results_with_band)
    .mark_area(opacity=0.4, tooltip=True)
    .encode(
        x="prediction_time:T",
        y="quantile_05:Q",
        y2="quantile_95:Q",
        color=altair.Color("band_type:N", scale=altair.Scale(range=["lightgreen"])),
    )
)

combined_chart = quantile_band_chart + median_chart
combined_chart.resolve_scale(color="independent").interactive()

# %% [markdown]
#
# While we should expend this plot on the test data and on several portions of the
# dataset, we observe a potential interesting pattern: during the week days, the
# 5th percentile is further from the median than during the weekend. However, for the
# 95th percentile, the opposite is observed.
#
# ### Evaluation via cross-validation
#
# We evaluate the performance of the quantile regressors via cross-validation.

# %%
cv_results_hgbr_05 = predictions_hgbr_05.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
cv_results_hgbr_50 = predictions_hgbr_50.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
cv_results_hgbr_95 = predictions_hgbr_95.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_results_hgbr_05[
    [col for col in cv_results_hgbr_05.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
cv_results_hgbr_50[
    [col for col in cv_results_hgbr_50.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
cv_results_hgbr_95[
    [col for col in cv_results_hgbr_95.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %% [markdown]
#
# Focusing on the different D2 scores, we observe that each model minimize the D2 score
# associated to the target quantile that we set. For instance, the model predicting the
# 5th percentile obtained the highest D2 pinball score with `alpha=0.05`. It is expected
# but a confirmation of what loss each model minimizes.
#
# Now, let's collect the cross-validated predictions and plot the residual vs predicted
# values for the different models.

# %%
cv_predictions_hgbr_05 = collect_cv_predictions(
    cv_results_hgbr_05["pipeline"], ts_cv_5, predictions_hgbr_05, prediction_time
)
cv_predictions_hgbr_50 = collect_cv_predictions(
    cv_results_hgbr_50["pipeline"], ts_cv_5, predictions_hgbr_50, prediction_time
)
cv_predictions_hgbr_95 = collect_cv_predictions(
    cv_results_hgbr_95["pipeline"], ts_cv_5, predictions_hgbr_95, prediction_time
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_05).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.05"
    )
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_50).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions for median"
    )
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_95).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.95"
    )
)

# %% [markdown]
#
# We observe an expected behaviour: the residuals are centered and symmetric around 0
# for the median model while not centered and biased for the 5th and 95th percentiles
# models.
#
# Note that we could obtain similar plots using scikit-learn's `PredictionErrorDisplay`.
# This display allows to also plot the observed values vs predicted values as well.

# %%
cv_predictions_hgbr_05_concat = pl.concat(cv_predictions_hgbr_05, how="vertical")
cv_predictions_hgbr_50_concat = pl.concat(cv_predictions_hgbr_50, how="vertical")
cv_predictions_hgbr_95_concat = pl.concat(cv_predictions_hgbr_95, how="vertical")

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


for kind in ["actual_vs_predicted", "residual_vs_predicted"]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_05_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[0],
    )
    axs[0].set_title("0.05 quantile regression")

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_50_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[1],
    )
    axs[1].set_title("Median regression")

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_95_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[2],
    )
    axs[2].set_title("0.95 quantile regression")

    fig.suptitle(f"{kind} for GBRT minimzing different quantile losses")

# %% [markdown]
#
# Those plots carry the same information than the previous ones.
#
# Now, we assess if the actual coverage of the models is close to the target coverage of
# 90%. In addition, we compute the average width of the bands.

# %%
def coverage(y_true, y_quantile_low, y_quantile_high):
    y_true = np.asarray(y_true)
    y_quantile_low = np.asarray(y_quantile_low)
    y_quantile_high = np.asarray(y_quantile_high)
    return float(
        np.logical_and(y_true >= y_quantile_low, y_true <= y_quantile_high)
        .mean()
        .round(4)
    )


def mean_width(y_true, y_quantile_low, y_quantile_high):
    y_true = np.asarray(y_true)
    y_quantile_low = np.asarray(y_quantile_low)
    y_quantile_high = np.asarray(y_quantile_high)
    return float(np.abs(y_quantile_high - y_quantile_low).mean().round(1))


# %%
coverage(
    cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
    cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
    cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
)

# %% [markdown]
#
# We see that the obtained coverage (~77%) on the cross-validated predictions is much
# lower than the target coverage of 90%. It means that the pair of regressors is not
# jointly calibrated to estimate the 90% interval.

# %%
mean_width(
    cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
    cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
    cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
)

# %% [markdown]
#
# In terms of interpretable measure, the mean width provides a measure in the original
# unit of the target variable in MW that is ~5,100 MW.
#
# We can go a bit further and bin the cross-validated predictions and check if some
# specific bins show a better or worse coverage.

# %%
binned_coverage_results = binned_coverage(
    [df["load_mw"].to_numpy() for df in cv_predictions_hgbr_50],
    [df["predicted_load_mw"].to_numpy() for df in cv_predictions_hgbr_05],
    [df["predicted_load_mw"].to_numpy() for df in cv_predictions_hgbr_95],
    n_bins=10,
)
binned_coverage_results

# %% [markdown]
#
# Let's make a plot to check those data visually.

# %%
coverage_by_bin = binned_coverage_results.copy()
coverage_by_bin["bin_label"] = coverage_by_bin.apply(
    lambda row: f"[{row.bin_left:.0f}, {row.bin_right:.0f}]", axis=1
)

# %%
ax = coverage_by_bin.boxplot(column="coverage", by="bin_label", whis=1000)
ax.axhline(y=0.9, color="red", linestyle="--", label="Target coverage (0.9)")
ax.set(xlabel="Load bins (MW)", ylabel="Coverage", title="Coverage Distribution by Load Bins")
ax.set_title("Coverage Distribution by Load Bins")
ax.legend()
plt.suptitle("")  # Remove automatic suptitle from boxplot
_ = plt.xticks(rotation=45)

# %% [markdown]
#
# We observe that the lower and higher bins, so low and high load, have the worse
# coverage with a high variability.
#
# ### Reliability diagrams and Lorenz curves for quantile regression

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_50, kind="quantile", quantile_level=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_05, kind="quantile", quantile_level=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_95, kind="quantile", quantile_level=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)


# %% [markdown]
#
# ## Quantile regression as classification
#
# In the following, we turn a quantile regression problem for all possible
# quantile levels into a multiclass classification problem by discretizing the
# target variable into bins and interpolating the cumulative sum of the bin
# membership probability to estimate the CDF of the distribution of the
# continuous target variable conditioned on the features.
#
# Ideally, the classifier should be efficient when trained on a large number of
# classes (induced by the number of bins). Therefore we use a Random Forest
# classifier as the default base estimator.
#
# There are several advantages to this approach:
# - a single model is trained and can jointly estimate quantiles for all
#   quantile levels (assuming a well tuned number of bins);
# - the quantile levels can be chosen at prediction time, which allows for a
#   flexible quantile regression model;
# - in practice, the resulting predictions are often reasonably well calibrated
#   as we will see in the reliability diagrams below.
#
# One possible drawback is that current implementations of gradient boosting
# models tend to be very slow to train with a large number of classes. Random
# Forests are much more efficient in this case, but they do not always provide
# the best predictive performance. It could be the case that combining this
# approach with tabular neural networks can lead to competitive results.
#
# However, the current scikit-learn API is not expressive enough to to handle
# the output shape of the quantile prediction function. We therefore cannot
# make it fit into a skrub pipeline.

# %%
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_consistent_length
from sklearn.utils import check_random_state
import numpy as np


class BinnedQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator=None,
        n_bins=100,
        quantile=0.5,
        random_state=None,
    ):
        self.n_bins = n_bins
        self.estimator = estimator
        self.quantile = quantile
        self.random_state = random_state

    def fit(self, X, y):
        # Lightweight input validation: most of the input validation will be
        # handled by the sub estimators.
        random_state = check_random_state(self.random_state)
        check_consistent_length(X, y)
        self.target_binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy="quantile",
            subsample=200_000,
            encode="ordinal",
            quantile_method="averaged_inverted_cdf",
            random_state=random_state,
        )

        y_binned = (
            self.target_binner_.fit_transform(np.asarray(y).reshape(-1, 1))
            .ravel()
            .astype(np.int32)
        )

        # Fit the multiclass classifier to predict the binned targets from the
        # training set.
        if self.estimator is None:
            estimator = RandomForestClassifier(random_state=random_state)
        else:
            estimator = clone(self.estimator)
        self.estimator_ = estimator.fit(X, y_binned)
        return self

    def predict_quantiles(self, X, quantiles=(0.05, 0.5, 0.95)):
        check_is_fitted(self, "estimator_")
        edges = self.target_binner_.bin_edges_[0]
        n_bins = edges.shape[0] - 1
        expected_shape = (X.shape[0], n_bins)
        y_proba_raw = self.estimator_.predict_proba(X)

        # Some might stay empty on the training set. Typically, classifiers do
        # not learn to predict an explicit 0 probability for unobserved classes
        # so we have to post process their output:
        if y_proba_raw.shape != expected_shape:
            y_proba = np.zeros(shape=expected_shape)
            y_proba[:, self.estimator_.classes_] = y_proba_raw
        else:
            y_proba = y_proba_raw

        # Build the mapper for inverse CDF mapping, from cumulated
        # probabilities to continuous prediction.
        y_cdf = np.zeros(shape=(X.shape[0], edges.shape[0]))
        y_cdf[:, 1:] = np.cumsum(y_proba, axis=1)
        return np.asarray([interp1d(y_cdf_i, edges)(quantiles) for y_cdf_i in y_cdf])

    def predict(self, X):
        return self.predict_quantiles(X, quantiles=(self.quantile,)).ravel()


# %%
quantiles = (0.05, 0.5, 0.95)
bqr = BinnedQuantileRegressor(
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.2,
        n_jobs=-1,
        random_state=0,
    ),
    n_bins=30,
)
bqr

# %%
from sklearn.model_selection import cross_validate

X, y = features_with_dropped_cols.skb.eval(), target.skb.eval()

cv_results_bqr = cross_validate(
    bqr,
    X,
    y,
    cv=ts_cv_5,
    scoring={
        "d2_pinball_50": make_scorer(d2_pinball_score, alpha=0.5),
    },
    return_estimator=True,
    return_indices=True,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_predictions_bqr_all = [
    cv_predictions_bqr_05 := [],
    cv_predictions_bqr_50 := [],
    cv_predictions_bqr_95 := [],
]
for fold_ix, (qreg, test_idx) in enumerate(
    zip(cv_results_bqr["estimator"], cv_results_bqr["indices"]["test"])
):
    print(f"CV iteration #{fold_ix}")
    print(f"Test set size: {test_idx.shape[0]} rows")
    print(
        f"Test time range: {prediction_time.skb.eval()[test_idx][0, 0]} to "
        f"{prediction_time.skb.eval()[test_idx][-1, 0]} "
    )
    y_pred_all_quantiles = qreg.predict_quantiles(X[test_idx], quantiles=quantiles)

    coverage_score = coverage(
        y[test_idx],
        y_pred_all_quantiles[:, 0],
        y_pred_all_quantiles[:, 2],
    )
    print(f"Coverage: {coverage_score:.3f}")

    mean_width_score = mean_width(
        y[test_idx],
        y_pred_all_quantiles[:, 0],
        y_pred_all_quantiles[:, 2],
    )
    print(f"Mean prediction interval width: " f"{mean_width_score:.1f} MW")

    for q_idx, (quantile, predictions) in enumerate(
        zip(quantiles, cv_predictions_bqr_all)
    ):
        observed = y[test_idx]
        predicted = y_pred_all_quantiles[:, q_idx]
        predictions.append(
            pl.DataFrame(
                {
                    "prediction_time": prediction_time.skb.eval()[test_idx],
                    "load_mw": observed,
                    "predicted_load_mw": predicted,
                }
            )
        )
        print(f"d2_pinball score: {d2_pinball_score(observed, predicted):.3f}")
    print()

# %% [markdown
# Let's assess the calibration of the quantile regression model:

# %%
plot_reliability_diagram(
    cv_predictions_bqr_50, kind="quantile", quantile_level=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr_05, kind="quantile", quantile_level=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr_95, kind="quantile", quantile_level=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %% [markdown]
#
# We can complement this assessment with the Lorenz curves, which only assess
# the ranking power of the predictions, irrespective of their absolute values.

# %%
plot_lorenz_curve(cv_predictions_bqr_50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr_05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr_95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)
