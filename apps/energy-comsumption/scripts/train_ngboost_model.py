# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from energy_consumption.metadata import (
    PATH_TO_METADATA,
    ROAD_TIME_COLUMN_NAME,
    TARGET_COLUMN_NAME,
    TRAINING_COLUMNS,
    load_and_prepare_metadata,
)
from ngboost import NGBRegressor
from ngboost.distns import Gamma, LogNormal, Normal
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

# %%
prepared_metadata_df = load_and_prepare_metadata(PATH_TO_METADATA.resolve())

# %%
# Investigate distribution of data
plot_df = prepared_metadata_df.assign(
    temperature_band=pd.cut(
        prepared_metadata_df["temperature_ambient_mean"] - 273.15,
        bins=[-np.inf, 10, 20, np.inf],
        labels=["Cold (<10 C)", "Mild (10-20 C)", "Warm (>=20 C)"],
        right=False,
    )
)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

sns.histplot(
    plot_df,
    x=TARGET_COLUMN_NAME,
    hue="temperature_band",
    # stat="density",
    common_norm=False,
    bins=30,
    kde=True,
    ax=axes[0][0],
)
axes[0][0].set_title("Energy consumption by temperature")

plot_df = prepared_metadata_df.assign(
    distance_band=pd.cut(
        prepared_metadata_df["drivenDistance"] / 1000,
        bins=[-np.inf, 100, 150, 200, 250, np.inf],
        right=False,
    )
)
sns.histplot(
    plot_df,
    x=TARGET_COLUMN_NAME,
    hue="distance_band",
    # stat="density",
    common_norm=False,
    bins=30,
    kde=True,
    ax=axes[0][1],
)
axes[0][1].set_title("Energy consumption by driven distance (in km)")

plot_df = prepared_metadata_df.assign(
    passenger_band=pd.cut(
        prepared_metadata_df["itcs_numberOfPassengers_mean"],
        bins=[0, 10, 20, 30, np.inf],
        right=False,
    )
)
sns.histplot(
    plot_df,
    x=TARGET_COLUMN_NAME,
    hue="passenger_band",
    # stat="density",
    common_norm=False,
    bins=30,
    kde=True,
    ax=axes[1][0],
)
axes[1][0].set_title("Energy consumption by # of passengers")

plot_df = prepared_metadata_df.assign(
    road_time_band=pd.cut(
        prepared_metadata_df[ROAD_TIME_COLUMN_NAME] / 60 / 60,
        bins=[-np.inf, 7, 11, 15, 19, np.inf],
        right=False,
    )
)
sns.histplot(
    plot_df,
    x=TARGET_COLUMN_NAME,
    hue="road_time_band",
    # stat="density",
    common_norm=False,
    bins=30,
    kde=True,
    ax=axes[1][1],
)
axes[1][1].set_title("Energy consumption by time spent on road (in h)")


plt.tight_layout()
plt.show()

print("Mean:", prepared_metadata_df[TARGET_COLUMN_NAME].mean())
print("Median:", prepared_metadata_df[TARGET_COLUMN_NAME].median())
print("Skewness:", prepared_metadata_df[TARGET_COLUMN_NAME].skew())
print("Kurtosis:", prepared_metadata_df[TARGET_COLUMN_NAME].kurtosis())

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
stats.probplot(
    prepared_metadata_df[TARGET_COLUMN_NAME].dropna(),
    dist="norm",
    plot=axes[0],
)
axes[0].set_title("Normal Q-Q plot")

sns.histplot(np.log(prepared_metadata_df[TARGET_COLUMN_NAME]), kde=True, ax=axes[1])
axes[1].set_title("Log energy consumption")
plt.show()

# %%
features = prepared_metadata_df[TRAINING_COLUMNS]
targets = prepared_metadata_df[TARGET_COLUMN_NAME]
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, random_state=42
)
# Split training set again for calibration
features_fit, features_cal, targets_fit, targets_cal = train_test_split(
    features_train, targets_train, test_size=0.25, random_state=42
)
cal_actual = targets_cal.to_numpy()
actual = targets_test.to_numpy()
comparison = []
predictions = {}
models = {}
for name, distribution in (
    ("Normal", Normal),
    ("LogNormal", LogNormal),
    ("Gamma", Gamma),
):
    model = NGBRegressor(
        Dist=distribution,
        n_estimators=500,
        learning_rate=0.03,
        verbose=False,
        random_state=42,
    )
    model.fit(features_fit, targets_fit)
    models[name] = model

    cal_dist = model.pred_dist(features_cal).dist
    # Measure how far each calibration value falls outside its predicted range.
    scores = np.maximum.reduce(
        (
            cal_dist.ppf(0.05) - cal_actual,
            cal_actual - cal_dist.ppf(0.95),
            np.zeros_like(cal_actual),
        )
    )
    rank = int(np.ceil((len(scores) + 1) * 0.90))
    adjustment = np.sort(np.append(scores, np.inf))[rank - 1]

    dist = model.pred_dist(features_test).dist
    predicted = np.asarray(dist.mean())
    raw_lower = dist.ppf(0.05)
    raw_upper = dist.ppf(0.95)
    lower = np.maximum(0, raw_lower - adjustment)
    upper = raw_upper + adjustment
    predictions[name] = (predicted, lower, upper)
    comparison.append(
        {
            "Distribution": name,
            "MAE (kWh)": np.mean(np.abs(actual - predicted)),
            "NLL": -np.mean(dist.logpdf(actual)),
            "Pinball 5% (kWh)": mean_pinball_loss(actual, raw_lower, alpha=0.05),
            "Pinball 95% (kWh)": mean_pinball_loss(actual, raw_upper, alpha=0.95),
            "90% coverage": np.mean((actual >= lower) & (actual <= upper)),
            "Mean width (kWh)": np.mean(upper - lower),
        }
    )

print(pd.DataFrame(comparison).set_index("Distribution").round(3).to_string())


# %%
def negative_95_pinball(estimator, features, targets):
    upper = estimator.pred_dist(features).dist.ppf(0.95)
    return -mean_pinball_loss(targets, upper, alpha=0.95)


# A larger score means shuffling the feature worsens the upper-quantile prediction.
importance = {}
for name, fitted_model in models.items():
    result = permutation_importance(
        fitted_model,
        features_test,
        targets_test,
        scoring=negative_95_pinball,
        n_repeats=10,
        random_state=42,
    )
    importance[name] = result.importances_mean

print("95% quantile permutation importance (increase in pinball loss, kWh):")
print(
    pd.DataFrame(importance, index=features_test.columns)
    .sort_values("Gamma", ascending=False)
    .round(3)
    .to_string()
)

# %%
fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for (name, (predicted, lower, upper)), (ax_scatter, ax_intervals) in zip(
    predictions.items(), axes
):
    ax_scatter.scatter(actual, predicted, alpha=0.5)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax_scatter.plot(limits, limits, "k--")
    ax_scatter.set(xlabel="Actual", ylabel="Predicted mean", title=f"{name}: accuracy")

    order = np.argsort(predicted)
    x = np.arange(len(order))
    ax_intervals.fill_between(
        x, lower[order], upper[order], alpha=0.25, label="90% interval"
    )
    ax_intervals.plot(x, predicted[order], label="Predicted mean")
    ax_intervals.scatter(x, actual[order], s=12, color="black", label="Actual")
    ax_intervals.set(
        xlabel="Test trips (sorted by predicted consumption)",
        ylabel="Consumption",
        title=f"{name}: predictive uncertainty",
    )
    ax_intervals.legend()

plt.tight_layout()
plt.show()

# %%
