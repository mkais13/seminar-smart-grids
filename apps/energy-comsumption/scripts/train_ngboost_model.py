# %%
import matplotlib.pyplot as plt
import numpy as np
from energy_consumption.metadata import (
    PATH_TO_METADATA,
    TARGET_COLUMN_NAME,
    TRAINING_COLUMNS,
    load_and_prepare_metadata,
)
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from sklearn.model_selection import train_test_split

# %%
# %%
prepared_metadata_df = load_and_prepare_metadata(PATH_TO_METADATA.resolve())

features = prepared_metadata_df[TRAINING_COLUMNS]
targets = prepared_metadata_df[TARGET_COLUMN_NAME]
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, random_state=42
)
# Split training set again for calibration
features_fit, features_cal, targets_fit, targets_cal = train_test_split(
    features_train, targets_train, test_size=0.25, random_state=42
)
model = NGBRegressor(
    Dist=LogNormal, n_estimators=500, learning_rate=0.03, verbose=False
)

model.fit(features_fit, targets_fit)

cal_dist = model.pred_dist(features_cal).dist
cal_actual = targets_cal.to_numpy()
# For each example, measure how far the real value is outside its range.
scores = np.maximum.reduce(
    (
        cal_dist.ppf(0.05) - cal_actual,
        cal_actual - cal_dist.ppf(0.95),
        np.zeros_like(cal_actual),
    )
)
# Find how much to widen the ranges to cover about 90% of examples.
rank = int(np.ceil((len(scores) + 1) * 0.90))
adjustment = np.sort(np.append(scores, np.inf))[rank - 1]

# Predict on the test set.
dist = model.pred_dist(features_test)

# Widen the ranges, but do not allow negative energyConsumption values.
mean = dist.dist.mean()
lower = np.maximum(0, dist.dist.ppf(0.05) - adjustment)
upper = dist.dist.ppf(0.95) + adjustment

# %%
actual = targets_test.to_numpy()
predicted = np.asarray(mean)

fig, (ax_scatter, ax_intervals) = plt.subplots(1, 2, figsize=(13, 5))

ax_scatter.scatter(actual, predicted, alpha=0.5)
limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
ax_scatter.plot(limits, limits, "k--")
ax_scatter.set(xlabel="Actual", ylabel="Predicted mean", title="Prediction accuracy")

order = np.argsort(predicted)
x = np.arange(len(order))
ax_intervals.fill_between(
    x, lower[order], upper[order], alpha=0.25, label="90% interval"
)
ax_intervals.plot(x, predicted[order], label="Predicted mean")
ax_intervals.scatter(x, actual[order], s=12, color="black", label="Actual")
ax_intervals.set(
    xlabel="Samples ordered by prediction",
    ylabel="Consumption",
    title="Predictive uncertainty",
)
ax_intervals.legend()

plt.tight_layout()
plt.show()

coverage = np.mean((actual >= lower) & (actual <= upper))
print(f"90% interval coverage: {coverage:.1%}")

# %%
