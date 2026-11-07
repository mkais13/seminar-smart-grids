"""Interactive python file to test training a prediction model."""

# %%
import matplotlib.pyplot as plt
import numpy as np
from energy_consumption.metadata import (
    PATH_TO_METADATA,
    TARGET_COLUMN_NAME,
    TRAINING_COLUMNS,
    load_and_prepare_metadata,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import HalvingRandomSearchCV, KFold, train_test_split

# %%
prepared_metadata_df = load_and_prepare_metadata(PATH_TO_METADATA.resolve())

features = prepared_metadata_df[TRAINING_COLUMNS]
targets = prepared_metadata_df[TARGET_COLUMN_NAME]
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets
)
model = HistGradientBoostingRegressor()
model.fit(features_train, targets_train)
predictions_test = model.predict(features_test)
mae = mean_absolute_error(targets_test, predictions_test)
mse = mean_squared_error(targets_test, predictions_test)
r2 = r2_score(targets_test, predictions_test)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

# %%
# Create unity plot of results
plt.scatter(x=targets_test, y=predictions_test)
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Unity line")
plt.title(
    f"Actual vs. Predicted Energy Consumption per trip. Total trips: {len(targets_test)}"
)
plt.xlabel("Actual Energy Consumption [kWh]")
plt.ylabel("Predicted Energy Consumption [kWh]")
plt.legend()
plt.grid()
plt.show()

# %%
# Train a quantile regressor to predict the possible range of energy consumption
alphas = [0.05, 0.5, 0.95]
quantile_models = {}
quantile_predictions = {}
quantile_metrics = {}
for alpha in alphas:
    model = HistGradientBoostingRegressor(loss="quantile", quantile=alpha)
    quantile_models[alpha] = model.fit(features_train, targets_train)
    quantile_predictions[alpha] = model.predict(features_test)
    quantile_metrics[alpha] = mean_pinball_loss(
        targets_test, quantile_predictions[alpha]
    )
print(quantile_metrics)
# %%
order = np.argsort(targets_test.to_numpy())
x_sorted = targets_test.to_numpy()[order]
lower = quantile_predictions[0.05][order]
upper = quantile_predictions[0.95][order]
mean = quantile_predictions[0.5][order]

plt.fill_between(
    x_sorted, lower, upper, color="orange", alpha=0.3, label="90% interval (5%–95%)"
)
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Unity line")
plt.scatter(x=x_sorted, y=mean, s=8, label="Mean prediction")
plt.title(
    f"Upper, Lower and Mean Energy Consumption per trip. Total trips: {len(order)}"
)
plt.xlabel("Actual Energy Consumption [kWh]")
plt.ylabel("Predicted Energy Consumption [kWh]")
plt.legend()
plt.grid()
plt.show()
# %%
# Since we want to predict the needed energy for the busses at different risk levels, we dont
# actually need to predict the whole interval. Predicting the upper limit of energy consumption
# is the conservative approach needed here.
upper_alphas = [0.8, 0.9, 0.95, 0.99]
quantile_models = {}
quantile_predictions = {}
quantile_metrics = {}
for upper_alpha in upper_alphas:
    model = HistGradientBoostingRegressor(loss="quantile", quantile=upper_alpha)
    quantile_models[upper_alpha] = model.fit(features_train, targets_train)
    quantile_predictions[upper_alpha] = model.predict(features_test)
    quantile_metrics[upper_alpha] = mean_pinball_loss(
        targets_test, quantile_predictions[upper_alpha]
    )
print(quantile_metrics)
# %%
order = np.argsort(targets_test.to_numpy())
x_sorted = targets_test.to_numpy()[order]

plt.figure()
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Unity line")
for upper_alpha in upper_alphas:
    plt.plot(
        x_sorted,
        quantile_predictions[upper_alpha][order],
        label=f"{upper_alpha:.0%} quantile",
        alpha=0.5,
    )
plt.title(f"Upper quantile predictions per trip. Total trips: {len(order)}")
plt.xlabel("Actual Energy Consumption [kWh]")
plt.ylabel("Predicted Energy Consumption [kWh]")
plt.legend()
plt.grid()
plt.show()
# %%
# Hyperparameter-tuning to find the best model setup for the different quantiles
search_results = {}
for upper_alpha in upper_alphas:
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20, 30, 50],
    }
    neg_mean_pinball_loss_05p_scorer = make_scorer(
        mean_pinball_loss,
        alpha=upper_alpha,
        greater_is_better=False,  # maximize the negative loss
    )
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=upper_alpha, random_state=42
    )
    search_results[upper_alpha] = HalvingRandomSearchCV(
        model,
        param_grid,
        resource="max_iter",
        min_resources=50,
        max_resources=500,
        scoring=neg_mean_pinball_loss_05p_scorer,
        n_jobs=2,
        random_state=42,
    ).fit(features_train, targets_train)
    print(
        f"Best parameters for alpha={upper_alpha}: {search_results[upper_alpha].best_params_}"
    )

# %%
# Hyperparameter-tuning shows that the best parameters are the same for all alphas,
# which makes it much easier. Training again with the best params to see performance.
best_hyperparameters = search_results[upper_alphas[0]].best_params_
quantile_models_opt = {}
quantile_predictions_opt = {}
quantile_metrics_opt = {}
for upper_alpha in upper_alphas:
    model = HistGradientBoostingRegressor(
        **best_hyperparameters,
        loss="quantile",
        quantile=upper_alpha,
    )
    quantile_models_opt[upper_alpha] = model.fit(features_train, targets_train)
    quantile_predictions_opt[upper_alpha] = model.predict(features_test)
    quantile_metrics_opt[upper_alpha] = mean_pinball_loss(
        targets_test, quantile_predictions_opt[upper_alpha]
    )
print(quantile_metrics_opt)
# %%
order = np.argsort(targets_test.to_numpy())
x_sorted = targets_test.to_numpy()[order]

plt.figure()
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Unity line")
for upper_alpha in upper_alphas:
    plt.plot(
        x_sorted,
        quantile_predictions_opt[upper_alpha][order],
        label=f"{upper_alpha:.0%} quantile",
        alpha=0.5,
    )
plt.title(
    f"Upper quantile predictions per trip, optimized hyperparams. Total_trips: {len(order)}"
)
plt.xlabel("Actual Energy Consumption [kWh]")
plt.ylabel("Predicted Energy Consumption [kWh]")
plt.legend()
plt.grid()
plt.show()

# %%
# With k-fold cross validation, we can test the model for all samples

predictions_kfold = {}
kfold = KFold(n_splits=5)
for train_indices, test_indices in kfold.split(features, targets):
    kfold_features_train = features.iloc[train_indices]
    kfold_features_test = features.iloc[test_indices]
    kfold_targets_train = targets.iloc[train_indices]
    kfold_targets_test = targets.iloc[test_indices]
    quantile_metrics_kfold = {}
    for upper_alpha in upper_alphas:
        model = HistGradientBoostingRegressor(
            **best_hyperparameters,
            loss="quantile",
            quantile=upper_alpha,
        )
        kfold_model = model.fit(kfold_features_train, kfold_targets_train)
        fold_predictions = kfold_model.predict(kfold_features_test)
        if predictions_kfold.get(upper_alpha) is None:
            predictions_kfold[upper_alpha] = np.empty(len(targets))
        predictions_kfold[upper_alpha][test_indices] = fold_predictions
        quantile_metrics_kfold[upper_alpha] = mean_pinball_loss(
            kfold_targets_test, fold_predictions
        )
    print(quantile_metrics_kfold)
# %%
order = np.argsort(targets.to_numpy())
x_sorted = targets.to_numpy()[order]

plt.figure()
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Unity line")
for upper_alpha in upper_alphas:
    plt.plot(
        x_sorted,
        predictions_kfold[upper_alpha][order],
        label=f"{upper_alpha:.0%} quantile",
        alpha=0.5,
    )
plt.title(
    f"Upper quantile predictions per trip, optimized hyperparams, k-folded with k=5. Total_trips: {len(order)}"
)
plt.xlabel("Actual Energy Consumption [kWh]")
plt.ylabel("Predicted Energy Consumption [kWh]")
plt.legend()
plt.grid()
plt.show()

# %%
