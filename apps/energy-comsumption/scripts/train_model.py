"""Interactive python file to test training a prediction model."""

# %%
import matplotlib.pyplot as plt
import numpy as np
from energy_consumption.metadata import (
    PATH_TO_METADATA,
    RELEVANT_COLUMNS,
    TARGET_COLUMN_NAME,
    load_and_prepare_metadata,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# %%
prepared_metadata_df = load_and_prepare_metadata(PATH_TO_METADATA.resolve())

features = prepared_metadata_df[RELEVANT_COLUMNS]
targets = prepared_metadata_df[TARGET_COLUMN_NAME]
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets
)
print(features_train.head())
print(targets_train.head())
model = HistGradientBoostingRegressor()
model.fit(features_train, targets_train)
predictions_test = model.predict(features_test)
print(predictions_test)
mae = mean_absolute_error(targets_test, predictions_test)
mse = mean_squared_error(targets_test, predictions_test)

print(f"MAE: {mae}")
print(f"MSE: {mse}")

# %%
# Create unity plot of results
plt.scatter(x=targets_test, y=predictions_test)
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Unity line")
plt.title("Actual vs. Predicted Energy Consumption per trip")
plt.xlabel("Actual Energy Consumption [kWh]")
plt.ylabel("Predicted Energy Consumption [kWh]")
plt.legend()
plt.grid()
plt.show()

# %%
