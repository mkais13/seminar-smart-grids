# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
path_to_dataset = Path("./../../../dataset")

for file_index, file in enumerate(path_to_dataset.iterdir()):
    print(file.name)

print(f"Number of files: {file_index}")
# %%
# Load an example file
example_file = next(path_to_dataset.iterdir())
print(f"Loading file {example_file.name}")
example_df = pd.read_csv(example_file)
print(example_df.info())

# %%
# Check the time range of one file
example_df["time"] = pd.to_datetime(example_df["time_iso"])
min_time = example_df["time"].min()
max_time = example_df["time"].max()
print(f"timerange of example file: {min_time} - {max_time}")
print(f"duration: {max_time - min_time}")

# %%
# Plot the power demand over time for one file
example_df.plot(x="time", y="electric_powerDemand")
# %%
