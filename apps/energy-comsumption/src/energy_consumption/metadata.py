import pathlib

import pandas as pd

PATH_TO_METADATA = pathlib.Path(__file__) / "../../../../../dataset/metaData.csv"
TARGET_COLUMN_NAME = "energyConsumption"  # Joule
TIME_COLUMNS = [
    "startTime_iso",
    "endTime_iso",
]
RELEVANT_COLUMNS = [
    "busNumber",
    "drivenDistance",  # Meter
    "busRoute",
    "temperature_ambient_mean",  # Kelvin
    "itcs_numberOfPassengers_mean",
    "status_gridIsAvailable_mean",
]


def load_and_prepare_metadata(path_to_metadata: pathlib.Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(PATH_TO_METADATA.resolve())
    filtered_metadata_df = metadata_df[
        [TARGET_COLUMN_NAME, *TIME_COLUMNS, *RELEVANT_COLUMNS]
    ]
    # calculate total time spend on the road
    filtered_metadata_df["roadTime"] = pd.to_datetime(
        metadata_df["startTime_iso"]
    ) - pd.to_datetime(metadata_df["endTime_iso"])
    filtered_metadata_df["roadTime"] = filtered_metadata_df[
        "roadTime"
    ].dt.total_seconds()
    filtered_metadata_df = filtered_metadata_df.drop(TIME_COLUMNS, axis=1)
    # turn categorical features busNumber and busRoute into numbers
    filtered_metadata_df["busNumber"] = pd.Categorical(
        filtered_metadata_df["busNumber"]
    ).codes
    filtered_metadata_df["busRoute"] = pd.Categorical(
        filtered_metadata_df["busRoute"]
    ).codes
    # convert energyConsumption to kWh
    filtered_metadata_df["energyConsumption"] = (
        filtered_metadata_df["energyConsumption"] / 3_600_000
    )
    return filtered_metadata_df
