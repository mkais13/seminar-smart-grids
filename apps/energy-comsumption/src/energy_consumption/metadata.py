import pathlib

import pandas as pd

PATH_TO_METADATA = pathlib.Path(__file__) / "../../../../../dataset/metaData.csv"
TARGET_COLUMN_NAME = "energyConsumption"  # Joule
RELEVANT_COLUMNS = [
    "busNumber",
    "startTime_iso",
    "endTime_iso",
    "drivenDistance",  # Meter
    "busRoute",
    "temperature_ambient_mean",  # Kelvin
    "itcs_numberOfPassengers_mean",
]


def load_and_prepare_metadata(path_to_metadata: pathlib.Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(PATH_TO_METADATA.resolve())
    filtered_metadata_df = metadata_df[[TARGET_COLUMN_NAME, *RELEVANT_COLUMNS]]
    filtered_metadata_df["startTime"] = pd.to_datetime(metadata_df["startTime_iso"])
    filtered_metadata_df["endTime"] = pd.to_datetime(metadata_df["endTime_iso"])
    # calculate total time spend on the road
    filtered_metadata_df["roadTime"] = (
        filtered_metadata_df["endTime"] - filtered_metadata_df["startTime"]
    )

    return filtered_metadata_df
