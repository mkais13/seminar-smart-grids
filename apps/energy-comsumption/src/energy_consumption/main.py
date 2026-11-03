import sklearn

from .metadata import PATH_TO_METADATA, load_and_prepare_metadata


def main():
    prepared_metadata_df = load_and_prepare_metadata(PATH_TO_METADATA.resolve())
    print(prepared_metadata_df.head(5))

    metadata_train, metadata_test = sklearn.model_selection.train_test_split(
        prepared_metadata_df
    )
    print(len(metadata_train))
    print(len(metadata_test))


if __name__ == "__main__":
    main()
