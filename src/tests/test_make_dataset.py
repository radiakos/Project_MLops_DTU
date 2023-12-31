import os
from pathlib import Path
from transformers import AutoFeatureExtractor
import pytest

from src.data.make_dataset import FruitsDataset
from src.tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

mock_dataset = _TEST_ROOT + "/test_dataset"


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/processed/train"), reason="Image files not found in the folder!"
)
@pytest.fixture(scope="module")
def train_dataset():
    train_data = FruitsDataset(input_filepath=mock_dataset + "/processed/train", data_type="train", feature_extractor=AutoFeatureExtractor)
    return train_data


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/processed/valid"), reason="Image files not found in the folder!"
)
@pytest.fixture(scope="module")
def valid_dataset():
    valid_data = FruitsDataset(input_filepath=mock_dataset + "/processed/valid", data_type="valid", feature_extractor=AutoFeatureExtractor)
    return valid_data


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/processed/test"), reason="Image files not found in the folder!"
)
@pytest.fixture(scope="module")
def test_dataset():
    test_data = FruitsDataset(input_filepath=mock_dataset + "/processed/test", data_type="test", feature_extractor=AutoFeatureExtractor)
    return test_data


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/processed"), reason="Image files not found in the folder!"
)
def test_datasets_n_classes(train_dataset, valid_dataset, test_dataset):
    assert (
        train_dataset.num_classes >= test_dataset.num_classes
    ), "Test dataset has more classes than the training set!"

    assert (
        train_dataset.num_classes >= valid_dataset.num_classes
    ), "Validation dataset has more classes than the training set!"


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/processed"), reason="Image files not found in the folder!"
)
def test_datasets_classes_inclusion(train_dataset, valid_dataset, test_dataset):
    # Check to see that all the classes in test and valid sets are included in the train set

    train_classes = train_dataset.label2id
    train_classes_list = list(train_classes.keys())
    valid_classes = valid_dataset.label2id
    test_classes = test_dataset.label2id

    for el in valid_classes.keys():
        assert (
            el in train_classes_list
        ), f"{el} class from validation dataset does not exist in the training dataset!"

    for el in test_classes.keys():
        assert el in train_classes_list, f"{el} class from test dataset does not exist in the training dataset!"


@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/processed"), reason="Image files not found in the folder!"
)
def test_datasets_output_shapes(train_dataset, valid_dataset, test_dataset):
    train_dataset_instance = train_dataset
    valid_dataset_instance = valid_dataset
    test_dataset_instance = test_dataset

    processed_train_path = Path(mock_dataset) / "processed" / "train"
    processed_valid_path = Path(mock_dataset) / "processed" / "valid"
    processed_test_path = Path(mock_dataset) / "processed" / "test"

    n_training_images = len(list(processed_train_path.glob("**/*.jpg")))
    n_validation_images = len(list(processed_valid_path.glob("**/*.jpg")))
    n_testing_images = len(list(processed_test_path.glob("**/*.jpg")))

    assert (
        n_training_images > n_validation_images
    ), "There exist more images in the validation dataset than in the training dataset!"
    assert (
        n_training_images > n_testing_images
    ), "There exist more images in the test dataset than in the training dataset!"