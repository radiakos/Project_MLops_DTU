import os
from pathlib import Path
import pytest

from PIL import Image
from src.data.data_cleaning import CleanData
from src.tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

mock_dataset = _TEST_ROOT + "/test_dataset"

HEIGHT= 128
WIDTH = 128

external_good_path = Path(mock_dataset) / "external" / "fruit_images/good_quality_fruits"
external_bad_path = Path(mock_dataset) / "external" / "fruit_images/bad_quality_fruits"
external_mixed_path = Path(mock_dataset) / "external" / "fruit_images/mixed_quality_fruits"

processed_train_path = Path(mock_dataset) / "processed" / "train"
processed_valid_path = Path(mock_dataset) / "processed" / "valid"
processed_test_path = Path(mock_dataset) / "processed" / "test"

@pytest.mark.skipif(
    not os.path.exists(mock_dataset + "/external"), reason="Image files not found in the folder!"
)
@pytest.fixture(scope="module")
def data_cleaning():
    data_cleaning_instance = CleanData(data_dir=mock_dataset + "/fruit_images", train_dir=mock_dataset + "/processed/train", 
                                       valid_dir=mock_dataset + "/processed/valid", test_dir=mock_dataset + "/processed/test",
                                       external_dir=mock_dataset + "/external")
    return data_cleaning_instance

def test_data_cleaning(data_cleaning):
    data = data_cleaning.create_df()
    assert data.notnull().all().all(), "Initial DataFrame contains null values"
    assert os.path.exists(external_good_path), f"Folder not found: {external_good_path}. Rename failed!"
    assert os.path.exists(external_bad_path), f"Folder not found: {external_bad_path}. Rename failed!"
    assert os.path.exists(external_mixed_path), f"Folder not found: {external_mixed_path}. Rename failed!"

    fruits = ["Apple", "Banana", "Lime", "Guava", "Orange", "Pomegranate"]

    data_cleaning.create_train_valid_test_folders(fruits)

    data_cleaning.df_information(df_=data)
    train_df, valid_df, test_df = data_cleaning.split_df_to_train_and_test(
        df_=data,
        test_valid_size=0.2,
        test_valid_split=0.5,
    )
    assert train_df.notnull().all().all(), "Train DataFrame contains null values"
    assert valid_df.notnull().all().all(), "Validation DataFrame contains null values"
    assert test_df.notnull().all().all(), "Test DataFrame contains null values"


    trimmed_train_df = data_cleaning.df_trim(
        df_=train_df, desired_samples_per_class=15
    )
    data_cleaning.df_information(df_=trimmed_train_df)

    df_balanced = data_cleaning.df_balance(
        df_=trimmed_train_df,
        desired_samples_per_class=15,
        flag=False,
    )
    assert df_balanced.notnull().all().all(), "Balanced DataFrame contains null values"

    df_balanced_resized = data_cleaning.resize_images_in_df(
        df_=df_balanced,
        save_folder=data_cleaning.train_dir,
        width=WIDTH,
        height=HEIGHT,
        flag=False,
    )
    assert df_balanced_resized.notnull().all().all(), "Balanced train DataFrame contains null values"

    df_valid_resized = data_cleaning.resize_images_in_df(
        df_=valid_df,
        save_folder=data_cleaning.valid_dir,
        width=WIDTH,
        height=HEIGHT,
        flag=False,
    )
    assert df_valid_resized.notnull().all().all(), "Balanced validation DataFrame contains null values"

    df_test_resized = data_cleaning.resize_images_in_df(
        df_=test_df,
        save_folder=data_cleaning.test_dir,
        width=WIDTH,
        height=HEIGHT,
        flag=False,
    )

    assert df_test_resized.notnull().all().all(), "Balanced Test DataFrame contains null values"

    assert (
        len(df_balanced_resized) > len(df_valid_resized)
    ), "There exist more images in the validation dataset than in the training dataset!"
    assert (
        len(df_balanced_resized) > len(df_test_resized)
    ), "There exist more images in the test dataset than in the training dataset!"

    assert os.path.exists(processed_train_path), f"Folder not found: {processed_train_path}"
    assert os.path.exists(processed_valid_path), f"Folder not found: {processed_valid_path}"
    assert os.path.exists(processed_test_path), f"Folder not found: {processed_test_path}"

    n_training_images = len(list(processed_train_path.glob("**/*.jpg")))
    n_validation_images = len(list(processed_valid_path.glob("**/*.jpg")))
    n_testing_images = len(list(processed_test_path.glob("**/*.jpg")))

    assert (n_training_images > 0 ), "Training folder contains no images!"

    assert (n_validation_images > 0 ), "Validation folder contains no images!"

    assert (n_testing_images > 0 ), "Testing folder contains no images!"


test_image = "src/tests/test_dataset/processed/test/Apple/IMG_8198.JPG"
train_image = "src/tests/test_dataset/processed/train/Orange/IMG_1878.JPG"
valid_image = "src/tests/test_dataset/processed/valid/Pomegranate/20190820_143524_22900.jpg"

@pytest.mark.parametrize("image_path", [test_image, train_image, valid_image])
def test_image_dimensions(image_path):

    assert os.path.exists(image_path), f"Image not found: {image_path}"

    with Image.open(image_path) as img:
        # Get the width and height of the image
        width, height = img.size

        # Your assertions on width and height
        assert width == WIDTH, f"Width mismatch. Expected: {WIDTH}, Actual: {width}"
        assert height == HEIGHT, f"Height mismatch. Expected: {HEIGHT}, Actual: {height}"

