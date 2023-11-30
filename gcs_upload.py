import glob
import os
from google.cloud import storage

def upload_from_directory(dir_path:str, credentials_file:str, bucket_name:str, blob_name:str):
    """Uploads a directory to a given bucket using a prefix.
    Args:
        dir_path (str): Path to the directory to be uploaded.
        bucket_name (str): Name of the bucket to upload to.
        blob_name (str): Prefix to be used for uploading.
    """
    client = storage.Client.from_service_account_json(credentials_file)
    bucket = client.get_bucket(bucket_name)
    rel_paths = glob.glob(dir_path + "/**", recursive=True)
    for local_file in rel_paths:
        remote_path = f'{blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
        print(f"File {local_file} uploaded to {remote_path}.")

if __name__ == "__main__":
    bucket_name = "dtu_mlops_special"
    models_dir = "src/models/saved_models"
    credentials_file = "dtumlops-406109-3703b69ca83d.json"
    name = "model1"
    upload_from_directory(os.path.join(models_dir,name), credentials_file, bucket_name, 'saved_models')