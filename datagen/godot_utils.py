import os
from pathlib import Path

import boto3

from common.imports import tqdm
from common.log_utils import logger

GODOT_TEXTURES_PATH = Path(".") / "datagen" / "godot" / "Textures"


def download_godot_textures(download_path: Path = GODOT_TEXTURES_PATH):
    logger.info(f"Copying Godot textures from S3 to {download_path}")
    if not os.path.exists(str(download_path)):
        os.mkdir(download_path)

    # Create a new session to make client creation thread-safe.
    s3 = boto3.session.Session().client("s3")
    BUCKET_NAME = "godot-textures"
    bucket = boto3.resource("s3").Bucket(BUCKET_NAME)
    for s3_object in tqdm(list(bucket.objects.all())):
        path, filename = os.path.split(s3_object.key)
        filepath = download_path / filename
        if not filepath.exists():
            with open(filepath, "wb") as f:
                s3.download_fileobj(BUCKET_NAME, s3_object.key, f)
