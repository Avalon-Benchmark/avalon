import os
import tarfile
from pathlib import Path
from typing import Optional
from typing import cast

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from avalon.contrib.aws_auth import load_aws_keys

REGION = "us-west-2"
TEMP_BUCKET_NAME = "untitled-ai-temp"


class SimpleS3Client:
    def __init__(self, bucket_name: str = TEMP_BUCKET_NAME, region: str = REGION) -> None:
        self.bucket_name = bucket_name
        self.region = region
        access_key: Optional[str] = None
        secret_key: Optional[str] = None
        if "AWS_ACCESS_KEY_ID" in os.environ:
            access_key, secret_key = os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
        else:
            access_key, secret_key = load_aws_keys()

        # Allow anonymous access, e.g. when the bucket is public
        config = None
        if access_key is None and secret_key is None:
            config = Config(signature_version=UNSIGNED)

        # Create a new session to make client creation thread-safe.
        self.client = boto3.session.Session().client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config,
        )

    def save(self, key: str, data: bytes) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
        )

    def load(self, key: str) -> bytes:
        try:
            data = self.client.get_object(Bucket=self.bucket_name, Key=key)["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(key) from e
            else:
                raise
        else:
            return cast(bytes, data)  # type: ignore

    def download_to_file(self, key: str, output_path: Path) -> None:
        self.client.download_file(self.bucket_name, key, str(output_path.absolute()))

    def upload_from_file(self, input_path: Path, key: str, is_overwrite_allowed: bool = False) -> None:
        # TASK 20097119-dfe7-4abb-bdb1-3895db6742e5: remove overwrite ability from S3
        assert is_overwrite_allowed or not self.is_present(
            key
        ), f"Tried to upload to {key} but that path already exists in bucket {self.bucket_name}. Go delete that file manually (or more likely realize you did something wrong and fix that). Technically the file you tried to upload was {str(input_path)}"
        self.client.upload_file(str(input_path.absolute()), self.bucket_name, key)

    def copy_object(self, src: str, dst: str, is_overwrite_allowed: bool = False) -> None:
        # TASK 20097119-dfe7-4abb-bdb1-3895db6742e5: remove overwrite ability from S3
        assert is_overwrite_allowed or not self.is_present(
            dst
        ), f"Tried to copy s3:{src} to s3:{dst} but the destination already exists in bucket {self.bucket_name}."

        self.client.copy_object(
            CopySource={
                "Bucket": self.bucket_name,
                "Key": src,
            },
            Bucket=self.bucket_name,
            Key=dst,
        )

    def is_present(self, path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=path)
        except ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                return False
            raise
        else:
            return True

    def make_bucket(self) -> None:
        self.client.create_bucket(
            Bucket=self.bucket_name,
            CreateBucketConfiguration={"LocationConstraint": self.region},  # type: ignore[typeddict-item]
        )


def download_tar_from_s3_and_unpack(tar_path: Path, client: SimpleS3Client) -> Path:
    client.download_to_file(key=tar_path.name, output_path=tar_path)
    tar = tarfile.open(tar_path)
    tar.extractall(path=tar_path.parent)
    tar.close()
    tar_path.unlink()
    return tar_path.parent
