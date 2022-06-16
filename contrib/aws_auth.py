import subprocess
from typing import Optional
from typing import Tuple


def load_aws_keys() -> Tuple[Optional[str], Optional[str]]:
    result = subprocess.run("cat ~/.aws/credentials", shell=True, capture_output=True)
    is_relevant_section = False
    access_key = None
    secret_key = None
    for line in result.stdout.decode("UTF-8").splitlines():
        if is_relevant_section:
            if access_key is not None and secret_key is not None:
                break
            elif line.startswith("aws_access_key_id = "):
                access_key = line.split(" ")[-1].strip()
            elif line.startswith("aws_secret_access_key = "):
                secret_key = line.split(" ")[-1].strip()
            elif line.strip() == "":
                pass
            else:
                raise Exception(f"Bad line in credentials file: {line}")
        if "[computronium]" in line:
            is_relevant_section = True
    return access_key, secret_key
