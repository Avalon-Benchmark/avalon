"""
Script to evaluate participant's submission and report to EvalAI.
Modified from: https://github.com/Cloud-CV/EvalAI-Starters/blob/master/remote_challenge_evaluation/evaluation_script_starter.py
"""
import base64
import json
import os
import re
import subprocess
import time
import urllib.request
from typing import Any
from typing import Dict

import boto3
import docker
from loguru import logger

from avalon.contest.test_runner.evalai_interface import EvalAI_Interface

# IDs (generated by EvalAI) to represent different challenge phases
PHASE_MINIVAL = 3747
PHASE_PUBLIC_TEST = 3748
PHASE_PRIVATE_TEST = 3773


def login_docker_client_to_aws_ecr() -> None:
    """
    Login to ECR, with a workaround for bug in docker SDK.
    See: https://stackoverflow.com/questions/53759374/docker-login-to-ecr-via-python-docker-sdk
    """
    ecr_client = boto3.client(
        "ecr",
        aws_access_key_id=os.getenv("EVALAI_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("EVALAI_AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("EVALAI_REGION_NAME"),
    )

    token = ecr_client.get_authorization_token()
    username, password = base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode().split(":")
    registry = token["authorizationData"][0]["proxyEndpoint"]

    command = 'docker login -u "%s" -p "%s" "%s"' % (username, password, registry)

    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True, bufsize=1)
    for line in iter(p.stdout.readline, b""):  # type: ignore
        logger.info(line)
    p.communicate()  # close p.stdout, wait for the subprocess to exit


def handle_submission(evalai: EvalAI_Interface, submission: Dict[str, Any]) -> None:
    input_file_url = submission["input_file"]

    with urllib.request.urlopen(input_file_url) as url:
        submitted_image_uri = json.load(url)["submitted_image_uri"]

    repository, tag = submitted_image_uri.split(":")
    repository = repository.replace("https://", "")

    logger.info("Retrieving image:", submitted_image_uri)
    logger.info("Repository:", repository)
    logger.info("Tag:", tag)

    login_docker_client_to_aws_ecr()

    docker_client = docker.from_env()
    submitted_image = docker_client.images.pull(repository=repository, tag=tag)

    # Update EvalAI right after sending the submission into "RUNNING" state
    evalai.update_submission_status({"submission": submission["id"], "submission_status": "RUNNING"})

    try:
        logger.info(f"Running submission: {submitted_image_uri}")

        if submission["challenge_phase"] == PHASE_MINIVAL:
            container_environment = {"FIXED_WORLDS_PATH": "/tmp/avalon_worlds/minival/"}
        elif submission["challenge_phase"] == PHASE_PUBLIC_TEST:
            container_environment = {"FIXED_WORLDS_PATH": "/tmp/avalon_worlds/public_test/"}
        else:
            # Todo: private challenge phase will need to be handled differently, since the data is not
            # available inside the container.
            raise Exception("Invalid challenge phase")

        container_logs = docker_client.containers.run(
            submitted_image,
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["compute", "utility", "graphics"]])],
            environment=container_environment,
        )

        logger.info(f"Finished running submission. Output: {container_logs}")

        # Parse submission score from container stdout
        regex_result = re.search(r"TEST RESULTS\\n({.*})", str(container_logs))
        if regex_result is None:
            raise Exception("Submission did not output any test results")

        result_json = regex_result.group(1)
        result_json = json.loads(result_json.replace("'", '"'))

        submission_data = [
            {
                "split": "default_split",
                "show_to_participant": True,
                "accuracies": {"Total": result_json["overall_success_rate"]},
            }
        ]

        # Update EvalAI after calculating final set of metrics and set submission status as "FINISHED"
        evalai.update_submission_data(
            {
                "challenge_phase": submission["challenge_phase"],
                "submission": submission["id"],
                "stdout": container_logs,
                "submission_status": "FINISHED",
                "result": json.dumps(submission_data),
            }
        )

        logger.info(f"Reported {submission['id']} to EvalAI: {submission_data}")

    except Exception as e:
        # Update EvalAI in case of errors and set submission status as "FAILED"
        evalai.update_submission_data(
            {
                "challenge_phase": submission["challenge_phase"],
                "submission": submission["id"],
                "submission_status": "FAILED",
            }
        )

        logger.error(f"Failed to run submitted image: {e}")
        raise


if __name__ == "__main__":
    auth_token = os.getenv("EVALAI_AUTH_TOKEN")
    evalai_api_server = "https://eval.ai"
    queue_name = "random-number-generator-challenge-1882-production-3364fa36-c6a1-47b5-a7b4-2e6400"
    challenge_pk = "1882"

    evalai = EvalAI_Interface(auth_token, evalai_api_server, queue_name, challenge_pk)

    while True:
        # Get the message from the queue
        message = evalai.get_message_from_sqs_queue()
        message_body = message.get("body")
        if message_body:
            submission_pk = message_body.get("submission_pk")
            challenge_pk = message_body.get("challenge_pk")
            phase_pk = message_body.get("phase_pk")
            # Get submission details -- This will contain the input file URL
            submission: Dict[str, Any] = evalai.get_submission_by_pk(submission_pk)

            if (
                submission.get("status") == "finished"
                or submission.get("status") == "failed"
                or submission.get("status") == "cancelled"
            ):
                message_receipt_handle = message.get("receipt_handle")
                evalai.delete_message_from_sqs_queue(message_receipt_handle)

            elif submission.get("status") == "running":
                # Do nothing on EvalAI
                pass

            else:
                handle_submission(evalai, submission)

        # Poll challenge queue for new submissions
        logger.info("Waiting for submission...")
        time.sleep(30)
