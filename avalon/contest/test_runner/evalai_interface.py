""""
Client interface for EvalAI.
Copied mostly without modification from:
https://github.com/Cloud-CV/EvalAI-Starters/blob/master/remote_challenge_evaluation/evaluation_script_starter.py
"""
import requests
from loguru import logger

URLS = {
    "get_message_from_sqs_queue": "/api/jobs/challenge/queues/{}/",
    "get_submission_by_pk": "/api/jobs/submission/{}",
    "delete_message_from_sqs_queue": "/api/jobs/queues/{}/",
    "update_submission": "/api/jobs/challenge/{}/update_submission/",
}


class EvalAI_Interface:
    def __init__(self, AUTH_TOKEN, EVALAI_API_SERVER, QUEUE_NAME, CHALLENGE_PK) -> None:  # type: ignore
        """Class to initiate call to EvalAI backend

        Arguments:
            AUTH_TOKEN {[string]} -- The authentication token corresponding to EvalAI
            EVALAI_API_SERVER {[string]} -- It should be set to https://eval.ai # For production server
            QUEUE_NAME {[string]} -- Unique queue name corresponding to every challenge
            CHALLENGE_PK {[integer]} -- Primary key corresponding to a challenge
        """

        self.AUTH_TOKEN = AUTH_TOKEN
        self.EVALAI_API_SERVER = EVALAI_API_SERVER
        self.QUEUE_NAME = QUEUE_NAME
        self.CHALLENGE_PK = CHALLENGE_PK

    def get_request_headers(self):  # type: ignore
        """Function to get the header of the EvalAI request in proper format

        Returns:
            [dict]: Authorization header
        """
        headers = {"Authorization": "Bearer {}".format(self.AUTH_TOKEN)}
        return headers

    def make_request(self, url, method, data=None):  # type: ignore
        """Function to make request to EvalAI interface

        Args:
            url ([str]): URL of the request
            method ([str]): Method of the request
            data ([dict], optional): Data of the request. Defaults to None.

        Returns:
            [JSON]: JSON response data
        """
        headers = self.get_request_headers()
        try:
            response = requests.request(method=method, url=url, headers=headers, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            logger.info("The server isn't able establish connection with EvalAI")
            raise
        return response.json()

    def return_url_per_environment(self, url):  # type: ignore
        """Function to get the URL for API

        Args:
            url ([str]): API endpoint url to which the request is to be made

        Returns:
            [str]: API endpoint url with EvalAI base url attached
        """
        base_url = "{0}".format(self.EVALAI_API_SERVER)
        url = "{0}{1}".format(base_url, url)
        return url

    def get_message_from_sqs_queue(self):  # type: ignore
        """Function to get the message from SQS Queue

        Docs: https://eval.ai/api/docs/#operation/get_submission_message_from_queue

        Returns:
            [JSON]: JSON response data
        """
        url = URLS.get("get_message_from_sqs_queue").format(self.QUEUE_NAME)
        url = self.return_url_per_environment(url)
        response = self.make_request(url, "GET")
        return response

    def delete_message_from_sqs_queue(self, receipt_handle):  # type: ignore
        """Function to delete the submission message from the queue

        Docs: https://eval.ai/api/docs/#operation/delete_submission_message_from_queue

        Args:
            receipt_handle ([str]): Receipt handle of the message to be deleted

        Returns:
            [JSON]: JSON response data
        """
        url = URLS.get("delete_message_from_sqs_queue").format(self.QUEUE_NAME)
        url = self.return_url_per_environment(url)
        data = {"receipt_handle": receipt_handle}
        response = self.make_request(url, "POST", data)
        return response

    def update_submission_data(self, data):  # type: ignore
        """Function to update the submission data on EvalAI

        Docs: https://eval.ai/api/docs/#operation/update_submission

        Args:
            data ([dict]): Data to be updated

        Returns:
            [JSON]: JSON response data
        """
        url = URLS.get("update_submission").format(self.CHALLENGE_PK)
        url = self.return_url_per_environment(url)
        response = self.make_request(url, "PUT", data=data)
        return response

    def update_submission_status(self, data):  # type: ignore
        """

        Docs: https://eval.ai/api/docs/#operation/update_submission

        Args:
            data ([dict]): Data to be updated

        Returns:
            [JSON]: JSON response data
        """
        url = URLS.get("update_submission").format(self.CHALLENGE_PK)
        url = self.return_url_per_environment(url)
        response = self.make_request(url, "PATCH", data=data)
        return response

    def get_submission_by_pk(self, submission_pk):  # type: ignore
        url = URLS.get("get_submission_by_pk").format(submission_pk)
        url = self.return_url_per_environment(url)
        response = self.make_request(url, "GET")
        return response
