from requests import Response

from common.slack_service.data_types import SlackResponse


class SlackException(Exception):
    pass


def make_slack_response(response: Response) -> SlackResponse:
    if response.status_code not in {200, 201, 202}:
        raise SlackException(f"Slack API bad status code: {response.status_code} | {str(response.content)}")
    data = response.json()
    if not data["ok"]:
        raise SlackException(response.content.decode("UTF-8"))
    return SlackResponse(message_timestamp=data.get("message", {}).get("ts"))


# run this to get a list of all the authors `git log --all --format='%aE' | sort -u` when we get new users
GIT_EMAIL_TO_SLACK_USER = {
    "abe@qandabe.com": "UT6BRF8AU",
    "abe@sourceress.co": "UT6BRF8AU",
    "bawr@hszm.pl": "USW925N0H",
    "bryden1995@gmail.com": "USW6LJU3B",
    "brydenfogelman@users.noreply.github.com": "USW6LJU3B",
    "hbogoevs@uwaterloo.ca": "USTHZD9DG",
    "hristijan.bogoevski@gmail.com": "USTHZD9DG",
    "jasoncbenn@gmail.com": "UT6BRBW9E",
    "thejash@gmail.com": "UT6BRCJU8",
    "zplizzi@users.noreply.github.com": "U0246KT4BJA",
    "zplizzi@gmail.com": "U0246KT4BJA",
}

NAME_TO_GIT_EMAIL = {
    "abe": "abe@sourceress.co",
    "bawr": "bawr@hszm.pl",
    "bryden": "bryden1995@gmail.com",
    "hristijan": "hbogoevs@uwaterloo.ca",
    "jason": "jasoncbenn@gmail.com",
    "josh": "thejash@gmail.com",
    "zack": "zplizzi@users.noreply.github.com",
}


def get_slack_user_mention(name: str):
    email = NAME_TO_GIT_EMAIL[name]
    return GIT_EMAIL_TO_SLACK_USER[email]
