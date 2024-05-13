import os
import subprocess
from pathlib import Path
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth


LEADERBOARD_URL = (
    "https://leaderboard-brewyonder.eu.live.external.example.com/add-user-score"
)


def post_score_to_api(score: float, user: str):
    if not user:
        git_info = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%H, %an"]
        ).decode()
        _, user = git_info.split(",")

    print(f"Sending data to leaderboard: User: '{user}', score: '{score}'")

    # authentication
    username = os.environ["LEADERBOARD_API_USERNAME"]
    password = os.environ["LEADERBOARD_API_PASSWORD"]

    r = requests.post(
        url=LEADERBOARD_URL,
        json={"user": user, "score": score},
        headers={"content-type": "application/json"},
        auth=HTTPBasicAuth(username, password),
        timeout=30,
    )

    if not r.ok:
        raise Exception(
            f"Updating score was not successful and failed with {r.status_code}"
        )
    print("Submitted successfully.")


def write_result_to_file(score: float, filename: str):
    with open(filename, "w") as f:
        f.write(str(score))
