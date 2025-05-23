#!/bin/bash

REPO=$REPO
ACCESS_TOKEN=$(cat /run/secrets/token)

REG_TOKEN=$(curl -X POST -H "Authorization: token ${ACCESS_TOKEN}" -H "Accept: application/vnd.github+json" https://api.github.com/repos/${REPO}/actions/runners/registration-token | jq .token --raw-output)

cd /home/runner/actions-runner

export RUNNER_ALLOW_RUNASROOT=true
./config.sh --url https://github.com/${REPO} --token ${REG_TOKEN}

cleanup() {
  echo "Removing runner..."
  ./config.sh remove --unattended --token ${REG_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!
