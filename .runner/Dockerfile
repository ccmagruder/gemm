FROM ccmagruder/devcontainer-gemm:latest
# FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04
# FROM ubuntu:24.04

ARG RUNNER_VERSION="2.321.0"

ARG DEBIAN_FRONTEND=noninteractive

COPY start.sh start.sh

RUN apt update -y && apt upgrade -y \
  && useradd -m docker \
  && apt install -y --no-install-recommends \
    build-essential \
    curl \
    jq \
    libffi-dev \
    libicu-dev \
    libssl-dev \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /home/docker/actions-runner \
  && cd /home/docker/actions-runner \
  && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  && cd / \
  && chown -R docker ~docker \
  && /home/docker/actions-runner/bin/installdependencies.sh \
  && chmod +x /start.sh

USER docker

CMD ["/start.sh"]
