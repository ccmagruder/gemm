FROM ubuntu:24.04

ARG RUNNER_VERSION="2.321.0"

ARG DEBIAN_FRONTEND=noninteractive

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
  && mkdir -p /home/docker/actions-runner \
  && cd /home/docker/actions-runner \
  && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  && cd / \
  && chown -R docker ~docker && /home/docker/actions-runner/bin/installdependencies.sh

COPY start.sh start.sh
RUN chmod +x start.sh

USER docker

ENTRYPOINT ["./start.sh"]
