# FROM ccmagruder/devcontainer-gemm:latest
FROM nestybox/ubuntu-noble-docker

ARG RUNNER_VERSION="2.321.0"

ARG DEBIAN_FRONTEND=noninteractive

COPY start.sh /start.sh

RUN apt update -y && \
    # useradd -m runner && \
    apt install -y --no-install-recommends \
        build-essential \
        curl \
        jq \
        libffi-dev \
        libicu-dev \
        libssl-dev \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /home/runner/actions-runner && \
    cd /home/runner/actions-runner && \
    curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    cd / && \
    # chown -R runner ~runner && \
    /home/runner/actions-runner/bin/installdependencies.sh && \
    chmod +x /start.sh && \
    # Install nvidia-container-toolkit
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt update -y && \
    apt install -y nvidia-container-toolkit && \
    nvidia-ctk runtime configure --runtime=docker && \
    # https://stackoverflow.com/questions/77051751/unable-to-run-nvidia-gpu-enabled-docker-containers-inside-an-lxc-container
    sed -i -e 's/#no-cgroups = false/no-cgroups = true/' /etc/nvidia-container-runtime/config.toml && \
    # https://forums.docker.com/t/ulimit-open-files-cannot-modify-limit-operation-not-permitted/8230
    sed -i '/ulimit -Hn 524288/d' /etc/init.d/docker
    

CMD service docker start && /start.sh
