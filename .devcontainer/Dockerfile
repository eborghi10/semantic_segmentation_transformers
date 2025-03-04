FROM nvcr.io/nvidia/pytorch:24.12-py3

# Setup environment
ENV TERM=bash

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && rm -rf requirements.txt

# [Optional] Uncomment this section to install additional OS packages.
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
