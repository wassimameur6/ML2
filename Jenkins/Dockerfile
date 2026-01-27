FROM jenkins/jenkins:lts

USER root

# Install lsb-release
RUN apt-get update && apt-get install -y lsb-release

# Install Python3 and pip
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean

# Install Docker CLI and Docker Compose
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli docker-compose-plugin && \
    apt-get clean

USER jenkins

# Install Jenkins plugins
RUN jenkins-plugin-cli --plugins git workflow-aggregator docker-workflow credentials-binding