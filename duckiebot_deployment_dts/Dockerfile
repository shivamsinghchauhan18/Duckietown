# Duckiebot RL Inference - DTS Compatible
# This uses the official Duckietown Software Stack (DTS) structure

ARG REPO_NAME="duckiebot-rl-inference"
ARG DESCRIPTION="RL model inference for Duckiebot using DTS"
ARG MAINTAINER="Your Name <your.email@example.com>"

# Use official Duckietown base image
ARG ARCH=arm64v8
ARG DISTRO=daffy
ARG BASE_TAG=${DISTRO}-${ARCH}
FROM duckietown/dt-ros-commons:${BASE_TAG}

# Set repository info
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
LABEL org.duckietown.label.module.description="${DESCRIPTION}" \
      org.duckietown.label.module.type="duckiebot" \
      org.duckietown.label.module.name="${REPO_NAME}" \
      org.duckietown.label.architecture="${ARCH}" \
      org.duckietown.label.code.location="/code" \
      org.duckietown.label.code.version.distro="${DISTRO}" \
      org.duckietown.label.base.image="duckietown/dt-ros-commons:${BASE_TAG}" \
      maintainer="${MAINTAINER}"

# Switch to root for installations
USER root

# Install Python dependencies for RL inference
RUN pip3 install --no-cache-dir \
    torch==1.9.0 \
    torchvision==0.10.0 \
    numpy>=1.19.0 \
    opencv-python-headless==4.5.3.56 \
    ray[rllib]==1.6.0 \
    gym==0.21.0 \
    pyyaml \
    scipy \
    scikit-image

# Copy source code
COPY . /code/

# Copy launch files
COPY ./launch/*.launch /code/launch/
COPY ./config/ /code/config/

# Set proper permissions
RUN chown -R duckie:duckie /code/
RUN chmod +x /code/launch_inference.sh

# Switch back to duckie user
USER duckie

# Set working directory
WORKDIR /code

# Set environment variables
ENV PYTHONPATH="/code:${PYTHONPATH}"
ENV ROS_PYTHON_VERSION=3

# Default command (DTS compatible)
CMD ["bash", "/code/launch_inference.sh"]