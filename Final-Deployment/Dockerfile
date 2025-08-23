# Simple Duckiebot RL Deployment
FROM duckietown/dt-ros-commons:daffy-arm64v8

# Install Python dependencies
USER root
RUN pip3 install --no-cache-dir \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache-dir \
    numpy==1.21.6 \
    opencv-python-headless==4.5.3.56 \
    pyyaml \
    rospkg

# Copy files
COPY champion_model.pth /data/models/champion_model.pth
COPY enhanced_config.yml /data/config/enhanced_config.yml
COPY rl_inference_node.py /code/rl_inference_node.py
COPY launch_rl.sh /code/launch_rl.sh

# Set permissions
RUN chmod +x /code/launch_rl.sh
RUN chown -R duckie:duckie /data /code

USER duckie
WORKDIR /code

CMD ["/code/launch_rl.sh"]