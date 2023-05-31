FROM tensorflow/tensorflow:latest-gpu

RUN  apt-get update && apt-get install -y git

RUN pip install --no-cache-dir wandb==0.15.3

CMD ["/bin/bash"]