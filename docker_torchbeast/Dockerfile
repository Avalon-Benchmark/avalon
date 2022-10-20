FROM avalon/training

COPY ./docker_torchbeast/90_* /tmp/
RUN bash /tmp/90_apt_sources.sh
RUN bash /tmp/90_cuda_11.sh
RUN bash /tmp/90_cudnn_8.sh
RUN bash /tmp/90_torchbeast.sh
