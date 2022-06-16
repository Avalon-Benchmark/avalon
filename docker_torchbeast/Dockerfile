# syntax = docker/dockerfile:1.2

FROM ubuntu:20.04

WORKDIR /tmp

CMD ["/main.sh"]
EXPOSE 8888

COPY ./docker/00_* /tmp/
RUN bash 00_apt_install.sh

COPY ./docker/01_* /tmp/
RUN bash 01_user_config.sh

COPY ./docker/99_container_main.sh /main.sh
RUN chmod ugo+x /main.sh



COPY ./docker/20_* /tmp/
RUN bash 20_pip_torch.sh

COPY ./docker/21_* /tmp/
RUN  bash 21_pip_install.sh

COPY ./docker/40_* /tmp/
RUN  bash 40_apt_extras.sh

COPY ./docker/41_* /tmp/
RUN  bash 41_pip_extras.sh

COPY ./docker/50_* /tmp/
RUN  bash 50_godot.sh

COPY ./docker/80_* /tmp/
RUN  bash 80_temporary_fixes.sh

COPY ./docker_torchbeast/90_* /tmp/
RUN  bash 90_apt_sources.sh
RUN  bash 90_cuda_11.sh
RUN  bash 90_cudnn_8.sh
RUN  bash 90_torchbeast.sh


RUN mkdir -p /opt/projects/avalon
RUN chown -R user:user /opt/projects

WORKDIR /opt/projects/avalon
USER user
