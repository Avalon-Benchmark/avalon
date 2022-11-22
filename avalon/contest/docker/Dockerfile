# syntax=docker/dockerfile:1
FROM avalon/train

# Download world files for minival and public test phases
RUN mkdir -p /tmp/avalon_worlds/minival/
WORKDIR /tmp/avalon_worlds/minival/
RUN wget https://avalon-benchmark.s3.us-west-2.amazonaws.com/contest/minival-20221117-03e70547d557.tar.gz
RUN tar zxvf minival-20221117-03e70547d557.tar.gz
RUN rm minival-20221117-03e70547d557.tar.gz

RUN mkdir -p /tmp/avalon_worlds/public_test/
WORKDIR /tmp/avalon_worlds/public_test/
RUN wget https://avalon-benchmark.s3.us-west-2.amazonaws.com/contest/public-test-20221117-e3d7079216f4.tar.gz
RUN tar zxvf public-test-20221117-e3d7079216f4.tar.gz
RUN rm public-test-20221117-e3d7079216f4.tar.gz

WORKDIR /opt/projects/avalon/avalon/contest
COPY . .

ENV WANDB_MODE=dryrun

CMD python eval.py