# syntax=docker/dockerfile:1
FROM avalon/train

# Download world files
RUN mkdir -p /tmp/avalon_worlds/2f788115-ea32-4041-8cae-6e7cd33091b7/
WORKDIR /tmp/avalon_worlds/2f788115-ea32-4041-8cae-6e7cd33091b7
RUN wget https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz
RUN tar zxvf avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz
RUN rm avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz

# Copy a few worlds into minieval set (10 eat + 10 scramble tasks)
RUN mkdir -p /tmp/avalon_worlds/minieval
RUN for world in $(ls | grep "eat_" | head -n 10); do cp -r $world ../minieval/; done
RUN for world in $(ls | grep "scramble_" | head -n 10); do cp -r $world ../minieval/; done

WORKDIR /opt/projects/avalon/avalon/contest
COPY . .

ENV WANDB_MODE=dryrun

CMD python eval.py