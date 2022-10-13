#!/bin/bash
set -e
set -u

export TIME_EXEC=10
export TIME_WAIT=5
export LEVELS_PACK='avalon_benchmark_perf_levels.tgz'

export GODOT_BIN='/usr/bin/godot-3.4.4-egl-mmap'
export GODOT_BIN='/usr/bin/godot'
export GODOT_SILENCE_ROOT_WARNING=1

export __GL_YIELD='NOTHING'
export __GL_SYNC_TO_VBLANK=0


if [[ ! -f ./project.godot ]]
then
    echo "missing project.godot"
    exit 1
fi

if [[ ! -f "./$LEVELS_PACK" ]]
then
    aws s3 cp "s3://avalon-benchmark/$LEVELS_PACK" "./$LEVELS_PACK"
    tar -xzvf "./$LEVELS_PACK"
fi

if [[ ! -f "${GODOT_BIN}" ]]
then
    echo "missing godot runtime"
    exit 2
fi


for CONFIG in basic fancy
do
for DRIVER in GLES2 GLES3
do
for NPROCS in 1 2 3 4 5
do
for NWORLD in 032 064 220 440
do

TAG="$CONFIG.$DRIVER.$NPROCS.$NWORLD"

ln -sf "./benchmark/settings/${CONFIG}.cfg" ./override.cfg

mkdir -p ./run_data/$TAG

ARGS=(
    --fixed-fps 10
    --config-file=res://benchmark/run.json
    --input-pipe-path="/dev/shm/godot/perf_$NWORLD.actions"
    --output-pipe-path="/dev/shm/godot/perf_$NWORLD.output"
)

for N in $(seq $NPROCS)
do
"${GODOT_BIN}" "${ARGS[@]}" --video-driver "$DRIVER" > ./run_data/$TAG/log.$N.txt 2> /dev/null &
done

echo
echo $TAG

sleep $TIME_EXEC
pkill -P $$
sleep $TIME_WAIT

tail -qn1 ./run_data/$TAG/log.?.txt
echo

done
done
done
done

rm -f ./override.cfg
