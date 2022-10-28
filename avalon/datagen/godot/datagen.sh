#!/bin/bash
set -e
set -u

export GODOT_THREADS='0'
export XDG_DATA_HOME='/tmp/xdg-godot'

export DISPLAY="${DISPLAY:-}"
export LANG='en_US.UTF-8'

RESOLUTION='96x96'

CHECK_ENV=''
CHILD_PID=''

THIS_FILE=$(realpath "$0")
THIS_PATH=$(dirname "${THIS_FILE}")

cd "$THIS_PATH"

if test -z "$GODOT_BINARY_PATH"
then
    # NOTE: Duplicated in interactive_godot_process.py
    GODOT_BINARY_PATH=$(realpath $THIS_FILE/../../bin/godot)
fi

# /usr/local/bin/godot
GODOT_CFG=()
GODOT_CMD=(
    "$GODOT_BINARY_PATH"
    '--audio-driver'    'Dummy'
    '--fixed-fps'       '10'
)


if [[ ($# == 0) || ($1 == -h) || ($1 == --help) ]]
then
cat << EOF
usage: $0
       [--thread_count[=N]]
       [--unbuffer_log]
       [--xorg_display[=N]]
       [--system_check]
        CONFIG
       [CONFIG]...
where:
     -T|--thread_count: thread mode for Godot, defaults to 4
     -U|--unbuffer_log: use line buffering for stdout/stderr
     -X|--xorg_display: use an existing display, don't start Xorg
     -Y|--system_check: run a system check only, don't start Godot
        CONFIG...: JSON input files to use
EOF
exit $(( $# == 0 ))
fi


while (( $# ))
do
    # split args like --foo=bawr
    IFS='=' read -r K V <<< "$1"
    case "$K"
    in
        -T|--thread_count)
            GODOT_THREADS="${V:-4}"
        ;;
        -U|--unbuffer_log)
            GODOT_CMD=('stdbuf' '-oL' '-eL' "${GODOT_CMD[@]}")
        ;;
        -X|--xorg_display)
            DISPLAY=":${V#:}"
        ;;
        -Y|--system_check)
            CHECK_ENV='1'
        ;;
        -D)
            GODOT_CMD+=('--debug')
        ;;
        -V)
            GODOT_CMD+=('--verbose')
        ;;
        --input_pipe_path)
            GODOT_CMD+=("--input-pipe-path=$V")
        ;;
        --output_pipe_path)
            GODOT_CMD+=("--output-pipe-path=$V")
        ;;
        --dev)
            GODOT_CMD+=("--dev")
        ;;
        --cuda-gpu-id)
            export EGL_CUDA_ID="$V"
        ;;
        --resolution)
            RESOLUTION="$V"
        ;;
        -*)
            echo "unknown argument: '$1'"
            exit 1
        ;;
        *)
            GODOT_CFG+=("--config-file=$(realpath -e -- $1)")
        ;;
    esac
    shift 1
done


if [[ "$CHECK_ENV" == '1' ]]
then
    if ! lspci -nn -d10de: -s.0 | grep -E '.+'
    then
        echo 'no GPUs are connected?'
        exit 100
    fi
    echo

    if ! nvidia-smi -L
    then
        echo 'nvidia-smi failed, are the nvidia drivers installed?'
        exit 101
    fi
    echo

fi


if [[ "$CHECK_ENV" != '0' ]]
then
    if ! realpath -e "$GODOT_BINARY_PATH"
    then
        echo "godot runtime not found at '$GODOT_BINARY_PATH'"
        exit 201
    fi
    echo

    if ! realpath -e 'project.godot'
    then
        echo "project.godot not found in '$THIS_PATH'"
        exit 202
    fi
    echo
fi

if [[ "${#GODOT_CFG[@]}" ]]
then
    exec "${GODOT_CMD[@]}" --resolution "$RESOLUTION" "${GODOT_CFG[@]}"
fi
