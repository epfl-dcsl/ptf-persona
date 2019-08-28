#!/usr/bin/env bash

set -e
set -u
set -o pipefail

interactive=false
export tensorflow_source=/home/whitlock/tf/barrier

print_help() {
    exit_code=$1
    cat <<EOF
Collect the stacktraces of all 'persona' commands.

  -i print out everything in interactive mode

If -i is not specified, this will save all pids to a file and return the local path where they can be found.
EOF
    exit $exit_code
}

parse_opts() {
    while getopts "hi" opt; do
        case "$opt" in
            i)
                echo "interactive set to true"
                interactive=true
                ;;
            h)
                print_help 0
                ;;
            \?)
                print_help 1
                ;;
        esac
    done
}

run_interactive() {
    pids=$(pgrep -u `whoami` -fa "python3 persona" | awk '{print $1}')
    for p in ${pids}; do
             echo "PID: $p"
             gdb -batch -p $p <<EOF
set pagination off
set interactive-mode off
directory $tensorflow_source
thread apply all backtrace
quit
EOF
    done
}

run_file_helper() {
    persona_pid=$1
    output_directory=$2
    gdb -p $persona_pid <<EOF >/dev/null
set pagination off
set interactive-mode off
directory $tensorflow_source
set logging file $output_directory/$(hostname)_${persona_pid}.txt
set logging on
thread apply all backtrace
set logging off
quit
EOF
}
export -f run_file_helper

run_file() {
    output_directory=$(mktemp -d --suffix=_persona_backtrace)
    pids=$(pgrep -u `whoami` -fa "python3 persona" | awk '{print $1}'A)
    parallel run_file_helper ::: $pids ::: "$output_directory"
    echo "$output_directory"
}

run_file_serial() {
    output_directory=$(mktemp -d --suffix=_persona_backtrace)
    pids=$(pgrep -u `whoami` -fa "python3 persona" | awk '{print $1}'A)
    set -x
    set -e
    for pid in $pids ; do
        run_file_helper $pid "$output_directory"
    done
    echo "$output_directory"
}

parse_opts "$@"

if [ "$interactive" = true ]; then
    run_interactive
else
    # run_file_serial
    run_file
fi
