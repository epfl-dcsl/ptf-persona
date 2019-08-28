#!/usr/bin/env bash

set -u
set -o pipefail
set -x

export remote_path='~/tf/shell'
output_dir=$(realpath .)
ssh_machines=""

print_help() {
    exit_code=$1
    cat <<EOF
Usage: $0 [-o] at least one ssh machine

Collect the stacktraces of all 'persona' commands on all the machines in the cluster.

  -o the directory to output the results. Must exist!
EOF
    exit $exit_code
}

parse_opts() {
    while getopts "ho:" opt; do
        case "$opt" in
            o)
                if ! [[ -e "$OPTARG" && -d "$OPTARG" ]]; then
                    echo "Please specify an existing directory for output. Got $OPTARG"
                    print_help 1
                fi
                output_dir=$(realpath "$OPTARG")
                ;;
            :)
                echo "Option '$OPTARG' requires an argument"
                print_help 1
                ;;
            h)
                print_help 0
                ;;
            \?)
                print_help 1
                ;;
        esac
    done
    shift $((OPTIND -1))
    ssh_machines="$@"
}

parse_opts "$@"

full_dir="$output_dir/backtraces"
if [ -e "$full_dir" ]; then
    rm -rf "$full_dir"
fi
mkdir "$full_dir"

remote_func() {
    remote_machine=$1
    output_dir=$2
    remote_dir=$(ssh $1 "$remote_path/util/record_backtraces.sh" | tail -n1)
    scp -r "${remote_machine}:${remote_dir}/" "$output_dir"
}
export -f remote_func

parallel remote_func ::: $ssh_machines ::: "$full_dir"
