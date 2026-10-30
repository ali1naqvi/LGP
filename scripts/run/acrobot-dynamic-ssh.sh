#!/bin/bash
set -euo pipefail
# The remote checkout must contain this change and a rebuilt TPG executable.
# Remote login-shell startup must load the cluster's usual compiler/MPI modules.
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 [--dry-run] user@host /absolute/remote/LGP [last_generation]" >&2
    exit 2
fi
host=$1
repo=$2
generations=${3:-5000}
[[ $host != -* && $repo == /* && $generations =~ ^[0-9]+$ ]] || {
    echo 'Expected an SSH host, absolute repository path, and integer generation count' >&2; exit 2;
}
# Quote twice: once for the login-shell script and once for SSH's remote shell.
printf -v command 'cd %q && export TPG=%q && sbatch scripts/run/acrobot-dynamic.slurm %q' \
    "$repo" "$repo" "$generations"
printf -v remote 'bash -lc %q' "$command"
if (( dry_run )); then
    printf 'ssh %q %q\n' "$host" "$remote"
else
    ssh "$host" "$remote"
fi
