#!/usr/bin/env bash
# Replace one Hugging Face Hub repo id with another in every tracked file.
#
#   scripts/rename_hub_repo.sh [--dry-run] OLD NEW
#   scripts/rename_hub_repo.sh maciek-wisniewski/molflash SFGLab/molflash
#
# Matches OLD as a fixed string, prints each changed line, and is safe to run twice: a second
# run finds nothing left to change. Uploading the kernel to the new repo is a separate step.
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
    echo "usage: $0 [--dry-run] OLD NEW" >&2
    exit 2
fi
old=$1
new=$2

cd "$(git rev-parse --show-toplevel)"

# Escape for a sed pattern (OLD) and a sed replacement (NEW), with | as the delimiter.
old_re=$(printf '%s' "$old" | sed 's/[][\\.*^$/|]/\\&/g')
new_re=$(printf '%s' "$new" | sed 's/[\\&|]/\\&/g')

mapfile -t files < <(git grep -l -F -e "$old" -- . ':!uv.lock' ':!scripts/rename_hub_repo.sh' || true)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "no tracked file mentions $old"
    exit 0
fi

for f in "${files[@]}"; do
    git grep -n -F -e "$old" -- "$f" | sed "s|^|  - |"
    if [[ $dry_run -eq 0 ]]; then
        sed -i "s|$old_re|$new_re|g" "$f"
        git grep -n -F -e "$new" -- "$f" | sed "s|^|  + |"
    fi
done

if [[ $dry_run -eq 1 ]]; then
    echo "dry run: ${#files[@]} file(s) would change"
else
    echo "changed ${#files[@]} file(s); review with git diff"
fi
