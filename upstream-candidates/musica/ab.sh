#!/bin/bash
# ab.sh: run a script on MUSICA against one of the two A/B builds (fork or pristine upstream master).
#
# Uses the musica-run skill's pipeline (~/.claude/skills/musica-run/musica-run.sh) with its own config:
# source tree ~/Applications/ngsuite/ab-src/<variant> (standalone clones pinned to one commit each, so
# other chats committing to ngsolve-src do not change the build), managed dir ~/musica-ab-<variant> on
# MUSICA, CPU only, UMFPACK on (remote-setup-ab.sh), no add-ons. Both variants get identical options.
#
#   ./ab.sh upstream smoke_ab.py --gpus 0 --cpus 4 --time 00:05:00   # first call: full build, 30-60 min
#   ./ab.sh fork sweep.py --gpus 0 --cpus 32 --time 03:00:00 --with ...
#
# Make the clones (once, or to move the pin):
#   git clone --no-checkout ~/Applications/ngsuite/ngsolve-src ab-src/<v>; git checkout <commit>;
#   point submodule.*.url at the local ngsolve-src submodules; git submodule update --init --recursive
set -eo pipefail
V=$1; shift
case "$V" in fork|upstream) ;; *) echo "usage: $0 fork|upstream script.py [musica-run options]"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
SKILL=$HOME/.claude/skills/musica-run
SRC=$HOME/Applications/ngsuite/ab-src/$V
[ -d "$SRC/.git" ] || { echo "missing clone $SRC"; exit 1; }
T=$(mktemp -d "${TMPDIR:-/tmp}/musica-ab-$V.XXXX")
cp "$SKILL/musica-run.sh" "$T/"
cp "$HERE/remote-setup-ab.sh" "$T/remote-setup.sh"
cat > "$T/config.sh" <<CFG
MUSICA_USER=                 # the ssh alias musica (~/.ssh/config) carries the user
HOST=musica
NGSOLVE_SRC=$SRC
PACKAGES=()
R_ROOT=musica-ab-$V
LOCAL_RUNS=$HOME/musica_runs/ab-$V
JOBS=8
CFG
exec "$T/musica-run.sh" "$@"
