# Select which NGSolve build Python imports, for the fork-vs-upstream A/B runs:
#   source env.sh fork       # ~/Applications/ngsuite/install-dev   (this fork, branch ngsolve)
#   source env.sh upstream   # ~/Applications/ngsuite/upstream-ref/install (pristine origin/master)
# Both builds have the same CMake options (RelWithDebInfo, no GUI, OCC, UMFPACK, Accelerate).
case "$1" in
  fork)     NGP="$HOME/Applications/ngsuite/install-dev" ;;
  upstream) NGP="$HOME/Applications/ngsuite/upstream-ref/install" ;;
  *) echo "usage: source env.sh fork|upstream" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export NGS_VARIANT="$1"
export PYTHONPATH="$NGP/Contents/Resources/lib/python3.13/site-packages"
export PATH="$NGP/Contents/MacOS:$PATH"
export NETGENDIR="$NGP/Contents/MacOS"
# DYLD_LIBRARY_PATH (set to Netgen.app by ~/.zprofile) beats the rpaths: point it at this build only
export DYLD_LIBRARY_PATH="$NGP/Contents/MacOS"
export NGSJIT_CACHE_DIR="$HOME/.cache/ngsjit-ab-$1"
