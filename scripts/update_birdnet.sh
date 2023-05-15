#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

SCRIPTS_DIR="$(getDirectory 'scripts')"
BIRDNET_PI_DIR="$(getDirectory 'birdnet_pi')"

# Update BirdNET-Pi's Git Repo
source /etc/birdnet/birdnet.conf
trap 'exit 1' SIGINT SIGHUP

usage() { echo "Usage: $0 [-r <remote name>] [-b <branch name>]" 1>&2; exit 1; }

# Defaults
remote="origin"
branch="main"

while getopts ":r:b:" o; do
  case "${o}" in
    r)
      remote=${OPTARG}
      git -C "$BIRDNET_PI_DIR" remote show $remote > /dev/null 2>&1
      ret_val=$?

      if [ $ret_val -ne 0 ]; then
        echo "Error: remote '$remote' not found. Add the upstream remote to your repository and try again."
        exit 1
      fi
      ;;
    b)
      branch=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

sudo_with_user () {
  set -x
  sudo -u $USER "$@"
  set +x
}

# Get current HEAD hash
commit_hash=$(sudo_with_user git -C "$BIRDNET_PI_DIR" rev-parse HEAD)

# Reset current HEAD to remove any local changes
sudo_with_user git -C "$BIRDNET_PI_DIR" reset --hard

# Fetches latest changes
sudo_with_user git -C "$BIRDNET_PI_DIR" fetch $remote $branch

# Switches git to specified branch
sudo_with_user git -C "$BIRDNET_PI_DIR" switch -C $branch --track $remote/$branch

# Prints out changes
sudo_with_user git -C "$BIRDNET_PI_DIR" diff --stat $commit_hash HEAD

sudo systemctl daemon-reload
sudo ln -sf $SCRIPTS_DIR/* /usr/local/bin/

# The script below handles changes to the host system
# Any additions to the updater should be placed in that file.
sudo $SCRIPTS_DIR/update_birdnet_snippets.sh
