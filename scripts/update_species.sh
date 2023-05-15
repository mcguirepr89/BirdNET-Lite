#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

birds_db_path="$(getFilePath 'birds.db')"
# Replace the home path (which is the user executing) with the home path for Birdnet
id_file_path="$(getFilePath 'IdentifiedSoFar.txt')"

# Update the species list
#set -x
source /etc/birdnet/birdnet.conf
if [ -f "$birds_db_path" ];then
sqlite3 "$birds_db_path" "SELECT DISTINCT(Com_Name) FROM detections" | sort >  ${id_file_path}
fi
