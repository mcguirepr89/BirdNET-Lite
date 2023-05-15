#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

PROCESSED_DIR="$(getDirectory 'processed')"

source /etc/birdnet/birdnet.conf
set -x

cd "${PROCESSED_DIR}" || exit 1
empties=($(find ${PROCESSED_DIR} -size 57c))
for i in "${empties[@]}";do
  rm -f "${i}"
  rm -f "${i/.csv/}"
done

if [[ "$(find ${PROCESSED_DIR} | wc -l)" -ge 100 ]];then
  ls -1t . | tail -n +100 | xargs -r rm -vv
fi

#accumulated_files=$(find $RECS_DIR -path $PROCESSED_DIR -prune -o -path $EXTRACTED -prune -o -type f -print | wc -l)
#[ $accumulated_files -ge 10 ] && stop_core_services.sh
#echo "$(date "+%b  %e %I:%M:%S") Stopped Core Services -- It looks like analysis stopped. Check raw recordings in $RECS_DIR and check the birdnet_analysis.service and birdnet_server.service \"journalctl -eu birdnet_analysis -u birdnet_server\"" | sudo tee -a /var/log/syslog
