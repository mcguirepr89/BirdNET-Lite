#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

EXTRACTED_DIR="$(getDirectory 'extracted')"
PROCESSED_DIR="$(getDirectory 'processed')"
RECORDINGS_DIR="$(getDirectory 'recordings_dir')"
disk_check_exclude_path=$(getFilePath 'disk_check_exclude.txt')

set -x
used="$(df -h / | tail -n1 | awk '{print $5}')"

if [ "${used//%}" -ge 95 ]; then
  source /etc/birdnet/birdnet.conf

  case $FULL_DISK in
    purge) echo "Removing oldest data"
        # shellcheck disable=SC2164
        cd ${EXTRACTED_DIR}/By_Date/
        curl localhost/views.php?view=Species%20Stats &>/dev/null
        if ! grep -qxFe \#\#start "${disk_check_exclude_path}"; then
            exit
        fi
        filestodelete=$(($(find ${EXTRACTED_DIR}/By_Date/* -type f | wc -l) / $(find ${EXTRACTED_DIR}/By_Date/* -maxdepth 0 -type d | wc -l)))
        iter=0
        for i in */*/*; do
            if [ $iter -ge $filestodelete ]; then
                break
            fi
            if ! grep -qxFe "$i" "${disk_check_exclude_path}"; then
                rm "$i"
            fi
            ((iter++))
        done
        find ${RECORDINGS_DIR} -type d -empty -mtime +90 -delete
        find ${EXTRACTED_DIR}/By_Date/ -empty -type d -delete;;

       #rm -drfv "$(find ${EXTRACTED_DIR}/By_Date/* -maxdepth 1 -type d -prune \
        # | sort -r | tail -n1)";;
    keep) echo "Stopping Core Services"
       /usr/local/bin/stop_core_services.sh;;
  esac
fi
sleep 1
if [ "${used//%}" -ge 95 ]; then
  case $FULL_DISK in
    purge) echo "Removing more data"
       rm -rfv ${PROCESSED_DIR}/*;;
    keep) echo "Stopping Core Services"
       /usr/local/bin/stop_core_services.sh;;
  esac
fi
