#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

BIRDS_DB_PATH="$(getFilePath 'birds.db')"

OLDIFS=$IFS
IFS=\|
most_recent_results=($(sqlite3 "$BIRDS_DB_PATH" \
  'SELECT Com_Name, Time, Date FROM detections
   ORDER BY Date DESC, Time DESC
   LIMIT 1'))
today=$(date +%F)
yesterday=$(date --date="yesterday" +%F)
two_days_ago=$(date --date="2 days ago" +%F)

echo -n The most recent detection was

if [[ "${most_recent_results[0]}" =~ ^[AEIOU].* ]];then
  echo -n " an ${most_recent_results[0]} at "
else
  echo -n " a ${most_recent_results[0]} at "
fi
most_recent_results[1]=$(date --date="${most_recent_results[1]}" +%l:%M%p)
echo -n ${most_recent_results[1]}
if [[ ${most_recent_results[2]} == $today ]];then
  echo " today."
elif [[ ${most_recent_results[2]} == $yesterday ]];then
  echo " yesterday."
elif [[ ${most_recent_results[2]} == $two_days_ago ]];then
  echo " two days ago."
else
  echo " on ${most_recent_results[2]}."
fi

IFS=$OLDIFS
