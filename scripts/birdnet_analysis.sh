#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

SCRIPTS_DIR="$(getDirectory 'scripts')"
RECORDING_DIR="$(getDirectory 'recs_dir')"
STREAMDATA_DIR="$(getDirectory 'stream_data')"
birdnet_conf_path="$(getFilePath 'etc_birdnet.conf')"

# Runs BirdNET-Lite
#set -x
source /etc/birdnet/birdnet.conf
# Document this run's birdnet.conf settings
# Make a temporary file to compare the current birdnet.conf with
# the birdnet.conf as it was the last time this script was called
if [ -z ${THIS_RUN} ];then THIS_RUN="$(getFilePath 'thisrun.txt')";fi
[ -f ${THIS_RUN} ] || touch ${THIS_RUN} && chmod g+w ${THIS_RUN}
if [ -z ${LAST_RUN} ];then LAST_RUN="$(getFilePath 'lastrun.txt')";fi
[ -z ${LATITUDE} ] && echo "LATITUDE not set, exiting 1" && exit 1
[ -z ${LONGITUDE} ] && echo "LONGITUDE not set, exiting 1" && exit 1
make_thisrun() {
  sleep .4
  awk '!/#/ && !/^$/ {print}' "${birdnet_conf_path}" \
    > >(tee "${THIS_RUN}")
  sleep .5
}
make_thisrun &> /dev/null
if ! diff ${LAST_RUN} ${THIS_RUN};then
  echo "The birdnet.conf file has changed"
  if grep REC <(diff $LAST_RUN $THIS_RUN);then
    echo "Recording element changed -- restarting 'birdnet_recording.service'"
    sudo systemctl stop birdnet_recording.service
    sudo rm -rf ${RECORDING_DIR}/$(date +%B-%Y/%d-%A)/*
    sudo systemctl start birdnet_recording.service
  fi
  cat ${THIS_RUN} > ${LAST_RUN}
fi

INCLUDE_LIST="$(getFilePath 'include_species_list.txt')"
EXCLUDE_LIST="$(getFilePath 'exclude_species_list.txt')"
if [ ! -f ${INCLUDE_LIST} ];then
  touch ${INCLUDE_LIST} &&
    chmod g+rw ${INCLUDE_LIST}
fi
if [ ! -f ${EXCLUDE_LIST} ];then
  touch ${EXCLUDE_LIST} &&
    chmod g+rw ${EXCLUDE_LIST}
fi
if [ "$(du ${INCLUDE_LIST} | awk '{print $1}')" -lt 4 ];then
	INCLUDE_LIST=null
fi
if [ "$(du ${EXCLUDE_LIST} | awk '{print $1}')" -lt 4 ];then
	EXCLUDE_LIST=null
fi

# Create an array of the audio files
# Takes one argument:
#   - {DIRECTORY}
get_files() {
  files=($( find ${1} -maxdepth 1 -name '*wav' ! -size 0\
  | sort \
  | awk -F "/" '{print $NF}' ))
  [ -n "${files[1]}" ] && echo "Files loaded"
}

# Move all files that have been analyzed already into newly created "Analyzed"
# directory
# Takes one argument:
#   - {DIRECTORY}
move_analyzed() {
  for i in "${files[@]}";do
    j="${i}.csv"
    if [ -f "${1}/${j}" ];then
      if [ ! -d "${1}-Analyzed" ];then
        mkdir -p "${1}-Analyzed" && echo "'Analyzed' directory created"
      fi
      mv "${1}/${i}" "${1}-Analyzed/"
      mv "${1}/${j}" "${1}-Analyzed/"
    fi
  done
}

# Run BirdNET-Lite on the WAVE files from get_files()
# Uses one argument:
#   - {DIRECTORY}
run_analysis() {
  PYTHON_VIRTUAL_ENV="$(getFilePath 'python3')"
  DIR="$(getDirectory 'scripts')"
  analyzing_now_path="$(getFilePath 'analyzing_now.txt')"

  sleep .5

  ### TESTING NEW WEEK CALCULATION
  WEEK_OF_YEAR="$(echo "($(date +%m)-1) * 4" | bc -l)"
  DAY_OF_MONTH="$(date +%d)"
  if [ ${DAY_OF_MONTH} -le 7 ];then
    WEEK="$(echo "${WEEK_OF_YEAR} + 1" |bc -l)"
  elif [ ${DAY_OF_MONTH} -le 14 ];then
    WEEK="$(echo "${WEEK_OF_YEAR} + 2" |bc -l)"
  elif [ ${DAY_OF_MONTH} -le 21 ];then
    WEEK="$(echo "${WEEK_OF_YEAR} + 3" |bc -l)"
  elif [ ${DAY_OF_MONTH} -ge 22 ];then
    WEEK="$(echo "${WEEK_OF_YEAR} + 4" |bc -l)"
  fi

  for i in "${files[@]}";do
    [ ! -f ${1}/${i} ] && continue
    echo "${1}/${i}" > "${analyzing_now_path}"
    [ -z ${RECORDING_LENGTH} ] && RECORDING_LENGTH=15
    echo "RECORDING_LENGTH set to ${RECORDING_LENGTH}"
    itr=0
    until [ -z "$(lsof -t ${1}/${i})" ];do
      itr=$((itr+1))
      if [ $itr -eq $(($RECORDING_LENGTH * 3)) ]; then
        echo "Maximum number of attempts exceeded. Exiting & restarting service."
        exit
      fi
      sleep 2
    done

    if ! grep 5050 <(netstat -tulpn 2>&1) &> /dev/null 2>&1;then
      echo "Waiting for socket"
      until grep 5050 <(netstat -tulpn 2>&1) &> /dev/null 2>&1;do
        sleep 1
      done
    fi
    # prepare optional parameters for analyze.py
    if [ -f ${INCLUDE_LIST} ]; then
      INCLUDEPARAM="--include_list ${INCLUDE_LIST}"
    else
      INCLUDEPARAM=""
    fi
    if [ -f ${EXCLUDE_LIST} ]; then
      EXCLUDEPARAM="--exclude_list ${EXCLUDE_LIST}"
    else
      EXCLUDEPARAM=""
    fi
    if [ ! -z $BIRDWEATHER_ID ]; then
      BIRDWEATHER_ID_PARAM="--birdweather_id ${BIRDWEATHER_ID}"
      BIRDWEATHER_ID_LOG="--birdweather_id \"IN_USE\""
    else
      BIRDWEATHER_ID_PARAM=""
      BIRDWEATHER_ID_LOG=""
    fi
    echo $PYTHON_VIRTUAL_ENV "$DIR/analyze.py" \
--i "${1}/${i}" \
--o "${1}/${i}.csv" \
--lat $(echo "${LATITUDE}" | awk '{print int($1+0.5)}').XX \
--lon $(echo "${LONGITUDE}" | awk '{print int($1+0.5)}').XX \
--week "${WEEK}" \
--overlap "${OVERLAP}" \
--sensitivity "${SENSITIVITY}" \
--min_conf "${CONFIDENCE}" \
${INCLUDEPARAM} \
${EXCLUDEPARAM} \
${BIRDWEATHER_ID_LOG}
    $PYTHON_VIRTUAL_ENV $DIR/analyze.py \
      --i "${1}/${i}" \
      --o "${1}/${i}.csv" \
      --lat "${LATITUDE}" \
      --lon "${LONGITUDE}" \
      --week "${WEEK}" \
      --overlap "${OVERLAP}" \
      --sensitivity "${SENSITIVITY}" \
      --min_conf "${CONFIDENCE}" \
      ${INCLUDEPARAM} \
      ${EXCLUDEPARAM} \
      ${BIRDWEATHER_ID_PARAM}
    if [ ! -z $HEARTBEAT_URL ]; then
      echo "Performing Heartbeat"
      IP=`curl -s ${HEARTBEAT_URL}`
      echo "Heartbeat: $IP"
    fi
  done
}

# The three main functions
# Takes one argument:
#   - {DIRECTORY}
run_birdnet() {
  get_files "${1}"
  move_analyzed "${1}"
  run_analysis "${1}"
}

until grep 5050 <(netstat -tulpn 2>&1) &> /dev/null 2>&1;do
  sleep 1
done

if [ $(find "${STREAMDATA_DIR}" -maxdepth 1 -name '*wav' 2>/dev/null| wc -l) -gt 0 ];then
  find "${RECORDING_DIR}" -maxdepth 1 -name '*wav' -type f -size 0 -delete
  run_birdnet "${STREAMDATA_DIR}"
fi

YESTERDAY="${STREAMDATA_DIR}/$(date --date="yesterday" "+%B-%Y/%d-%A")"
TODAY="${STREAMDATA_DIR}/$(date "+%B-%Y/%d-%A")"
if [ $(find ${YESTERDAY} -name '*wav' 2>/dev/null | wc -l) -gt 0 ];then
  find $YESTERDAY -name '*wav' -type f -size 0 -delete
  run_birdnet "${YESTERDAY}"
elif [ $(find ${TODAY} -name '*wav' 2>/dev/null | wc -l) -gt 0 ];then
  find $TODAY -name '*wav' -type f -size 0 -delete
  run_birdnet "${TODAY}"
fi
