#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

apprise_txt_path="$(getFilePath 'apprise.txt')"
PYTHON3_VE_DIR="$(getDirectory 'python3_ve')"

source /etc/birdnet/birdnet.conf
if [ ${APPRISE_WEEKLY_REPORT} == 1 ];then
	NOTIFICATION=$(curl 'localhost/views.php?view=Weekly%20Report&ascii=true')
	NOTIFICATION=${NOTIFICATION#*#}
	firstLine=`echo "${NOTIFICATION}" | head -1`
	NOTIFICATION=`echo "${NOTIFICATION}" | tail -n +2`
	"$PYTHON3_VE_DIR"/apprise -vv -t "${firstLine}" -b "${NOTIFICATION}" --input-format=html --config="$apprise_txt_path"
fi