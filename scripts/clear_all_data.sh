#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

USER=$(awk -F: '/1000/ {print $1}' /etc/passwd)
HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)
SCRIPTS_DIR="$(getDirectory 'scripts')"
HOME_DIR="$(getDirectory 'home')"
BIRDNET_PI_DIR="$(getDirectory 'birdnet_pi')"
RECORDING_DIR="$(getDirectory 'recs_dir')"
EXTRACTED_DIR="$(getDirectory 'extracted')"
id_file_path="$(getFilePath 'IDFILE')"
bird_db_txt_path="$(getFilePath 'BirdDB.txt')"

source /etc/birdnet/birdnet.conf

# This script removes all data that has been collected. It is tantamount to
# starting all data-collection from scratch. Only run this if you are sure
# you are okay will losing all the data that you've collected and processed
# so far.
set -x

echo "Stopping services"
sudo systemctl stop birdnet_recording.service
sudo systemctl stop birdnet_analysis.service
sudo systemctl stop birdnet_server.service
echo "Removing all data . . . "
sudo rm -drf "${RECORDING_DIR}"
sudo rm -f "${id_file_path}"
sudo rm -f "${bird_db_txt_path}"

echo "Re-creating necessary directories"
[ -d ${EXTRACTED_DIR} ] || sudo -u ${USER} mkdir -p ${EXTRACTED_DIR}
[ -d ${EXTRACTED_DIR}/By_Date ] || sudo -u ${USER} mkdir -p ${EXTRACTED_DIR}/By_Date
[ -d ${EXTRACTED_DIR}/Charts ] || sudo -u ${USER} mkdir -p ${EXTRACTED_DIR}/Charts
[ -d ${EXTRACTED_DIR} ] || sudo -u ${USER} mkdir -p ${EXTRACTED_DIR}

sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/exclude_species_list.txt $SCRIPTS_DIR
sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/include_species_list.txt $SCRIPTS_DIR
sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/homepage/* ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/model/labels.txt ${SCRIPTS_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/play.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/spectrogram.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/overview.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/stats.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/todays_detections.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/history.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/weekly_report.php ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $SCRIPTS_DIR/homepage/images/favicon.ico ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $HOME_DIR/phpsysinfo ${EXTRACTED_DIR}
sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/templates/phpsysinfo.ini ${HOME_DIR}/phpsysinfo/
sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/templates/green_bootstrap.css ${HOME_DIR}/phpsysinfo/templates/
sudo -u ${USER} ln -fs $BIRDNET_PI_DIR/templates/index_bootstrap.html ${HOME_DIR}/phpsysinfo/templates/html
chmod -R g+rw $SCRIPTS_DIR
chmod -R g+rw $RECORDING_DIR


echo "Dropping and re-creating database"
createdb.sh
echo "Re-generating BirdDB.txt"
touch "${bird_db_txt_path}"
echo "Date;Time;Sci_Name;Com_Name;Confidence;Lat;Lon;Cutoff;Week;Sens;Overlap" > "${bird_db_txt_path}"
ln -sf ${bird_db_txt_path} "${SCRIPTS_DIR}/BirdDB.txt"
chown $USER:$USER "${SCRIPTS_DIR}/BirdDB.txt" && chmod g+rw "${SCRIPTS_DIR}/BirdDB.txt"
echo "Restarting services"
#restart_services.sh
