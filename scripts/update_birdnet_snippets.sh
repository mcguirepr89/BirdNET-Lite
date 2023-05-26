#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

SCRIPTS_DIR="$(getDirectory 'scripts')"
apprise_txt_path="$(getFilePath 'apprise.txt')"
birds_db_path="$(getFilePath 'birds.db')"
etc_birdnet_conf_path="$(getFilePath 'etc_birdnet.conf')"
birdnet_conf_path="$(getFilePath 'birdnet.conf')"
labels_txt_path="$(getFilePath 'labels.txt')"
labels_txt_old_path="$(getFilePath 'labels.txt.old')"
python3_exec_path="$(getFilePath 'python3')"
thisrun_txt_path="$(getFilePath 'thisrun.txt')"

HOME_DIR="$(getDirectory 'home')"
BIRDNET_PI_DIR="$(getDirectory 'birdnet_pi')"
EXTRACTED_DIR="$(getDirectory 'extracted')"
TEMPLATES_DIR="$(getDirectory 'templates')"
PYTHON3_VE_DIR="$(getDirectory 'python3_ve')"
WWW_HOMEPAGE_DIR="$(getDirectory 'web')"

# Update BirdNET-Pi
source /etc/birdnet/birdnet.conf
trap 'exit 1' SIGINT SIGHUP
USER=$(awk -F: '/1000/ {print $1}' /etc/passwd)
HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)

# Sets proper permissions and ownership
#sudo -E chown -R $USER:$USER $HOME/*
#sudo chmod -R g+wr $HOME/*
find $HOME_DIR/Bird* -type f ! -perm -g+wr -exec chmod g+wr {} + 2>/dev/null
find $HOME_DIR/Bird* -not -user $USER -execdir sudo -E chown $USER:$USER {} \+
chmod 666 "$SCRIPTS_DIR"/*.txt
chmod 666 "$BIRDNET_PI_DIR"/*.txt
find $BIRDNET_PI_DIR -path "$BIRDNET_PI_DIR/birdnet" -prune -o -type f ! -perm /o=w -exec chmod a+w {} \;

# remove world-writable perms
chmod -R o-w $TEMPLATES_DIR/*


# Create blank sitename as it's optional. First time install will use $HOSTNAME.
if ! grep SITE_NAME "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "SITE_NAME=\"\"" >> "$etc_birdnet_conf_path"
fi

if ! grep PRIVACY_THRESHOLD "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "PRIVACY_THRESHOLD=0" >> "$etc_birdnet_conf_path"
  git -C "$BIRDNET_PI_DIR" rm $SCRIPTS_DIR/privacy_server.py
fi
if [ -f $SCRIPTS_DIR/privacy_server ] || [ -L /usr/local/bin/privacy_server.py ];then
  rm -f $SCRIPTS_DIR/privacy_server.py
  rm -f /usr/local/bin/privacy_server.py
fi

# Adds python virtual-env to the python systemd services
if ! grep 'BirdNET-Pi/birdnet/' $TEMPLATES_DIR/birdnet_server.service &>/dev/null || ! grep 'BirdNET-Pi/birdnet' $TEMPLATES_DIR/chart_viewer.service &>/dev/null;then
  sudo -E sed -i "s|ExecStart=.*|ExecStart=$python3_exec_path /usr/local/bin/server.py|" $TEMPLATES_DIR/birdnet_server.service
  sudo -E sed -i "s|ExecStart=.*|ExecStart=$python3_exec_path /usr/local/bin/daily_plot.py|" $TEMPLATES_DIR/chart_viewer.service
  sudo systemctl daemon-reload && restart_services.sh
fi

if grep privacy $TEMPLATES_DIR/birdnet_server.service &>/dev/null;then
  sudo -E sed -i 's/privacy_server.py/server.py/g' \
    $TEMPLATES_DIR/birdnet_server.service
  sudo systemctl daemon-reload
  restart_services.sh
fi
if ! grep APPRISE_NOTIFICATION_TITLE "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFICATION_TITLE=\"New BirdNET-Pi Detection\"" >> "$etc_birdnet_conf_path"
fi
if ! grep APPRISE_NOTIFICATION_BODY "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFICATION_BODY=\"A \$sciname \$comname was just detected with a confidence of \$confidence\"" >> "$etc_birdnet_conf_path"
fi
if ! grep APPRISE_NOTIFY_EACH_DETECTION "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_EACH_DETECTION=0 " >> "$etc_birdnet_conf_path"
fi
if ! grep APPRISE_NOTIFY_NEW_SPECIES "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_NEW_SPECIES=0 " >> "$etc_birdnet_conf_path"
fi
if ! grep APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY=0 " >> "$etc_birdnet_conf_path"
fi
if ! grep APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES=0 " >> "$etc_birdnet_conf_path"
fi

# If the config does not contain the DATABASE_LANG setting, we'll want to add it.
# Defaults to not-selected, which config.php will know to render as a language option.
# The user can then select a language in the web interface and update with that.
if ! grep DATABASE_LANG "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "DATABASE_LANG=not-selected" >> "$etc_birdnet_conf_path"
fi

apprise_installation_status=$($python3_exec_path -c 'import pkgutil; print("installed" if pkgutil.find_loader("apprise") else "not installed")')
if [[ "$apprise_installation_status" = "not installed" ]];then
  $PYTHON3_VE_DIR/pip3 install -U pip
  $PYTHON3_VE_DIR/pip3 install apprise==1.2.1
fi
[ -f $apprise_txt_path ] || sudo -E -ucaddy touch $apprise_txt_path
if ! which lsof &>/dev/null;then
  sudo apt update && sudo apt -y install lsof
fi
if ! grep RTSP_STREAM "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "RTSP_STREAM=" >> "$etc_birdnet_conf_path"
fi
if grep bash $TEMPLATES_DIR/web_terminal.service &>/dev/null;then
  sudo sed -i '/User/d;s/bash/login/g' $TEMPLATES_DIR/web_terminal.service
  sudo systemctl daemon-reload
  sudo systemctl restart web_terminal.service
fi
[ -L $EXTRACTED_DIR/static ] || ln -sf $WWW_HOMEPAGE_DIR/static $EXTRACTED_DIR
if ! grep FLICKR_API_KEY "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "FLICKR_API_KEY=" >> "$etc_birdnet_conf_path"
fi
if systemctl list-unit-files pushed_notifications.service &>/dev/null;then
  sudo systemctl disable --now pushed_notifications.service
  sudo rm -f /usr/lib/systemd/system/pushed_notifications.service
  sudo rm $TEMPLATES_DIR/pushed_notifications.service
fi

if [ ! -f $labels_txt_path ];then
  [ $DATABASE_LANG == 'not-selected' ] && DATABASE_LANG=en
  $SCRIPTS_DIR/install_language_label.sh -l $DATABASE_LANG \
  && logger "[$0] Installed new language label file for '$DATABASE_LANG'";
fi

if ! grep FLICKR_FILTER_EMAIL "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "FLICKR_FILTER_EMAIL=" >> "$etc_birdnet_conf_path"
fi

pytest_installation_status=$($python3_exec_path -c 'import pkgutil; print("installed" if pkgutil.find_loader("pytest") else "not installed")')
if [[ "$pytest_installation_status" = "not installed" ]];then
  $PYTHON3_VE_DIR/pip3 install -U pip
  $PYTHON3_VE_DIR/pip3 install pytest==7.1.2 pytest-mock==3.7.0
fi

[ -L $EXTRACTED_DIR/weekly_report.php ] || ln -sf $SCRIPTS_DIR/weekly_report.php $EXTRACTED_DIR

if ! grep weekly_report /etc/crontab &>/dev/null;then
  sed "s/\$USER/$USER/g" $TEMPLATES_DIR/weekly_report.cron | sudo tee -a /etc/crontab
fi
if ! grep APPRISE_WEEKLY_REPORT "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_WEEKLY_REPORT=1" >> "$etc_birdnet_conf_path"
fi

if ! grep SILENCE_UPDATE_INDICATOR "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "SILENCE_UPDATE_INDICATOR=0" >> "$etc_birdnet_conf_path"
fi

if ! grep '\-\-browser.gatherUsageStats false' $TEMPLATES_DIR/birdnet_stats.service &>/dev/null ;then
  sudo -E sed -i "s|ExecStart=.*|ExecStart=$PYTHON3_VE_DIR/streamlit run $SCRIPTS_DIR/plotly_streamlit.py --browser.gatherUsageStats false --server.address localhost --server.baseUrlPath \"/stats\"|" $TEMPLATES_DIR/birdnet_stats.service
  sudo systemctl daemon-reload && restart_services.sh
fi

# Make IceCast2 a little more secure
sudo sed -i.bak -e 's|<!-- <bind-address>.*|<bind-address>127.0.0.1</bind-address>|;s|<!-- <shoutcast-mount>.*|<shoutcast-mount>/stream</shoutcast-mount>|' /etc/icecast2/icecast.xml && if [ -s /etc/icecast2/icecast.xml.bak ] && ! sudo diff /etc/icecast2/icecast.xml /etc/icecast2/icecast.xml.bak > /dev/null; then sudo systemctl restart icecast2; fi

if ! grep FREQSHIFT_TOOL "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_TOOL=sox" >> "$etc_birdnet_conf_path"
fi
if ! grep FREQSHIFT_HI "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_HI=6000" >> "$etc_birdnet_conf_path"
fi
if ! grep FREQSHIFT_LO "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_LO=3000" >> "$etc_birdnet_conf_path"
fi
if ! grep FREQSHIFT_PITCH "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_PITCH=-1500" >> "$etc_birdnet_conf_path"
fi
if ! grep ACTIVATE_FREQSHIFT_IN_LIVESTREAM "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "ACTIVATE_FREQSHIFT_IN_LIVESTREAM=\"false\"" >> "$etc_birdnet_conf_path"
fi
if ! grep HEARTBEAT_URL /etc/birdnet/birdnet.conf &>/dev/null;then
  sudo -u$USER echo "HEARTBEAT_URL=" >> /etc/birdnet/birdnet.conf
fi

if ! grep MODEL "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "MODEL=BirdNET_6K_GLOBAL_MODEL" >> "$etc_birdnet_conf_path"
fi
if ! grep SF_THRESH "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "SF_THRESH=0.03" >> "$etc_birdnet_conf_path"
fi
sudo chmod +x $SCRIPTS_DIR/install_language_label_nm.sh

sqlite3 "$birds_db_path" << EOF
CREATE INDEX IF NOT EXISTS "detections_Com_Name" ON "detections" ("Com_Name");
CREATE INDEX IF NOT EXISTS "detections_Date_Time" ON "detections" ("Date" DESC, "Time" DESC);
EOF

apprise_version=$($python3_exec_path -c "import apprise; print(apprise.__version__)")
streamlit_version=$($PYTHON3_VE_DIR/pip3 show streamlit 2>/dev/null | grep Version | awk '{print $2}')

[[ $apprise_version != "1.2.1" ]] && $PYTHON3_VE_DIR/pip3 install apprise==1.2.1
[[ $streamlit_version != "1.19.0" ]] && $PYTHON3_VE_DIR/pip3 install streamlit==1.19.0

if ! grep -q 'RuntimeMaxSec=' "$TEMPLATES_DIR/birdnet_analysis.service"&>/dev/null; then
    sudo -E sed -i '/\[Service\]/a RuntimeMaxSec=3600' "$TEMPLATES_DIR/birdnet_analysis.service"
    sudo systemctl daemon-reload && restart_services.sh
fi

if ! grep RAW_SPECTROGRAM "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "RAW_SPECTROGRAM=0" >> "$etc_birdnet_conf_path"
fi

if ! grep CUSTOM_IMAGE "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "CUSTOM_IMAGE=" >> "$etc_birdnet_conf_path"
fi
if ! grep CUSTOM_IMAGE_TITLE "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "CUSTOM_IMAGE_TITLE=\"\"" >> "$etc_birdnet_conf_path"
fi

if ! grep APPRISE_ONLY_NOTIFY_SPECIES_NAMES "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_ONLY_NOTIFY_SPECIES_NAMES=\"\"" >> "$etc_birdnet_conf_path"
fi
if ! grep APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2 "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "APPRISE_ONLY_NOTIFY_SPECIES_NAMES_2=\"\"" >> "$etc_birdnet_conf_path"
fi

if ! grep RTSP_STREAM_TO_LIVESTREAM "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "RTSP_STREAM_TO_LIVESTREAM=\"0\"" >> "$etc_birdnet_conf_path"
fi

suntime_installation_status=$($python3_exec_path -c 'import pkgutil; print("installed" if pkgutil.find_loader("suntime") else "not installed")')
if [[ "$suntime_installation_status" = "not installed" ]];then
  $PYTHON3_VE_DIR/pip3 install -U pip
  $PYTHON3_VE_DIR/pip3 install suntime
fi

# For new Advanced Setting Logging level options
if ! grep LogLevel_BirdnetRecordingService "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "LogLevel_BirdnetRecordingService=\"error\"" >> "$etc_birdnet_conf_path"
fi

if ! grep LogLevel_LiveAudioStreamService "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "LogLevel_LiveAudioStreamService=\"error\"" >> "$etc_birdnet_conf_path"
fi

if ! grep LogLevel_SpectrogramViewerService "$etc_birdnet_conf_path" &>/dev/null;then
  sudo -u$USER echo "LogLevel_SpectrogramViewerService=\"error\"" >> "$etc_birdnet_conf_path"
fi

if grep -q '^MODEL=BirdNET_GLOBAL_3K_V2.2_Model_FP16$' "$etc_birdnet_conf_path";then
  language=$(grep "^DATABASE_LANG=" "$etc_birdnet_conf_path" | cut -d= -f2)
  sed -i 's/BirdNET_GLOBAL_3K_V2.2_Model_FP16/BirdNET_GLOBAL_3K_V2.3_Model_FP16/' "$etc_birdnet_conf_path"
  sed -i 's/BirdNET_GLOBAL_3K_V2.2_Model_FP16/BirdNET_GLOBAL_3K_V2.3_Model_FP16/' "$thisrun_txt_path"
  sed -i 's/BirdNET_GLOBAL_3K_V2.2_Model_FP16/BirdNET_GLOBAL_3K_V2.3_Model_FP16/' "$birdnet_conf_path"
  cp -f "$labels_txt_path" "$labels_txt_old_path"
  sudo chmod +x "$SCRIPTS_DIR"/install_language_label_nm.sh && "$SCRIPTS_DIR"/install_language_label_nm.sh -l "$language"
fi

# Link in new bash script common.sh to local/bin and make sure the config directory is also
[ -L /usr/local/bin/common.sh ] || ln -sf ${SCRIPTS_DIR}/common.sh /usr/local/bin/
[ -L /usr/local/bin/config ] || ln -sf ${BIRDNET_PI_DIR}/config /usr/local/bin/


sudo systemctl daemon-reload
restart_services.sh
