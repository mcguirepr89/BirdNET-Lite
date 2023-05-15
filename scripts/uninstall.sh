#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

gotty_path="$(getFilePath '.gotty')"
birdnet_conf_path="$(getFilePath 'birdnet.conf')"
SCRIPTS_DIR="$(getDirectory 'scripts')"
BIRDNETPI_DIR="$(getDirectory 'birdnet_pi')"

# Uninstall script to remove everything
#set -x # Uncomment to debug
trap 'rm -f ${TMPFILE}' EXIT
SCRIPTS=($(ls -1 ${SCRIPTS_DIR}) ${gotty_path})
set -x
services=($(awk '/service/ && /systemctl/ && !/php/ {print $3}' ${SCRIPTS_DIR}/install_services.sh | sort) custom_recording.service avahi-alias@.service)

remove_services() {
  for i in "${services[@]}"; do
    if [ -L /etc/systemd/system/multi-user.target.wants/"${i}" ];then
      sudo systemctl disable --now "${i}"
    fi
    if [ -L /lib/systemd/system/"${i}" ];then
      sudo rm -f /lib/systemd/system/$i
    fi
    if [ -f /etc/systemd/system/"${i}" ];then
      sudo rm /etc/systemd/system/"${i}"
    fi
    if [ -d /etc/systemd/system/"${i}" ];then
      sudo rm -drf /etc/systemd/system/"${i}"
    fi
  done
  set +x
  remove_icecast
  remove_crons
}

remove_crons() {
  sudo sed -i '/birdnet/,+1d' /etc/crontab
}

remove_icecast() {
  if [ -f /etc/init.d/icecast2 ];then
    sudo /etc/init.d/icecast2 stop
    sudo systemctl disable --now icecast2
  fi
}

remove_scripts() {
  for i in "${SCRIPTS[@]}";do
    if [ -L "/usr/local/bin/${i}" ];then
      sudo rm -v "/usr/local/bin/${i}"
    fi
  done
}

remove_services
remove_scripts
if [ -d /etc/birdnet ];then sudo rm -drf /etc/birdnet;fi
if [ -f "${birdnet_conf_path}" ];then sudo rm -f "${birdnet_conf_path}";fi
echo "Uninstall finished. Remove this directory with 'rm -drfv ${BIRDNETPI_DIR}' to finish."
