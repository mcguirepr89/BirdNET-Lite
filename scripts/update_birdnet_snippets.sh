#!/usr/bin/env bash
# Update BirdNET-Pi
source /etc/birdnet/birdnet.conf
trap 'exit 1' SIGINT SIGHUP
USER=$(awk -F: '/1000/ {print $1}' /etc/passwd)
HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)
my_dir=$HOME/BirdNET-Pi/scripts
if ! grep python3 <(head -n1 $my_dir/analyze.py) &>/dev/null;then
  echo "Ensure all python scripts use the virtual environment"
  sudo -u$USER sed -si "1 i\\#\!/usr/bin/env python3" $my_dir/*.py
fi
if ! grep PRIVACY_THRESHOLD /etc/birdnet/birdnet.conf &>/dev/null;then
  sudo -u$USER echo "PRIVACY_THRESHOLD=0" >> /etc/birdnet/birdnet.conf
  git -C $HOME/BirdNET-Pi rm $my_dir/privacy_server.py 
fi
if [ -f $my_dir/privacy_server ] || [ -L /usr/local/bin/privacy_server.py ];then
  rm -f $my_dir/privacy_server.py
  rm -f /usr/local/bin/privacy_server.py
fi
if grep privacy ~/BirdNET-Pi/templates/birdnet_server.service &>/dev/null;then
  sudo -E sed -i 's/privacy_server.py/server.py/g' \
    ~/BirdNET-Pi/templates/birdnet_server.service
  sudo systemctl daemon-reload
  restart_services.sh
fi
if ! grep APPRISE_NOTIFICATION_TITLE /etc/birdnet/birdnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFICATION_TITLE=\"New BirdNET-Pi Detection\"" >> /etc/birdnet/birdnet.conf
fi
if ! grep APPRISE_NOTIFICATION_BODY /etc/birdnet/birdnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFICATION_BODY=\"A \$sciname \$comname was just detected with a confidence of \$confidence\"" >> /etc/birdnet/birdnet.conf
fi
if ! grep APPRISE_NOTIFY_EACH_DETECTION /etc/birdnet/birdnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_EACH_DETECTION=false" >> /etc/birdnet/birdnet.conf
fi
apprise_installation_status=$(~/BirdNET-Pi/birdnet/bin/python3 -c 'import pkgutil; print("installed" if pkgutil.find_loader("apprise") else "not installed")')
if [[ "$apprise_installation_status" = "not installed" ]];then
  ~/BirdNET-Pi/birdnet/bin/pip3 install -U pip
  ~/BirdNET-Pi/birdnet/bin/pip3 install apprise
fi
[ -f $HOME/BirdNET-Pi/apprise.txt ] || sudo -E -u$USER touch $HOME/BirdNET-Pi/apprise.txt
if ! which lsof &>/dev/null;then
  sudo apt update && sudo apt -y install lsof
fi
if ! grep RTSP_STREAM /etc/birdnet/birdnet.conf &>/dev/null;then
  sudo -u$USER echo "RTSP_STREAM=" >> /etc/birdnet/birdnet.conf
fi
if grep bash $HOME/BirdNET-Pi/templates/web_terminal.service;then
  sudo sed -i '/User/d;s/bash/login/g' $HOME/BirdNET-Pi/templates/web_terminal.service
  sudo systemctl daemon-reload
  sudo systemctl restart web_terminal.service
fi
