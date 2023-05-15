#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

# Restarts ALL services and removes ALL unprocessed audio
source /etc/birdnet/birdnet.conf
SCRIPTS_DIR="$(getDirectory 'scripts')"

# Raise script debug flag
set -x

sudo systemctl stop birdnet_server.service
sudo systemctl stop birdnet_recording.service

services=(chart_viewer.service
  spectrogram_viewer.service
  icecast2.service
  extraction.service
  birdnet_recording.service
  birdnet_log.service
  birdnet_stats.service)

for i in  "${services[@]}";do
  sudo systemctl restart "${i}"
done

sudo systemctl start birdnet_server.service
sleep 5

for i in {1..5}; do
  # We want to loop here (5*5seconds) until the server is running and listening on its port
  systemctl is-active --quiet birdnet_server.service \
	  && grep 5050 <(netstat -tulpn 2>&1) \
	  && logger "[$0] birdnet_server.service is running" \
	  && break

  sleep 5
done

# Let's check a final time to ensure the server is running
systemctl is-active --quiet birdnet_server.service && grep 5050 <(netstat -tulpn 2>&1)
status=$?

if (( status != 0 )); then
  logger "[$0] Unable to start birdnet_server.service... Looping until it start properly"

  until grep 5050 <(netstat -tulpn 2>&1);do
    sudo systemctl restart birdnet_server.service
    sleep 45
  done
fi

# Finally start the birdnet_analysis.service
sudo systemctl restart birdnet_analysis.service
