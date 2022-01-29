#!/usr/bin/env bash
# Restarts ALL services and removes ALL unprocessed audio
source /etc/birdnet/birdnet.conf
my_dir=/home/pi/BirdNET-Pi/scripts

sudo systemctl stop birdnet_recording.service
sudo rm -rf ${RECS_DIR}/$(date +%B-%Y/%d-%A)/*
sudo systemctl start birdnet_recording.service

services=($(awk '/service/ && /systemctl/ && !/php/ {print $3}' ${my_dir}/install_services.sh | sort))

for i in  "${services[@]}";do

echo "restarting ${i}"
sudo systemctl restart "${i}"
done
sudo systemctl restart extraction.timer
