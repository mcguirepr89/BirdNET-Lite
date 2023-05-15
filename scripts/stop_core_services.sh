#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

RECORDINGS_DIRECTORY="$(getDirectory 'recordings_dir')"

# Stops ALL services and removes ALL unprocessed audio
services=(birdnet_recording.service
custom_recording.service
birdnet_analysis.service
birdnet_server.service
chart_viewer.service
extraction.service
spectrogram_viewer.service)

for i in  "${services[@]}";do
  sudo systemctl stop  ${i}
done
sudo rm -rf "${RECORDINGS_DIRECTORY}"/$(date +%B-%Y/%d-%A)/*
