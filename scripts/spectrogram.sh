#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

analyzing_now_txt_path="$(getFilePath 'analyzing_now.txt')"
HOME_DIR="$(getDirectory 'home')"
EXTRACTED_DIR="$(getDirectory 'extracted')"

# Make sox spectrogram
source /etc/birdnet/birdnet.conf

# Read the logging level from the configuration option
LOGGING_LEVEL="${LogLevel_SpectrogramViewerService}"
# If empty for some reason default to log level of error
[ -z $LOGGING_LEVEL ] && LOGGING_LEVEL='error'
# Additionally if we're at debug or info level then allow printing of script commands and variables
if [ "$LOGGING_LEVEL" == "info" ] || [ "$LOGGING_LEVEL" == "debug" ];then
  # Enable printing of commands/variables etc to terminal for debugging
  set -x
fi

# Time to sleep between generating spectrogram's, default set the recording length
# To try catch the spectrogram as soon as possible run at a smaller intervals
SLEEP_DELAY=$((RECORDING_LENGTH / 4))

# Continuously loop generating a spectrogram every 10 seconds
while true; do
  analyzing_now="$(cat $analyzing_now_txt_path)"

  if [ ! -z "${analyzing_now}" ] && [ -f "${analyzing_now}" ]; then
    spectrogram_png=${EXTRACTED_DIR}/spectrogram.png
    sox -V1 "${analyzing_now}" -n remix 1 rate 24k spectrogram -c "${analyzing_now//$HOME_DIR\//}" -o "${spectrogram_png}"
  fi

  sleep $SLEEP_DELAY
done
