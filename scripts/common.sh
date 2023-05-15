#!/usr/bin/env bash
# Load config and settings
source /etc/birdnet/birdnet.conf
#source ./birdnet.conf
# $HOME here is home of the user executing the script
EXPANDED_HOME=$HOME

# Get the user and home directory of the user with id 1000 (normally pi)
USER=$(awk -F: '/1000/ {print $1}' /etc/passwd)
HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)

# If script is being run as sudo get the home path for the user that's running sudo
#if [ $SUDO_USER ]; then HOME=$(getent passwd $SUDO_USER | cut -d: -f6); fi


getDirectory() {
  DIR_NAME=$1
  DIR_NAME=$(echo "$DIR_NAME" | tr '[:upper:]' '[:lower:]')

  if [ "$DIR_NAME" = "home" ]; then
    echo $HOME
    ##
  elif [ "$DIR_NAME" = "birdnet-pi" ] || [ "$DIR_NAME" = "birdnet_pi" ]; then
    echo "$(getDirectory 'home')/BirdNET-Pi"
    ##
  elif [ "$DIR_NAME" = "recs_dir" ] || [ "$DIR_NAME" = "recordings_dir" ]; then
    home_dir="$(getDirectory 'home')"
    recs_dir_setting=$RECS_DIR
    echo ${recs_dir_setting//$EXPANDED_HOME/$home_dir}
    ##
  elif [ "$DIR_NAME" = "processed" ]; then
    #replace the entire ${RECS_DIR} expansion as it is expanded on $HOME of the user executing the script
    #this may or may not be correct, so we want to be able to replace it and put in place the calculated recordings dir
    processed_dir_setting=${PROCESSED//"$EXPANDED_HOME/BirdSongs"/""}
    echo "$(getDirectory 'recs_dir')${processed_dir_setting//"$(getDirectory 'recs_dir')"/}"
    ##
  elif [ "$DIR_NAME" = "extracted" ]; then
    #replace the entire ${RECS_DIR} expansion as it is expanded on $HOME of the user executing the script
    #this may or may not be correct, so we want to be able to replace it and put in place the calculated recordings dir
    home_dir="$(getDirectory 'home')"
    extracted_dir_setting=${EXTRACTED//"$EXPANDED_HOME/BirdSongs"/""}
    echo "$(getDirectory 'recs_dir')${extracted_dir_setting//"$(getDirectory 'recs_dir')"/}"
    ##
  elif [ "$DIR_NAME" = "extracted_bydate" ] || [ "$DIR_NAME" = "extracted_by_date" ]; then
    echo "$(getDirectory 'extracted')/By_Date"
    ##
  elif [ "$DIR_NAME" = "shifted_audio" ] || [ "$DIR_NAME" = "shifted_dir" ]; then
    echo "$(getDirectory 'home')/BirdSongs/Extracted/By_Date/shifted"
    ##
  elif [ "$DIR_NAME" = "database" ]; then
    #		 NOT USED
    echo "$(getDirectory 'birdnet_pi')/database"
    ##
  elif [ "$DIR_NAME" = "config" ]; then
    #		 NOT USED
    echo "$(getDirectory 'birdnet_pi')/config"
    ##
  elif [ "$DIR_NAME" = "models" ] || [ "$DIR_NAME" = "model" ]; then
    echo "$(getDirectory 'birdnet_pi')/model"
    ##
  elif [ "$DIR_NAME" = "python3_ve" ]; then
    echo "$(getDirectory 'birdnet_pi')/birdnet/bin"
    ##
  elif [ "$DIR_NAME" = "scripts" ]; then
    echo "$(getDirectory 'birdnet_pi')/scripts"
    ##
  elif [ "$DIR_NAME" = "stream_data" ]; then
    echo "$(getDirectory 'recs_dir')/StreamData"
    ##
  elif [ "$DIR_NAME" = "templates" ]; then
    echo "$(getDirectory 'birdnet_pi')/templates"
    ##
  elif [ "$DIR_NAME" = "web" ] || [ "$DIR_NAME" = "www" ]; then
    echo "$(getDirectory 'birdnet_pi')/homepage"
    ##
  else
    echo
  fi
}

getFilePath() {
  FILENAME=$1

  if [ "$FILENAME" = "analyzing_now.txt" ]; then
    echo "$(getDirectory 'birdnet_pi')/analyzing_now.txt"
    ##
  elif [ "$FILENAME" = "apprise.txt" ]; then
    echo "$(getDirectory 'birdnet_pi')/apprise.txt"
    ##
  elif [ "$FILENAME" = "birdnet.conf" ]; then
    echo "$(getDirectory 'birdnet_pi')/birdnet.conf"
    ##
  elif [ "$FILENAME" = "etc_birdnet.conf" ]; then
    echo "/etc/birdnet/birdnet.conf"
    ##
  elif [ "$FILENAME" = "BirdDB.txt" ]; then
    echo "$(getDirectory 'birdnet_pi')/BirdDB.txt"
    ##
  elif [ "$FILENAME" = "birds.db" ]; then
    echo "$(getDirectory 'scripts')/birds.db"
    ##
  elif [ "$FILENAME" = "blacklisted_images.txt" ]; then
    echo "$(getDirectory 'scripts')/blacklisted_images.txt"
    ##
  elif [ "$FILENAME" = "disk_check_exclude.txt" ]; then
    echo "$(getDirectory 'scripts')/disk_check_exclude.txt"
    ##
  elif [ "$FILENAME" = "email_template" ]; then
    echo "$(getDirectory 'scripts')/email_template"
    ##
  elif [ "$FILENAME" = "email_template2" ]; then
    echo "$(getDirectory 'scripts')/email_template2"
    ##
  elif [ "$FILENAME" = "exclude_species_list.txt" ]; then
    echo "$(getDirectory 'scripts')/exclude_species_list.txt"
    ##
  elif [ "$FILENAME" = "firstrun.ini" ]; then
    echo "$(getDirectory 'birdnet_pi')/firstrun.ini"
    ##
  elif [ "$FILENAME" = ".gotty" ]; then
    echo "$(getDirectory 'home')/.gotty"
  ##
  elif [ "$FILENAME" = "HUMAN.txt" ]; then
    echo "$(getDirectory 'birdnet_pi')/HUMAN.txt"
    ##
  elif [ "$FILENAME" = "IdentifiedSoFar.txt" ] || [ "$FILENAME" = "IDFILE" ]; then
    id_home_dir=$(getDirectory 'home')
    echo "${IDFILE//$EXPANDED_HOME/$id_home_dir}"
  ##
  elif [ "$FILENAME" = "include_species_list.txt" ]; then
    echo "$(getDirectory 'scripts')/include_species_list.txt"
    ##
  elif [ "$FILENAME" = "labels.txt" ] || [ "$FILENAME" = "labels.txt.old" ]; then
    echo "$(getDirectory 'model')/$FILENAME"
    ##
  elif [ "$FILENAME" = "labels_flickr.txt" ]; then
    echo "$(getDirectory 'model')/labels_flickr.txt"
    ##
  elif [ "$FILENAME" = "labels_l18n.zip" ]; then
    echo "$(getDirectory 'model')/labels_l18n.zip"
    ##
  elif [ "$FILENAME" = "labels_lang.txt" ]; then
    echo "$(getDirectory 'model')/labels_lang.txt"
    ##
  elif [ "$FILENAME" = "labels_nm.zip" ]; then
    echo "$(getDirectory 'model')/labels_nm.zip"
    ##
  elif [ "$FILENAME" = "lastrun.txt" ]; then
    echo "$(getDirectory 'scripts')/lastrun.txt"
    ##
  elif [ "$FILENAME" = "python3" ]; then
    echo "$(getDirectory 'python3_ve')/python3 "
    ##
  elif [ "$FILENAME" = "python3_appraise" ]; then
    echo "$(getDirectory 'python3_ve')/apprise "
    ##
  elif [ "$FILENAME" = "species.py" ]; then
    echo "$(getDirectory 'scripts')/species.py"
    ##
  elif [ "$FILENAME" = "thisrun.txt" ]; then
    echo "$(getDirectory 'scripts')/thisrun.txt"
    ##
  else
    echo
  fi
}
