#!/usr/bin/env bash
#set -e

#BINDIR=$(cd $(dirname $0) && pwd)
. ../scripts/common.sh

#We only need these setting values during testing
RECS_DIR=$HOME/BirdSongs
PROCESSED=${RECS_DIR}/Processed
EXTRACTED=${RECS_DIR}/Extracted
IDFILE=$HOME/BirdNET-Pi/IdentifiedSoFar.txt

#Initialize to expected home location
HOME="/home/pi"

#Test over varring home folders
TEST_HOMES=('/home/pi' '/home/another_user' '/opt/birdnet')

for _test_home in "${TEST_HOMES[@]}"; do
  HOME="$_test_home"

  ## DIRECTORY TESTS
  result="$(getDirectory 'birdnet_pi')"
  expected="$_test_home/BirdNET-Pi"
  [ "$result" == $expected ] || echo "directory 'birdnet_pi' failed, expected: $expected - got: $result"

  result="$(getDirectory 'recs_dir')"
  expected="$_test_home/BirdSongs"
  [ "$result" == $expected ] || echo "directory 'recs_dir' failed, expected: $expected - got: $result"

  result="$(getDirectory 'processed')"
  expected="$_test_home/BirdSongs/Processed"
  [ "$result" == $expected ] || echo "directory 'processed' failed, expected: $expected - got: $result"

  result="$(getDirectory 'extracted')"
  expected="$_test_home/BirdSongs/Extracted"
  [ "$result" == $expected ] || echo "directory 'extracted' failed, expected: $expected - got: $result"

  result="$(getDirectory 'extracted_by_date')"
  expected="$_test_home/BirdSongs/Extracted/By_Date"
  [ "$result" == $expected ] || echo "directory 'extracted_by_date' failed, expected: $expected - got: $result"

  result="$(getDirectory 'shifted_audio')"
  expected="$_test_home/BirdSongs/Extracted/By_Date/shifted"
  [ "$result" == $expected ] || echo "directory 'shifted_audio' failed, expected: $expected - got: $result"

  result="$(getDirectory 'database')"
  expected="$_test_home/BirdNET-Pi/database"
  [ "$result" == $expected ] || echo "directory 'database' failed, expected: $expected - got: $result"

  result="$(getDirectory 'config')"
  expected="$_test_home/BirdNET-Pi/config"
  [ "$result" == $expected ] || echo "directory 'config' failed, expected: $expected - got: $result"

  result="$(getDirectory 'models')"
  expected="$_test_home/BirdNET-Pi/model"
  [ "$result" == $expected ] || echo "directory 'models' failed, expected: $expected - got: $result"

  result="$(getDirectory 'python3_ve')"
  expected="$_test_home/BirdNET-Pi/birdnet/bin"
  [ "$result" == $expected ] || echo "directory 'python3_ve' failed, expected: $expected - got: $result"

  result="$(getDirectory 'scripts')"
  expected="$_test_home/BirdNET-Pi/scripts"
  [ "$result" == $expected ] || echo "directory 'scripts' failed, expected: $expected - got: $result"

  result="$(getDirectory 'stream_data')"
  expected="$_test_home/BirdSongs/StreamData"
  [ "$result" == $expected ] || echo "directory 'stream_data' failed, expected: $expected - got: $result"

  result="$(getDirectory 'templates')"
  expected="$_test_home/BirdNET-Pi/templates"
  [ "$result" == $expected ] || echo "directory 'templates' failed, expected: $expected - got: $result"

  result="$(getDirectory 'web')"
  expected="$_test_home/BirdNET-Pi/homepage"
  [ "$result" == $expected ] || echo "directory 'web' failed, expected: $expected - got: $result"

  ## FILE PATH TESTS
  result="$(getFilePath 'analyzing_now.txt')"
  expected="$_test_home/BirdNET-Pi/analyzing_now.txt"
  [ "$result" == $expected ] || echo "file path 'analyzing_now.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'apprise.txt')"
  expected="$_test_home/BirdNET-Pi/apprise.txt"
  [ "$result" == $expected ] || echo "file path 'apprise.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'birdnet.conf')"
  expected="$_test_home/BirdNET-Pi/birdnet.conf"
  [ "$result" == $expected ] || echo "file path 'birdnet.conf' failed, expected: $expected - got: $result"

  result="$(getFilePath 'etc_birdnet.conf')"
  expected="/etc/birdnet/birdnet.conf"
  [ "$result" == $expected ] || echo "file path 'etc_birdnet.conf' failed, expected: $expected - got: $result"

  result="$(getFilePath 'BirdDB.txt')"
  expected="$_test_home/BirdNET-Pi/BirdDB.txt"
  [ "$result" == $expected ] || echo "file path 'BirdDB.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'birds.db')"
  expected="$_test_home/BirdNET-Pi/scripts/birds.db"
  [ "$result" == $expected ] || echo "file path 'birds.db' failed, expected: $expected - got: $result"

  result="$(getFilePath 'blacklisted_images.txt')"
  expected="$_test_home/BirdNET-Pi/scripts/blacklisted_images.txt"
  [ "$result" == $expected ] || echo "file path 'blacklisted_images.txt'' failed, expected: $expected - got: $result"

  result="$(getFilePath 'disk_check_exclude.txt')"
  expected="$_test_home/BirdNET-Pi/scripts/disk_check_exclude.txt"
  [ "$result" == $expected ] || echo "file path 'disk_check_exclude.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'email_template')"
  expected="$_test_home/BirdNET-Pi/scripts/email_template"
  [ "$result" == $expected ] || echo "file path 'email_template' failed, expected: $expected - got: $result"

  result="$(getFilePath 'email_template2')"
  expected="$_test_home/BirdNET-Pi/scripts/email_template2"
  [ "$result" == $expected ] || echo "file path 'email_template2' failed, expected: $expected - got: $result"

  result="$(getFilePath 'exclude_species_list.txt')"
  expected="$_test_home/BirdNET-Pi/scripts/exclude_species_list.txt"
  [ "$result" == $expected ] || echo "file path 'exclude_species_list.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'firstrun.ini')"
  expected="$_test_home/BirdNET-Pi/firstrun.ini"
  [ "$result" == $expected ] || echo "file path 'firstrun.ini' failed, expected: $expected - got: $result"

  result="$(getFilePath '.gotty')"
  expected="$_test_home/.gotty"
  [ "$result" == $expected ] || echo "file path '.gotty' failed, expected: $expected - got: $result"

  result="$(getFilePath 'HUMAN.txt')"
  expected="$_test_home/BirdNET-Pi/HUMAN.txt"
  [ "$result" == $expected ] || echo "file path 'HUMAN.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'IdentifiedSoFar.txt')"
  expected="$_test_home/BirdNET-Pi/IdentifiedSoFar.txt"
  [ "$result" == $expected ] || echo "file path 'IdentifiedSoFar.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'include_species_list.txt')"
  expected="$_test_home/BirdNET-Pi/scripts/include_species_list.txt"
  [ "$result" == $expected ] || echo "file path 'include_species_list.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'labels.txt')"
  expected="$_test_home/BirdNET-Pi/model/labels.txt"
  [ "$result" == $expected ] || echo "file path 'labels.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'labels.txt.old')"
  expected="$_test_home/BirdNET-Pi/model/labels.txt.old"
  [ "$result" == $expected ] || echo "file path 'labels.txt.old' failed, expected: $expected - got: $result"

  result="$(getFilePath 'labels_flickr.txt')"
  expected="$_test_home/BirdNET-Pi/model/labels_flickr.txt"
  [ "$result" == $expected ] || echo "file path 'labels_flickr.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'labels_l18n.zip')"
  expected="$_test_home/BirdNET-Pi/model/labels_l18n.zip"
  [ "$result" == $expected ] || echo "file path 'labels_l18n.zip' failed, expected: $expected - got: $result"

  result="$(getFilePath 'labels_lang.txt')"
  expected="$_test_home/BirdNET-Pi/model/labels_lang.txt"
  [ "$result" == $expected ] || echo "file path 'labels_lang.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'labels_nm.zip')"
  expected="$_test_home/BirdNET-Pi/model/labels_nm.zip"
  [ "$result" == $expected ] || echo "file path 'labels_nm.zip' failed, expected: $expected - got: $result"

  result="$(getFilePath 'lastrun.txt')"
  expected="$_test_home/BirdNET-Pi/scripts/lastrun.txt"
  [ "$result" == $expected ] || echo "file path 'lastrun.txt' failed, expected: $expected - got: $result"

  result="$(getFilePath 'python3')"
  expected="$_test_home/BirdNET-Pi/birdnet/bin/python3 "
  [ "$result" == "$expected" ] || echo "file path 'python3' failed, expected: $expected - got: $result"

  result="$(getFilePath 'python3_appraise')"
  expected="$_test_home/BirdNET-Pi/birdnet/bin/apprise "
  [ "$result" == "$expected" ] || echo "file path 'python3_appraise' failed, expected: $expected - got: $result"

  result="$(getFilePath 'species.py')"
  expected="$_test_home/BirdNET-Pi/scripts/species.py"
  [ "$result" == $expected ] || echo "file path 'species.py' failed, expected: $expected - got: $result"

  result="$(getFilePath 'thisrun.txt')"
  expected="$_test_home/BirdNET-Pi/scripts/thisrun.txt"
  [ "$result" == $expected ] || echo "file path 'thisrun.txt' failed, expected: $expected - got: $result"

done
