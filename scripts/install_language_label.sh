#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

MODEL_DIR="$(getDirectory 'model')"
labels_l18n_zip_path="$(getFilePath 'labels_l18n.zip')"
labels_txt_path="$(getFilePath 'labels.txt')"
labels_flickr_path="$(getFilePath 'labels_flickr.txt')"

usage() { echo "Usage: $0 -l <language i18n id>" 1>&2; exit 1; }

while getopts "l:" o; do
  case "${o}" in
    l)
      lang=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

#HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)

label_file_name="labels_${lang}.txt"

unzip -o $labels_l18n_zip_path $label_file_name \
  -d $MODEL_DIR \
  && mv -f $MODEL_DIR/$label_file_name $labels_txt_path \
  && logger "[$0] Changed language label file to '$label_file_name'";

label_file_name_flickr="labels_en.txt"

unzip -o $labels_l18n_zip_path $label_file_name_flickr \
  -d $MODEL_DIR \
  && mv -f $$MODEL_DIR/$label_file_name_flickr $labels_flickr_path \
  && logger "[$0] Set Flickr labels '$label_file_name_flickr'";

exit 0
