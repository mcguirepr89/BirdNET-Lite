#!/usr/bin/env bash
BINDIR=$(cd $(dirname $0) && pwd)
. ${BINDIR}/common.sh

BIRDS_DB="$(getFilePath 'birds.db')"

source /etc/birdnet/birdnet.conf
sqlite3 $BIRDS_DB << EOF
DROP TABLE IF EXISTS detections;
CREATE TABLE IF NOT EXISTS detections (
  Date DATE,
  Time TIME,
  Sci_Name VARCHAR(100) NOT NULL,
  Com_Name VARCHAR(100) NOT NULL,
  Confidence FLOAT,
  Lat FLOAT,
  Lon FLOAT,
  Cutoff FLOAT,
  Week INT,
  Sens FLOAT,
  Overlap FLOAT,
  File_Name VARCHAR(100) NOT NULL);
CREATE INDEX "detections_Com_Name" ON "detections" ("Com_Name");
CREATE INDEX "detections_Date_Time" ON "detections" ("Date" DESC, "Time" DESC);
EOF
chown $USER:$USER $BIRDS_DB
chmod g+w $BIRDS_DB
