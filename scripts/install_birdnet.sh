#!/usr/bin/env bash
#
# Install BirdNET script

set -x # Debugging
exec > >(tee -i installation-$(date +%F).txt) 2>&1 # Make log
set -e # Exit installation if anything fails

# Check OS platform
if [ "$(uname -m)" != "aarch64" ];then
  echo "BirdNET-Pi requires a 64-bit OS.
It looks like your operating system is using $(uname -m),
but would need to be aarch64.
Please take a look at https://birdnetwiki.pmcgui.xyz for more
information"
  exit 1
fi

# Determine the base BirdNET-Pi diectory
my_dir=$HOME/BirdNET-Pi
export my_dir=$my_dir

# Make sure that all scripts are executable
chmod +x $my_dir/scripts/*.sh || exit 1

# Install & configure the config file /etc/birdnet/birdnet.conf
# and install the services
cd $my_dir/scripts || exit 1
./install_config.sh || exit 1
sudo -E HOME=$HOME USER=$USER ./install_services.sh || exit 1
source /etc/birdnet/birdnet.conf

install_birdnet() {
  cd ~/BirdNET-Pi || exit 1
  echo "Establishing a python virtual environment"
  python3 -m venv birdnet
  source ./birdnet/bin/activate
  pip3 install -U -r $HOME/BirdNET-Pi/requirements.txt
}

# Generate the recording directory
[ -d ${RECS_DIR} ] || mkdir -p ${RECS_DIR} &> /dev/null

# Install the python environment
install_birdnet

# Install the language labels of the new model
# according to the configured database language
cd $my_dir/scripts || exit 1
./install_language_label_nm.sh -l $DATABASE_LANG || exit 1

exit 0
