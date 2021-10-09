# Birding Pi
![version](https://img.shields.io/badge/version-0.1-blue)

A realtime acoustic bird classification system for the Reaspberry Pi 4B

## Inroduction
The Birding Pi project is based on the BirdNet-Lite project and is able to recognize bird sounds from the microphone and the sound card in realtime. The system installs all needed services on the Raspberry Pi. 

## Requirements
* A Raspberry Pi 4B
* A 64GB SD Card with the 64-bit version of RaspiOS installed [(download the latest here)](https://downloads.raspberrypi.org/raspios_arm64/images/)
* A USB Microphone or Sound Card

## Installation
The system can be installed with:
```
curl -s https://raw.githubusercontent.com/mcguirepr89/Birding-Pi/rpi4/Birders_Guide_Installer.sh | bash
```

The script first enables and configures the zRAM kernel module for swapping, and reboots. 
After the reboot, the configuration file is opened for editing. Here, you will input your latitude and longitude and will set two passwords to protect your Pi. When in installation has finished, the Birding Pi is ready to start collecting and analyzing data on the next boot. 

If you have trouble with the installation script, you can cancel and rerun the installer:
```
~/Birding-Pi/Birders_Guide_installer.sh
```

## Access
The Birding Pi system can be accessed from any the web browser on the same network:
- http://birdingpi.local

#### Access Credentials:
- Username:`birdnet`
- Password: The "CADDY_PWD" password you set during installation 

## Uninstallation
```
/usr/local/bin/uninstall.sh && cd ~ && rm -drf Birding-Pi
```

## ToDo, Notes, and Comming Soon 

### Internationalisation:
The bird names are in English by default, but other localized versions are available. Please download the labels_l18n.zip file and replace the `model/labels.txt` with the corresponding language.

### Tips:
You can try to overclock your Pi by placing the following in your `/boot/config.txt` file:

```
over_voltage=6
arm_freq=1750
```
Be sure the Pi is adequately cooled and powered. Check http://birdingpi.local:9090 for CPU temperature and over-volatage warnings (anything other than "throttled:0x0" is bad).
