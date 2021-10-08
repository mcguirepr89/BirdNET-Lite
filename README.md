# Birding Pi
![version](https://img.shields.io/badge/version-0.1-blue)

A realtime acoustic bird classification system for the Reaspberry Pi 4B

## Inroduction
The birding pi project is based on the BirdNet-Lite project and is able to recognize bird sounds from the microphne and the sound card in realtime. The system installs all needed services on the rasperi pi. 

## Requirements
* A Raspberry Pi 4B
* A SD Card with the 64-bit version of Raspberi OS installed (download link)
* A USB Microphone or Sound Card

## Installation
The system can be installed with:

`curl -s https://raw.githubusercontent.com/mcguirepr89/BirdNET-Lite/rpi4/Birders_Guide_Installer.sh | bash`

The script installs the `Birding-Pi` system in the Home folder and restartes the system. Also the geo coordinates are required to fullfil the configuration file including some passowrds. Restart the rasperry pi anter the installer script has finished. If you have trouble with the installation script you can restart the installation at any time while running the script again `~/Birding-Pi/Birders_Guide_installer.sh`.

## Access
The Birding Pi system can be accessed with the webbrowser:

```http://birdnetsystem.local```

## Uninstallation
`/usr/local/bin/uninstall.sh && cd ~ && rm -drf Birding-Pi`

## ToDo, Notes, and Comming Soon 

### Internationalisation:
The bird names are by default in english, but there are also localozed versions availabe. Please download the labels_l18n.zip file and replace the `model/labels.txt` with the corresponding language.

### Tips:
You try to overclock your raspery with:

```
over_voltage=6
arm_freq=1750
```
But be aware of the stabillity and the power consumption.
