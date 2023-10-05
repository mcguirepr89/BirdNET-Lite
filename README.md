## BatNET-Pi - Automated real-time bat detector

**Note: The system is under heavy development and not (as yet) fit for production use. You are welcome to try it and send feedback.**
## Purpose

## Features

* Scans ultrasound with 256kHz sampling rate continuously 24/7 
* Automated bat ID sing the companion https://github.com/rdz-oss/BatNET-Analyzer.
* Inherits many great things from BirdNET-Pi
* Right now only enabled for European bat species
* US species will be added soon

### License

Enjoy! Feel free to use BatNET-Pi for your acoustic analyses and research. If you do, please cite as:
``` bibtex
@misc{Zinck2023,
  author = {Zinck, R.D.},
  title = {BatNET-PI: Automated real-time bat detector},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rdz-oss/BatNET-Pi }}
}
```

LICENSE: http://creativecommons.org/licenses/by-nc-sa/4.0/  
Also consider the references at the end of the page.


## Install
You can follow the instructions for installing BirdNET-Pi to the point of flashing the sd card with the operating system. After that you will need to call
the install script from this repository:
```sh
curl -s https://raw.githubusercontent.com/rdz-oss/BattyBirdNET-Pi/main/newinstaller.sh | bash
```

### Screenshot
Overview page
![main page](homepage/images/BatNET-Pi-Screen.png "Main page")

Including spectrograms to 128 kHz
![main page](homepage/images/BatNET-Pi-Screen-2.png "Main page")

## References

### Papers

FROMMOLT, KARL-HEINZ. "The archive of animal sounds at the Humboldt-University of Berlin." Bioacoustics 6.4 (1996): 293-296.

Görföl, Tamás, et al. "ChiroVox: a public library of bat calls." PeerJ 10 (2022): e12445.

Gotthold, B., Khalighifar, A., Straw, B.R., and Reichert, B.E., 2022, 
Training dataset for NABat Machine Learning V1.0: U.S. Geological Survey 
data release, https://doi.org/10.5066/P969TX8F.

Kahl, Stefan, et al. "BirdNET: A deep learning solution for avian diversity monitoring." Ecological Informatics 61 (2021): 101236.

### Links

https://www.museumfuernaturkunde.berlin/en/science/animal-sound-archive

https://www.chirovox.org/

https://www.sciencebase.gov/catalog/item/627ed4b2d34e3bef0c9a2f30

https://github.com/kahst/BirdNET-Analyzer
