fitburst
========

This repository contains functions, objects, and scripts for modeling dynamic spectra of dispersed astrophysical signals at radio frequencies.

## Installation

Currently, `fitburst` can be installed by cloning the repo and running `pip` in the following way:

```
pc> git clone git@github.com:CHIMEFRB/fitburst.git
pc> cd fitburst
pc/fitburst> pip install . # add the --user option if you're looking to install in your local environment.
```

## Usage and Documentation
Please refer to the documentation linked above to find desciptions on the codebase and examples for interacting with it. 

## Publication
The theory behind the modeling and analysis routines is presented in a publication currently under review, [but available on the arXiv](https://arxiv.org/abs/2311.05829). This publication includes a variety of fitting examples and discussions on the treatment of biasing effects (e.g., intra-channel smearing from pulse dispersion) that can be accounted for within `fitburst`. If you use this codebase and publish results obtained with it, we ask that you cite the `fitburst` publication using the following BibTex entry:

``` python
@article{fpb+24,
       author = {{Fonseca}, E. and {Pleunis}, Z. and {Breitman}, D. and {Sand}, K.~R. and {Kharel}, B. and {Boyle}, P.~J. and {Brar}, C. and {Giri}, U. and {Kaspi}, V.~M. and {Masui}, K.~W. and {Meyers}, B.~W. and {Patel}, C. and {Scholz}, P. and {Smith}, K.},
        title = "{Modeling the Morphology of Fast Radio Bursts and Radio Pulsars with fitburst}",
      journal = {\apjs},
     keywords = {Pulsars, Radio transient sources, Interstellar medium, Interstellar scintillation, Astronomy software, 1306, 2008, 847, 855, 1855, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = apr,
       volume = {271},
       number = {2},
          eid = {49},
        pages = {49},
          doi = {10.3847/1538-4365/ad27d6},
archivePrefix = {arXiv},
       eprint = {2311.05829},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJS..271...49F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Credits
All authors of the `fitburst` papers are the founding developers of `fitburst`, with Emmanuel Fonseca leading the development team. We welcome novel and meaningful contributions from interested users!

This package was built using [namoona](https://github.com/CHIMEFRB/namoona).
