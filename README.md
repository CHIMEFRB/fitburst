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
@article{fpb+23,
       author = {{Fonseca}, Emmanuel and {Pleunis}, Ziggy and {Breitman}, Daniela and {Sand}, Ketan R. and {Kharel}, Bikash and {Boyle}, Patrick J. and {Brar}, Charanjot and {Giri}, Utkarsh and {Kaspi}, Victoria M. and {Masui}, Kiyoshi W. and {Meyers}, Bradley W. and {Patel}, Chitrang and {Scholz}, Paul and {Smith}, Kendrick},
        title = "{Modeling the Morphology of Fast Radio Bursts and Radio Pulsars with fitburst}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2023,
        month = nov,
          eid = {arXiv:2311.05829},
        pages = {arXiv:2311.05829},
          doi = {10.48550/arXiv.2311.05829},
archivePrefix = {arXiv},
       eprint = {2311.05829},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231105829F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Credits
All authors of the `fitburst` papers are the founding developers of `fitburst`, with Emmanuel Fonseca leading the development team. We welcome novel and meaningful contributions from interested users!

This package was built using [namoona](https://github.com/CHIMEFRB/namoona).
