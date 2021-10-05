fitburst
========

This repo contains functions and objects for modeling dynamic spectra of dispersed astrophysical signals at radio frequencies.  

## Installation

Currently, `fitburst` can be installed by cloning the repo and running `pip` in the following way:

``` 
pc> git clone git@github.com:CHIMEFRB/fitburst.git
pc> cd fitburst
pc/fitburst> pip install . # add the --user option if you're looking to install in your local environment.
```

## Usage

Once installed, the `fitburst` functionality can be imported as a Python package for various custom purposes. For example, if you wish to simply read in CHIME/FRB metadata only (e.g., parameter estimates made by the various online pipelines) for a specific event, you could do the following:

```
pc> python
>>> from fitburst.backend.chimefrb import DataReader
>>> data = DataReader(48851362)
....
>>> print(data.burst_parameters) # a dictionary of pipeline-specific parameters
>>> print(data.files) # a list of filenames for the total-intensity data set
```

There are also several example scripts available in the `fitburst/pipelines` section of the repo that utilize the `fitburst` package in slightly different ways. The current script used to analyze CHIME/FRB data (`fitburst/pipelines/fitburst_example_chimefrb.py`) comes with a variety of options for interacting with the full algorithm at the command line. For example, if you wish to run the full `fitburst` pipeline on an event (i.e., data I/O and pre-processing, setup and fitting of model against pre-processed intensity data) using the intensity-DM pipeline parameters as your initial guess, and ignoring the fit of scattering parameters, you should run:
```
pc> python /path/to/fitburst_example_chimefrb.py 48851362 --pipeline dm
```

If you wish to change the size of the windowed spectrum, fit for a scattering timescale of a thin-screen model, and toggle its initial guess to a value of your choosing, you should instead run:

```
pc> python /path/to/fitburst_example_chimefrb.py 48851362 --pipeline dm --window 0.12 --fit scattering_timescale --scattering_timescale 0.05
```

Use the `-h` option in the above script to see all available options and units for various numerical quantities.
