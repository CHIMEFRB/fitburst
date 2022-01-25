The `fitburst` codebase can be used in two different ways: as a Python package; or through Python3 scripts provided in the repository. This page describes how to interact with the two usage modes.

## `fitburst` as a Python package
Once properly installed, `fitburst` can be immediate accessed through a Python interpreter as an importable package:

``` python
user@pc > python
>>> import fitburst
>>>
```

All underlying objects and functions can be accessed in this package format, with Python docstrings that emulate the `numpy` documentation format.

## Example `fitburst` Pipelines
The full `fitburst` pipeline consists of the following major sections, in order of operation:

0. command-line interface
1. data I/O
2. declaration of initial guess for burst parameters
3. configuration of `fitburst` model object
4. configuration of `fitburst` "fitter" object
5. execution of fit
6. generation of figures, fit-summary files, and/or convergence status.

We have provided several "pipeline" scripts as demonstrations of `fitburst` as an importable package. These Python3 scripts will perform a weighted least-squares fit of a model against different types of existing data formats. For example, a user with data in the `fitburst`-compliant "generic" format (outlined in this documation) can run the `fitburst_example_generic.py` script with the `--verbose` option if more output is desired:

``` python
> python /path/to/fitburst_example_generic.py /location/of/input_data.npz --verbose
INFO: no solution file found or provided; proceeding with fit...
INFO: input data cube is already dedispersed!
INFO: setting 'dm' entry to 0, now considered a dm-offset parameter...
INFO: initial guess for 1-component model:
    * ref_freq: [406.95]
    * arrival_time: [0.5]
    * dm: [0.0]
    * burst_width: [0.05]
    * amplitude: [-3.0]
    * scattering_timescale: [0.0]
    * dm_index: [-2.0]
    * scattering_index: [-4.0]
    * spectral_index: [-1.0]
    * spectral_running: [1.0]
INFO: computing dedispersion-index matrix
INFO: initializing model
INFO: dimensions of model spectrum set to (32, 256)
INFO: removing the following parameters: dm_index, scattering_index, scattering_timescale
INFO: new list of fit parameters: amplitude, arrival_time, burst_width, dm, spectral_index, spectral_running
INFO: fit successful!
INFO: derived uncertainties and fit statistics
INFO: best-fit estimate for 1-component model:
    * amplitude: [-6.324027826884258] +/- [0.06858211715116933]
    * arrival_time: [0.09509827951169918] +/- [0.004102709304261897]
    * burst_width: [0.02668654193357436] +/- [0.0011340937579950628]
    * dm: [0.7976612513974163] +/- [0.28790218747669244]
    * spectral_index: [3.0854722262265004] +/- [0.8459390426969531]
    * spectral_running: [-3.555893925137327] +/- [1.0570358282340104]
Data vmin, vmax = -0.00, 0.00
Model vmin, vmax = -0.00, 0.00
```
