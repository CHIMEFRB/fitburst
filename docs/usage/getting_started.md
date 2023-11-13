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
Any "full `fitburst` pipeline" consists of the following major sections, in order of operation:

0. command-line interface
1. data I/O
2. declaration of initial guess for burst parameters
3. configuration of `fitburst` model object
4. configuration of `fitburst` "fitter" object
5. execution of fit
6. generation of figures, fit-summary files, etc.

Of course, some users may be interested in a "simulation pipeline" which simulates radio pulses with features they wish to model in the presence of controllable noise. We provide two scripts for interested users to get started in using `fitburst` under these conditions. Both scripts are located in the pipeline subdirectory of the `fitburst` codebase. 

One of these scripts will create a simulated, de-dispersed dynamic spectrum. For example, execute the following lines after compiling `fitburst`:

``` python
user@pc> cd /path/to/fitburst/fitburst/pipelines/
user@pc | pipelines> python simulate_burst.py
```

The output of this script is a plot printed to the screen for reference, and a file in the "fitburst-generic" format (`simulated_data.npz`), described in a separate documentation page, that is saved to the local area. This file contains the simulated spectrum shown in the plot, various parameters that describe the context of the spectrum, and an initial guess of model parameters.

The second script is an example of a "full pipeline" version of fitburst that performs I/O, model instantiation, and least-squares fitting:

``` python
user@pc | pipelines> python fitburst_pipeline.py simulated_data.npz --verbose
```

The output of this second script consists of three items: a `.png` file contain a three-panel plot showing the data, best-fit model, and their difference; a JSON file that contains the best-fit parameters and statistics of the fit; and terminal output that shows similar information due to the use of the `--verbose` option:

``` python
INFO: no solution file found or provided; proceeding with fit...
INFO: there are 256 frequencies and 128 time samples.
INFO: there are 256 good frequencies...
INFO: input data cube is already dedispersed!
INFO: setting 'dm' entry to 0, now considered a dm-offset parameter...
INFO: initial guess for 3-component model:
    * amplitude: [0.0, 0.0, 0.0]
    * arrival_time: [0.03, 0.04, 0.05]
    * burst_width: [0.001, 0.002, 0.0005]
    * dm: [0.0, 0.0, 0.0]
    * dm_index: [-2.0, -2.0, -2.0]
    * ref_freq: [1500.0, 1400.0, 1300.0]
    * scattering_index: [-4.0, -4.0, -4.0]
    * scattering_timescale: [0.0, 0.0, 0.0]
    * spectral_index: [0.0, 0.0, 0.0]
    * spectral_running: [-300.0, -300.0, -300.0]
INFO: computing dedispersion-index matrix
INFO: initializing model
INFO: removing the following parameters: dm_index, scattering_index, scattering_timescale
INFO: new list of fit parameters: amplitude, arrival_time, burst_width, dm, spectral_index, spectral_running
0.00000  0.00000  0.03000   -4.00000  0.00000  0.00100 0.00000  -300.00000
0.00000  0.00000  0.04000   -4.00000  0.00000  0.00200 0.00000  -300.00000
0.00000  0.00000  0.05000   -4.00000  0.00000  0.00050 0.00000  -300.00000
0.02077  0.00435  0.02999   -4.00000  0.00000  0.00096 -0.06705  -301.05914
0.02077  -0.01179  0.04000   -4.00000  0.00000  0.00204 0.83079  -295.33102
0.02077  0.02074  0.05001   -4.00000  0.00000  0.00045 0.08439  -272.67249
0.03633  0.00446  0.02999   -4.00000  0.00000  0.00096 -0.06086  -301.33863
0.03633  -0.01175  0.04000   -4.00000  0.00000  0.00204 0.83347  -294.97891
0.03633  0.02339  0.05001   -4.00000  0.00000  0.00046 -0.00992  -275.70777
0.03592  0.00446  0.02999   -4.00000  0.00000  0.00096 -0.06146  -301.29372
0.03592  -0.01178  0.04000   -4.00000  0.00000  0.00204 0.83464  -294.98868
0.03592  0.02348  0.05001   -4.00000  0.00000  0.00046 0.00581  -275.27610
0.03615  0.00446  0.02999   -4.00000  0.00000  0.00096 -0.06141  -301.29554
0.03615  -0.01178  0.04000   -4.00000  0.00000  0.00204 0.83460  -294.98740
0.03615  0.02349  0.05001   -4.00000  0.00000  0.00046 0.00438  -275.31794
INFO: fit successful!
INFO: computing hessian matrix with best-fit parameters.
0.03615  0.00446  0.02999   -4.00000  0.00000  0.00096 -0.06141  -301.29554
0.03615  -0.01178  0.04000   -4.00000  0.00000  0.00204 0.83460  -294.98740
0.03615  0.02349  0.05001   -4.00000  0.00000  0.00046 0.00438  -275.31794
0.03615  0.00446  0.02999   -4.00000  0.00000  0.00096 -0.06141  -301.29554
0.03615  -0.01178  0.04000   -4.00000  0.00000  0.00204 0.83460  -294.98740
0.03615  0.02349  0.05001   -4.00000  0.00000  0.00046 0.00438  -275.31794
INFO: best-fit estimate for 3-component model:
    * amplitude: [0.004458347857993017, -0.01178052452614616, 0.023488538876660896] +/- [0.011361366063813588, 0.0090246635765475, 0.01652378379027083]
    * arrival_time: [0.029986353743391356, 0.040002393748902666, 0.050007330750880916] +/- [2.443033563379835e-05, 4.181234443170404e-05, 1.518622689210296e-05]
    * burst_width: [0.0009574508453823643, 0.002039008450473889, 0.00045508475860427344] +/- [2.403509547560637e-05, 4.033352991181347e-05, 1.906271029135856e-05]
    * dm: [0.036151607836687895] +/- [0.09274487453599654]
    * spectral_index: [-0.06141150018163158, 0.8345952746331111, 0.00437634181414019] +/- [0.6464300408655368, 0.4603917717520832, 0.8428930404305951]
    * spectral_running: [-301.2955381690263, -294.9873984865791, -275.31793520453783] +/- [17.684251952258453, 11.120116046037642, 21.199888485781074]
```

The `fitburst_pipeline.py` script comes with a variety of options that may be useful when working with "real" data, but aren't necessary to use for the data simulated above. This script will work with real data so long as the input file matches the "fitburst-generic" format of the simulated data; explore these options on real data as you see fit!
