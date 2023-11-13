We have developed a Python class, called the `SpectrumModeler`, for generating models of dynamic spectra. The `SpectrumModeler` is designed for interaction with downstream fitting routines; however, it can nonetheless be used as a standalone object. The simplest version of a call to the `SpectrumModeler` is given here: 

``` python
from fitburst.analysis.model import SpectrumModeler
freqs = ... # array of frequency labels, in MHz
times = ... # array of timestamps, in seconds
model = SpectrumModeler(freqs, times)
```

where (`freqs`, `times`) define the centers of frequency channels and time bins, respectively.

## Parameters of the Model Object
As described in the `fitburst` paper, the `SpectrumModeler` is a function of nine fittable parameters. A tenth parameter, called `ref_freq` cannot be fitted as it serves as a frequency to which the amplitude and SED parameters are referenced. The full list of parameters can be retrieved as follows:

``` python
>>> print(model.parameters)
['amplitude', 'arrival_time', 'burst_width', 'dm', 'dm_index', 'ref_freq', 'scattering_timescale', 'scattering_index', 'spectral_index', 'spectral_running']
```

Please refer to Section 2 and Table 1 of the `fitburst` paper for a description of these parameters.

## Loading Parameter Values into the Model Object

In order to load parameter values, we use the `update_parameters()` method of the `SpectrumModeler`:

``` python
# define dictiomary containing parameter values.
burst_parameters = {
    "amplitude"            : [0.],
    "arrival_time"         : [0.04],
    "burst_width"          : [0.003],
    "dm"                   : [349.5],
    "dm_index"             : [-2.],
    "ref_freq"             : [1400.],
    "scattering_index"     : [-4.],
    "scattering_timescale" : [0.],
    "spectral_index"       : [10.],
    "spectral_running"     : [-100.],
}

# now instantiate the SpectrumModeler
model = SpectrumModeler(freqs, times)

# update the SpectrumModeler to use the above values.
model.update_parameters(burst_parameters)

# slightly adjust the DM only, leaving all others unchanged in the model object.
model.update_parameters({"dm": [348.95]})
```

The `update_parameters()` method receives a dictionary with one or more parameters with values loaded in Python lists, as shown in the second method call above. This feature exists to allow for flexibility in generating models for fitting where one or more parameters are fixed to pre-determined values.

## Generating Mulit-Component Models

The `SpectrumModeler` is also capable of generating models of a multi-component spectrum, i.e., a dynamic spectrum with $N$ distinct pulses. Such models can be created with the same code above, but with values of the `burst_parameters` dicitionary that are lists of length $N$. For example, the following code will overwrite the above parameters to instantiate a model with 3 components:

```python
# define dictiomary containing parameter values.
burst_parameters = {                                                     
    "amplitude"            : [0., 0., 0.], 
    "arrival_time"         : [0.03, 0.04, 0.05],
    "burst_width"          : [0.001, 0.003, 0.0005],
    "dm"                   : [349.5, 349.5, 349.5], 
    "dm_index"             : [-2., -2., -2.],
    "ref_freq"             : [1400., 1400., 1400.],
    "scattering_index"     : [-4., -4., -4.],
    "scattering_timescale" : [0., 0., 0.],
    "spectral_index"       : [10., 0., -10.],
    "spectral_running"     : [-100., -100., -100.],
}

# now instantiate the SpectrumModeler for a three-component model.
num_components = len(burst_parameters["dm"])
model = SpectrumModeler(freqs, times, num_components = num_components)
                    
# now update Gaussian-SED model object to use the above values.
model.update_parameters(burst_parameters)

``` 

## Creating Models for De-dispersed Data
Users will typically want to fit models of dynamic spectra against data that are already de-dispersed. The `SpectrumModeler` can be used as shown above, but with one caveat: the `dm` parameter is treated as a "DM offset" for de-dispersed spectra, instead of the "full" DM whose values are supplied in the above examples. Once this configuration is done, we can then compute a model spectrum with the `compute_mode()` method within the `SpectrumModeler`.

The following code with use the latest example above and perform the adjustment needed for generating a de-dispersed dynamic spectum:

``` python
# indicate whether the spectrum is de-dispersed or not.
is_dedispersed = True

# define dictiomary containing parameter values.
burst_parameters = {                                                     
    "amplitude"            : [0., 0., 0.], 
    "arrival_time"         : [0.03, 0.04, 0.05],
    "burst_width"          : [0.001, 0.003, 0.0005],
    "dm"                   : [349.5, 349.5, 349.5], 
    "dm_index"             : [-2., -2., -2.],
    "ref_freq"             : [1400., 1400., 1400.],
    "scattering_index"     : [-4., -4., -4.],
    "scattering_timescale" : [0., 0., 0.],
    "spectral_index"       : [10., 0., -10.],
    "spectral_running"     : [-100., -100., -100.],
}

# adjust DM value to zero offset, if necessary.
num_components = len(burst_parameters["dm"])

if is_dedispersed:
    burst_parameters["dm"] = [0.] * num_components

# now instantiate the SpectrumModeler for a three-component model.
model = SpectrumModeler(freqs, times, num_components = num_components)

# grab the model spectrum.
spectrum_model = model.compute_model()
```

The above call with return a NumPy `ndarray` with shape `(num_freq, num_time)`.

## Creating Models for Dispersed Data
In rare or simulation cases, it may be desired to create a dispersed dynamic spectrum. This spectrum can be generated using the latest example above, but instead setting `is_dedispersed = False`.
