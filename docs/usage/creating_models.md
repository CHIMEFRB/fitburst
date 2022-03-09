We have developed a Python class, called the `SpectrumModeler`, for generating data models in a manner suitable for interaction with downstreaming fitting routines. The simplest version of a call to the model class is, 

``` python
>>> from fitburst.analysis.model import SpectrumModeler
>>> model = SpectrumModeler(num_freq, num_time)
```

where the quantities (`num_freq`, `num_time`) define the dimensions of the model spectrum.

## Parameters of the Model Object
The exact list of parameters will depend on the assumed shape of the spectral energy distribution (SED). By default, the `SpectrumModeler` assumes a "running power-law" model for the SED, and so the spectral parameters will be the `spectral_index` and `spectral_running`. The full list of parameters can be retrieved as follows:

``` python
>>> print(model.parameters)
['amplitude', 'arrival_time', 'burst_width', 'dm', 'dm_index', 'scattering_timescale', 'scattering_index', 'spectral_index', 'spectral_running']
```

If a Gaussian SED is instead desired, then you can instantiate the `SpectrumModeler` and set the correct option to indicate this choice:

``` python
>>> model = SpectrumModeler(num_freq, num_time, freq_model = "gaussian")
>>> print(model.parameters)
['amplitude', 'arrival_time', 'burst_width', 'dm', 'dm_index', 'scattering_timescale', 'scattering_index', 'freq_mean', 'freq_width']
```

Notice that all but two parameters have changed, and that the `freq_mean` and `freq_width` now characterize the (Gaussian) shape of the SED. So far, only the `powerlaw` and `gaussian` SED models are available in `fitburst`.

## Loading Parameter Values into the Model Object

The above calls to the `SpectrumModeler` object yield an "empty" model object; the object is configured but contains no information on the model parameters. In order to load parameter values, we use the `update_parameters()` method of the `SpectrumModeler`:

``` python
# define dictiomary containing parameter values.
burst_parameters = {
    "amplitude"            : [0.0],
    "arrival_time"         : [0.5],
    "burst_width"          : [0.005],
    "dm"                   : [557.0],
    "dm_index"             : [-2.0],
    "ref_freq"             : [600.0],
    "scattering_index"     : [-4.0],
    "scattering_timescale" : [0.0],
    "freq_mean"            : [450.0],
    "freq_width"           : [43.0],
}

# now update Gaussian-SED model object to use the above values.
model.update_parameters(burst_parameters)

# adjust the DM value while leaving all others unchanged in the model object.
model.update_parameters({"DM": [557.5]})
```

The `update_parameters()` method is able to receive a dictionary that contains only one or a partial set of the full parameter list, as shown in the second method call above. This feature is important when fitting models to data where one or more model parameters are desired to be fixed to pre-determined values.

The `SpectrumModeler` is also capable of generating models of a multi-component spectrum. The only changed needed for this to occur is for the values of the above `burst_parameters` dicitionary to be lists of length greater than 1, where the $i$th element for each list corresponds to parameters of "sub-burst" $i$.

## Creating Models for De-dispersed Data
Once the above configuration is done, we can then compute a model spectrum with the `compute_mode()` method within the `SpectrumModeler`. 

``` python
freqs = ... # array of frequency labels, in MHz
times = ... # array of timestamps, in seconds

spectrum_model = model.compute_model(times, freqs)
```

If you're using a `DataReader` to load and prepare data, then the above arrays can be accessed through the `freqs` and `times` attributes.

## Creating Models for Dispersed Data
