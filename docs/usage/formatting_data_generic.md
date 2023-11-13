For ease of use, we have defined a `fitburst`-compliant ("generic") data format for loading all required data into the `fitburst` data-reading object. Users can adopt this generic format to ensure initialization of required variables and arrays used within `fitburst`. The generic-format data are stored in and read from a Python3 Numpy `.npz` file.

## Concept of Generic Format
A generic-compatible data file, e.g., "input\_data.npz", is assumed to contain three entries:

- `metadata`: a Python dictionary containing data that describe aspects of the observation;
- `burst_parameters`: a Python dictionary containing intrinsic burst parameters and their pre-determined values or initial guesses;
- `data_full`: a NumPy `ndarray` containing the dynamic spectrum.

These three `.npz` arrays are defined further below. Once defined, the file can be created by executing the following lines in your local data-preparation script:

``` python
import numpy as np

# define data_full, metadata and burst_parameter objects as needed.
# ...

# now write data to file.
np.savez(
    "input_data.npz", 
    data_full=data_full, 
    metadata=metadata, 
    burst_parameters=burst_parameters
)
```

## Required Metadata
The following data in the `metadata` dictionary are used for configuration of internal arrays, and are not parameters considered for least-squares optimization.

``` python
metadata = {
    "bad_chans"      : # a Python list of indices corresponding to frequency channels to zero-weight
    "freqs_bin0"     : # a floating-point scalar indicating the value of frequency bin at index 0, in MHz
    "is_dedispersed" : # a boolean indicating if spectrum is already dedispersed (True) or not (False)
    "num_freq"       : # an integer scalar indicating the number of frequency bins/channels
    "num_time"       : # an integer scalar indicating the number of time bins
    "times_bin0"     : # a floating-point scalar indicating the value of time bin at index 0, in MJD
    "res_freq"       : # a floating-point scalar indicating the frequency resolution, in MHz
    "res_time"       : # a floating-point scalar indicating the time resolution, in seconds
}
```

## Required Burst Parameters
Unless otherwise noted, all data in the `burst_parameters` dictionary are used as initial guesses and are subject to least-squares optimization. (Only the `ref_freq` parameter is held fixed permanently; all other parameters can be fitted or held fixed at the discretion of the user.) As with other "initial-guess" problems, we recommended that as many initial guesses as possible be "good enough" based on prior information. However, the `fitburst` API allows for initial-guess modifications at later, pre-fit stages. 

All dictionary keys contain Python lists of floating-point values. The number of list elements is equal to the number of burst components to be modeled. For example, a "simple" burst described with a single profile will have `burst_parameters` entries with `len(burst_parameters[name]) = 1`.

``` python
burst_parameters = {
    "amplitude"            : # a list containing the the log (base 10) of the overall signal amplitude
    "arrival_time"         : # a list containing the arrival times, in seconds
    "burst_width"          : # a list containing the temporal widths, in seconds
    "dm"                   : # a list containing the dispersion measures (DM), in parsec per cubic centimeter
    "dm_index"             : # a list containing the exponents of frequency dependence in DM delay
    "ref_freq"             : # a list containing the reference frequencies for arrival-time and power-law parameter estimates, in MHz (held fixed)
    "scattering_index"     : # a list containing the exponents of frequency dependence in scatter-broadening
    "scattering_timescale" : # a list containing the scattering timescales, in seconds
    "spectral_index"       : # a list containing the power-law spectral indices
    "spectral_running"     : # a list containing the power-law spectral running
}
```

## Required Spectrum
The `data_full` entry contains the observed dynamic spectrum as a Numpy `ndarray` with shape `(num_freq, num_time)`.
