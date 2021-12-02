The `fitburst` codebase uses a `DataReader` object for the following purposes:

1. to ensure that all attributes necessary for downstream analysis are initialized;
2. to provide methods for normalizing and (incoherently) dedispersing the input spectrum, if necessary;
3. to allow for experiment-specific modularity in pipeline settings.

## A Base Class for Data Readers
A `DataReader` is a child of the `ReaderBaseClass` object. The `ReaderBaseClass` object defines all key attributes and nearly all methods used in a typical execution of `fitburst`. One method of the base class, `load_data()`, is intentionally left undefined as it depends on the nature of the input data.

## A Data Reader for the "Generic" Format
As an example of how to use the `ReaderBaseClass`, we have provided a `DataReader` that parses the "generic" data format discussed on the preceding page. It can be imported and invoked in the following way:

``` python
from fitburst.backend.generic import DataReader

# read in data stored in the "generic" format.
data = DataReader()
data.load_data("input_data.npz")
```

## Customizing Data Readers
The `DataReader` example shown above loads in all data from the input file into the various attributes instantiated by the `ReaderBaseClass` object. However, it is important to note that all input-dependent steps (e.g., the file format) are encapsulated in the `load_data()` method only. It is therefore possible to modularize `fitburst` such that the algorithm can work for a wide range of data formats. The only necessary development would be in creating a new `DataReader` that can correclty parse the input data format.

## Cleaning and Flagging Data
There is a `preprocess_data()` method in the `ReaderBaseClass` object that normalizes and baseline-subtracts each channel and determines a set of "good" frequencies used by downstream fitting routines. The determination of good and bad frequencies is based on outliers of variance and skewness distributions for the time-averaged spectrum.

All options for the `preprocess_data()` method are optional. Below is an example of its invocation with all arguments set to their default values:

``` python
# now apply cleaning algorithm.
data.preprocess_data(
    normalize_variance = True,
    skewness_range = [-3., 3.],
    variance_range = [0.2, 0.8], 
    variance_weight = 1.,
)
```

The above method call with replace the original, raw spectrum stored in `data.data_full` with the normalized, cleaned spectrum. Also, the above spectrum will overload the `data.good_freqs` attribute with a list of booleans that indicate frequencies which are deemed useable (`True`) or unusable (`False`).

## Retrieving Burst Parameters
The generic-format data stores previous estimates of the burst parameters in the `.npz` data file. The `DataReader` for the generic format then stores these parameters as a Python dicitonary:

``` python
# now extract parameters from npz file.
initial_parameters = data.burst_parameters
print("DM values: ", initial_parameters["dm"])
```
Let's assume that the `input_data.npz` file contains data for a three-component burst from FRB 121102, where the burst-averaged DM was previously found to be 557.0 pc cm$^{-3}$. If the `.npz` file was generated correctly, then the above `print()` statement should show:

``` python
DM values: [557.0, 557.0, 557.0]
```

## Dedispersing and/or Windowing the Input Spectrum
The `ReaderBaseClass` contains two algorithms for de-dispersing and windowing the raw spectrum data. These methods are optional, but may be necessary if the input spectrum spans several seconds of data, and/or if the data are either dispersed or de-dispersed to a suboptimal DM value. In all cases, the `data.is_dedispersed` attribute must accurately reflect whether the input spectrum is already de-dispersed (`True`) or not (`False`).


Here's an example of a de-dispersion call:

``` python
data.dedisperse(
    initial_parameters["dm"][0],
    initial_parameters["arrival_time"][0],
    reference_freq = initial_parameters["ref_freq"][0]
)
```

The above call will use the input values and axes information (e.g., `data.freqs`. `data.times`, etc.) to compute a map of de-dispersion index values. These index values are then used by the `window_data` method to obtain a "windowed" (i.e., zoomed-in) version of the de-dispersed spectrum:

``` python
window = 0.08 # in seconds

# before doing anything, check if window size doesn't extend beyond data set.
# if it does, adjust down by an appropriate amount.
window_max = data.times[-1] - initial_parameters["arrival_time"][0]

if window > window_max:
    window = window_max - 0.001
    print("INFO: window size adjusted to +/- {0:.1f} ms".format(window * 1e3))

data_windowed, times_windowed = data.window_data(params["arrival_time"][0], window=window)
```
