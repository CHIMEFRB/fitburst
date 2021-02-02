# Writing Data Readers for `fitburst`

In a pipeline setting, `fitburst` is designed to isolate all telescope-specific dependencies (e.g., the format and structure of local data archiving servers) to the `backend/` subdirectory. Here, users are encouraged to write Python objects -- using the `utilities.bases.ReaderBaseClass` as a foundation -- that enable `fitburst` to work on their raw data. Once all required, telescope-specific data are successfully loaded into a `ReaderBaseClass`-compatible structure, the remaining parts of the full `fitburst` pipeline should work with no issue. (And if there are issues, submit a GitHub issue!)

## The `ReaderBaseClass()` Object

The `ReaderBaseClass()` structure is meant to standardize key attributes and methods to be used for downstream processing within `fitburst`. Users are encouraged to define additional class members that help iniatilize the standard attributes defined in `ReaderBaseClass()`; however, those telescope-specific members will not be detected or used in later operations.

### Key Attributes
- `burst_parameters`: a Python dictionary containing initial guesses of `fitburst` parameters
- `data_full`: a (`num_freq` x `num_time`) Numpy array containing the observed spectrum
- `data_weights`: a list of length `num_freq`, containing boolean values that indicate whether each channel is usable or masked due to RFI
- `dedispersion_idx`: TBD
- `freqs`: a Numpy array containing the frequency centers for each channel
- `num_freq`: the number of frequency channels in `data_full`
- `num_time`: the number of time samples in `data_full`
- `res_freq`: the frequency resolution of `data_full`
- `res_time`: the time resolution of `data_full`
- `times`: a Numpy array containing timestamps for each sample

### Key Methods
- `dedisperse()`:
- `load_data()`:
- `preprocess_data()`:
- `window_data()`:
