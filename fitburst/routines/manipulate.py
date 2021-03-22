import numpy as np

def downsample_1d(array_orig: np.ndarray, factor: int, boolean: bool = False):
    """
    Downsamples an input array by the specified factor. It is assumed that the 
    input array is regularly sampled (i.e., the difference in adjacent bins 
    have the same value.)
    """

    # reshape original array and marginalize.
    num_elements = len(array_orig)
    array_reshaped = np.reshape(array_orig, (num_elements // factor, factor))
    array_downsampled = np.sum(array_reshaped, axis=1) / factor

    if boolean:
        array_downsampled = array_downsampled.astype(bool)

    return array_downsampled

def downsample_2d(spectrum_orig: np.ndarray, factor: int, axis: str = "freq"):
    """
    Downsamples a two-dimensional dynamic spectrum and its time/frequency arrays 
    by specified factors.
    """

    # compute original and new matrix shapes.
    num_freq, num_time = spectrum_orig.shape
    shape_new = (num_freq // factor, factor, num_time)

    if axis == "time":
        shape_new = (num_freq, factor, num_time // factor)

    # now reshape and average to downsample.
    spectrum_reshaped = np.reshape(spectrum_orig.copy(), shape_new)
    spectrum_downsampled = np.sum(spectrum_reshaped, axis=1) / factor

    return spectrum_downsampled

def upsample(input_array, factor):
    """
    Upsamples an input array by the specified factor. It is assumed that the 
    input array is regularly sampled (i.e., the difference in adjacent bins 
    have the same value.)
    """

    # compute difference in original array.
    res_orig = np.diff(input_array)[1]

    # now compute new base array with appropriate resolution.
    new_array = (np.arange(factor, dtype=float) + 0.5) / factor - 0.5
    new_array *= res_orig

    # finally, add together to get units correct.
    output_array = input_array[:, None] + new_array[None, :]

    return output_array
