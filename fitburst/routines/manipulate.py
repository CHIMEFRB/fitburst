import numpy as np

def downsample_1d(array_orig: np.ndarray, factor: int, boolean: bool = False):
    """
    Downsamples an input array by the specified factor. It is assumed that the 
    input array is regularly sampled (i.e., the difference in adjacent bins 
    have the same value.)
    """

    # reshape original array and marginalize.
    
    num_elements_orig = len(array_orig)
    array_reshaped = np.reshape(array_orig, (num_elements_orig // factor, factor))
    array_downsampled = np.sum(array_reshaped, axis=1) / factor

    if boolean:
        array_downsampled = array_downsampled.astype(bool)

    return array_downsampled

def downsample_2d(spectrum_orig: np.ndarray, factor_freq: int, factor_time: int):
    """
    Downsamples a two-dimensional dynamic spectrum and its time/frequency arrays 
    by specified factors.
    """

    # compute original and new matrix shapes.
    num_freq, num_time = spectrum_orig.shape
    shape_new = (num_freq // factor_freq, factor_freq, num_time // factor_time, factor_time)

    # now reshape and average to downsample.
    spectrum_downsampled = spectrum_orig.reshape(shape_new).mean(-1).mean(1)

    return spectrum_downsampled

def upsample_1d(array_orig: float, diff_orig: float, factor: int):
    """
    Upsamples an input array by the specified factor. It is assumed that the 
    input array is regularly sampled (i.e., the difference in adjacent bins 
    have the same value).
    Parameters
    ----------
    array_orig : array_like
        a one-dimensional array of values to be degraded to lower resolution 
    diff_orig : float
        the resolution of the original array
    factor : int
        an upsampling factor 
    Returns
    -------
    array_downsampled : array_like
        the new, downsampled array
    """

    # define bounds of upsampled array.
    diff_new = diff_orig / factor
    bound_lo = array_orig[0] - (diff_orig / 2) + (diff_new / 2)
    bound_hi = array_orig[-1] + (diff_orig / 2) - (diff_new / 2)

    # now create upsampled version of input array.
    num_new = len(array_orig) * factor
    array_upsampled = np.linspace(bound_lo, bound_hi, num=num_new)

    return array_upsampled


def upsample_orig(input_array, factor):
    """
    Upsamples an input array by the specified factor. It is assumed that the 
    input array is regularly sampled (i.e., the difference in adjacent bins 
    have the same value.)
    Parameters
    ----------
    array_orig : array_like
        a one-dimensional array of values to be degraded to lower resolution 
    diff_orig : float
        the resolution of the original array
    factor : int
        an upsampling factor 
    Returns
    -------
    array_downsampled : array_like
        the new, upsampled array
    Notes 
    -----
        This algorithm is used in the 'original' version of fitburst.
    """

    # compute difference in original array.
    res_orig = np.diff(input_array)[1]

    # now compute new base array with appropriate resolution.
    new_array = (np.arange(factor, dtype=float) + 0.5) / factor - 0.5
    new_array *= res_orig

    # finally, add together to get units correct.
    output_array = input_array[:, None] + new_array[None, :]

    return output_array