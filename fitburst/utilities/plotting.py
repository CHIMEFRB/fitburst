"""
Routines for Creating Plots of Spectra and their Models

This module contains functions for plotting data in various ways,
as well as computing downsampled versions of data for plotting.
"""


# pylint: disable=wrong-import-position
# import and configure matplotlig for GUI-less node.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# now import fitburst-specific things.
import fitburst.routines.manipulate as manip

def compute_downsampled_data(times: float, freqs: float, spectrum_data: float,
    good_freq: float, spectrum_model: float = None, factor_freq: int = 1,
    factor_time: int = 1) -> dict:
    """
    Computes downsampled arrays and matrices and stashes them in a dictionary,
    to be used by plotting funtions below.

    Parameters
    ----------
    times : array_like
        an array of timestamps

    freqs : array_like
        an array of observing frequencies

    spectrum_data : float
        a (num_freq x num_time) matrix containing the dynamic spectrum, where
        num_freq = len(freqs) and num_time = len(times)

    good_freq : float
        an array of boolean values indicating if frequency is usable (True)
        or masked due to RFI (False)

    spectrum_model : float, optional
        a (num_freq x num_time) matrix containing the best-fit model of the dynamic spectrum

    factor_freq : int, optional
        a integer number by which to downsample spectrum in frequency

    factor_time : int, optional
        a integer number by which to downsample spectrum in time

    """

    # pylint: disable=too-many-arguments,too-many-locals

    # downsample spectrum data to desired number of subbands.
    good_freq_downsampled = manip.downsample_1d(good_freq, factor_freq, boolean=True)
    spectrum_data_downsampled = manip.downsample_2d(spectrum_data, factor_freq, factor_time)
    spectrum_data_downsampled *= good_freq_downsampled[:, None]

    # if a model is supplied, then compute the downsampled model and residuals.
    residuals_downsampled = None

    if spectrum_model is not None:
        spectrum_model_downsampled = manip.downsample_2d(spectrum_model, factor_freq, factor_time)
        spectrum_model_downsampled *= good_freq_downsampled[:, None]
        residuals_downsampled = spectrum_data_downsampled - spectrum_model_downsampled

    # downsampled arrays that define axes.
    freqs_downsampled = manip.downsample_1d(freqs, factor_freq)
    times_downsampled = manip.downsample_1d(times, factor_time)

    # compute new resolutions.
    res_freq_downsampled = freqs_downsampled[1] - freqs_downsampled[0]
    res_time_downsampled = times_downsampled[1] - times_downsampled[0]

    # stash in dictionary and return.
    data_downsampled = {
        "data" : spectrum_data_downsampled,
        "freqs" : freqs_downsampled,
        "good_freq" : good_freq_downsampled,
        "model" : spectrum_model_downsampled,
        "num_freq" : len(freqs_downsampled),
        "num_time" : len(times_downsampled),
        "res_freq" : res_freq_downsampled,
        "res_time" : res_time_downsampled,
        "residuals" : residuals_downsampled,
        "times" : times_downsampled,
    }

    return data_downsampled

def plot_summary_triptych(data_dict: dict, num_std: int = 4, output_name: str = "summary.png",
    show: bool = True) -> None:
    """
    Creates a three-panel ("triptych") plot of data and best-fit model/residuals.

    Parameters
    ----------
    data_dict : dict,
        A dictionary containing a collection of data needed for triptych plotting.
        This dictionary should conform to definition used in the 'compute_downsampled_data'
        function above

    num_std : int, optional
        the number of standard deviations in residuals to use in weighting color maps

    output_name : str, optional
        the desired name of the PNG file containing the summary plots

    show : bool, optional
        plots the summary plot

    Returns
    -------
    None : NoneType
        a PNG plot is created and recorded to disk, but the function returns nothing
    """

    # pylint: disable=too-many-locals,too-many-statements

    # get indices of good frequencies
    idx_good_freq = np.where(data_dict["good_freq"])[0]

    # compute bounds of plotting region.
    freq_initial = data_dict["freqs"][0]
    freq_final = data_dict["freqs"][-1]
    min_time = 0.
    max_time = data_dict["num_time"] * data_dict["res_time"] * 1e3
    times_plot = np.linspace(min_time + data_dict["res_time"] / 2,
                             max_time - data_dict["res_time"] / 2, num=data_dict["num_time"])

    # suss out frequency edges correctly.
    if freq_final > freq_initial:
        freq_initial -= data_dict["res_freq"] / 2
        freq_final += data_dict["res_freq"] / 2

    else:
        freq_initial += data_dict["res_freq"] / 2
        freq_final -= data_dict["res_freq"] / 2

    # now set up figure and gridspec axes.
    fig = plt.figure(figsize=(15,12))
    gs_plot = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 3],
                                   hspace=0.0, wspace=0.1)

    panel2d_data = plt.subplot(gs_plot[3])
    panel2d_model = plt.subplot(gs_plot[4], sharey=panel2d_data)
    panel2d_residual = plt.subplot(gs_plot[5], sharey=panel2d_data)
    panel1d_data = plt.subplot(gs_plot[0], sharex=panel2d_data)
    panel1d_model = plt.subplot(gs_plot[1], sharex=panel2d_model)
    panel1d_residual = plt.subplot(gs_plot[2], sharex=panel2d_residual)

    # before proceeding, compute heatmap min/max ranges.
    residual_median = np.nanmedian(data_dict["residuals"])
    residual_std = np.nanstd(data_dict["residuals"])
    vmin = residual_median - residual_std * num_std
    vmax = residual_median + residual_std * num_std
    extent = [min_time, max_time, freq_initial, freq_final]

    # plot data panels first.
    timeseries = np.nanmean(data_dict["data"][idx_good_freq], axis=0)
    timeseries *= np.sqrt(np.count_nonzero(
        ~np.isnan(np.nansum(data_dict["data"][idx_good_freq], axis=-1))))

    panel1d_data.plot(times_plot, timeseries)
    panel2d_data.imshow(data_dict["data"], origin="lower", aspect="auto",
                        interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax)

    # extract y-axis limits from data panel for model and residual panels.
    y_min, y_max = panel1d_data.get_ylim()
    panel1d_model.set_ylim(y_min, y_max)
    panel1d_residual.set_ylim(y_min, y_max)

    # plot model panel.
    if data_dict["model"] is not None:
        timeseries = np.nanmean(data_dict["model"][idx_good_freq], axis=0)
        timeseries *= np.sqrt(np.count_nonzero(
            ~np.isnan(np.nansum(data_dict["model"][idx_good_freq], axis=-1))))

        panel1d_model.plot(times_plot, timeseries)
        panel2d_model.imshow(data_dict["model"], origin="lower", aspect="auto",
                             interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax)

    # plot residual panel.
    if data_dict["residuals"] is not None:
        timeseries = np.nanmean(data_dict["residuals"][idx_good_freq], axis=0)
        timeseries *= np.sqrt(np.count_nonzero(
            ~np.isnan(np.nansum(data_dict["residuals"][idx_good_freq], axis=-1))))

        panel1d_residual.plot(times_plot, timeseries)
        panel2d_residual.imshow(data_dict["residuals"], origin="lower", aspect="auto", cmap="bwr",
                                interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax)

    # remove the appropriate 2D labels and tickmarks.
    plt.setp(panel2d_model.get_yticklabels(), visible=False)
    plt.setp(panel2d_residual.get_yticklabels(), visible=False)

    # remove labels and tickmarks for timeseries panels.
    plt.setp(panel1d_data.get_xticklabels(), visible=False)
    panel1d_data.set_yticklabels([], visible=True)
    panel1d_data.set_yticks([])
    panel1d_data.set_xlim(min_time, max_time)

    plt.setp(panel1d_model.get_xticklabels(), visible=False)
    panel1d_model.set_yticklabels([], visible=True)
    panel1d_model.set_yticks([])
    panel1d_model.set_xlim(min_time, max_time)

    plt.setp(panel1d_residual.get_xticklabels(), visible=False)
    panel1d_residual.set_yticklabels([], visible=True)
    panel1d_residual.set_yticks([])
    panel1d_residual.set_xlim(min_time, max_time)

    # add data/model/residual labels.
    x_pos = (max_time - min_time) * 0.96 + min_time
    y_pos = (y_max - y_min) * 0.90 + y_min
    panel1d_data.text(x_pos, y_pos, "D", ha="right", va="top", fontsize=8)
    panel1d_model.text(x_pos, y_pos, "M", ha="right", va="top", fontsize=8)
    panel1d_residual.text(x_pos, y_pos, "R", ha="right", va="top", fontsize=8)
    panel2d_data.set_xlabel("Time [ms]")
    panel2d_data.set_ylabel("Frequency [MHz]")
    panel2d_model.set_xlabel("Time [ms]")
    panel2d_residual.set_xlabel("Time [ms]")

    # now save figure.
    plt.savefig(output_name, dpi=150, bbox_inches="tight")

    # if desired, print figure to screen.
    if show:
        plt.show()

    # finally, close figure.
    plt.close(fig)
