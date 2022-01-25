# import general packages.
import numpy as np
import copy
import sys

# import and configure matplotlig for GUI-less node.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# now import fitburst-specific things.
import fitburst.routines.manipulate as manip

def plot_summary_triptych(times: np.ndarray, freqs: np.ndarray, spectrum_orig: np.ndarray, 
    mask_freq: np.ndarray, factor_time: int = 1, model: np.ndarray = None, factor_freq: int = 1, 
    num_std: int = 4, output_name: str = "summary.png", residuals: np.array = None, show: bool = True) -> None:
    """
    Creates a three-panel ("triptych") plot of data and best-fit model/residuals.

    Parameters
    ----------
    times : np.ndarray
        an array of timestamps

    freqs : np.ndarray 
        an array of observing frequencies

    spectrum_orig : np.ndarray
        a (num_freq x num_time) matrix containing the dynamic spectrum, where num_freq = len(freqs) 
        and num_time = len(times)

    mask_freq : np.ndarray 
        an array of boolean values indicating if frequency is usable (True) or masked (False)

    factor_time : int, optional
        a integer number by which to downsample spectrum in time

    model : np.ndarray, optional
        a (num_freq x num_time) matrix containing the best-fit model of the dynamic spectrum

    output_name : str, optional
        the desired name of the PNG file containing the summary plots

    factor_freq : int, optional
        a integer number by which to downsample spectrum in frequency

    num_std : int, optional
        the number of standard deviations in residuals to use in weighting color maps

    residuals : np.ndarray, optional
        a (num_freq x num_time) matrix containing the best-fit residuals

    show : bool, optional
        plots the summary plot
    Returns
    -------
    None
        a PNG plot is created and recorded to disk, but the function returns nothing
    """

    # get dimensions and ensure consistency with inputs.
    num_freq, num_time = spectrum_orig.shape
    assert num_freq == len(freqs)

    # derive original resolutions.
    res_freq_orig = freqs[1] - freqs[0]
    res_freq = res_freq_orig / factor_freq
    res_time = times[1] - times[0]

    # downsample data to desired number of subbands.
    spectrum_downsampled = manip.downsample_2d(spectrum_orig, factor_freq, factor_time)
    model_downsampled = manip.downsample_2d(model, factor_freq, factor_time)
    residuals_downsampled = manip.downsample_2d(residuals, factor_freq, factor_time)
    freqs_downsampled = manip.downsample_1d(freqs, factor_freq)
    mask_freq_downsampled = manip.downsample_1d(mask_freq, factor_freq, boolean=True)
    spectrum_downsampled *= mask_freq_downsampled[:, None]
    model_downsampled *= mask_freq_downsampled[:, None]
    residuals_downsampled *= mask_freq_downsampled[:, None]
    
    idx_good_freq = np.where(mask_freq_downsampled)[0]
    idx_bad_freq = np.where(np.logical_not(mask_freq_downsampled))[0]

    # compute bounds of plotting region.
    min_time = 0.0
    max_time = num_time * res_time * 1e3 # last term converts to ms
    freq_initial = freqs_downsampled[idx_good_freq][0]
    freq_final = freqs_downsampled[idx_good_freq][-1]

    if freq_final > freq_initial:
        freq_initial -= res_freq_orig / 2
        freq_final += res_freq_orig / 2

    else:
        freq_initial += res_freq_orig / 2
        freq_final -= res_freq_orig / 2

    times_plot = np.linspace(min_time + res_time / 2., max_time - res_time / 2., num=num_time)

    # now set up figure and gridspec axes.
    fig = plt.figure(figsize=(15,12))#(3.25,3.25))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 3], 
         hspace=0.0, wspace=0.1
    )

    panel2d_data = plt.subplot(gs[3])
    panel2d_model = plt.subplot(gs[4], sharey=panel2d_data)
    panel2d_residual = plt.subplot(gs[5], sharey=panel2d_data)
    panel1d_data = plt.subplot(gs[0], sharex=panel2d_data)
    panel1d_model = plt.subplot(gs[1], sharex=panel2d_model)
    panel1d_residual = plt.subplot(gs[2], sharex=panel2d_residual)
    
    residuals = spectrum_downsampled[idx_good_freq] - model_downsampled[idx_good_freq]
    residual_median = np.nanmedian(residuals)
    residual_std = np.nanstd(residuals)
    vmin = residual_median - residual_std * num_std
    vmax = residual_median + residual_std * num_std
    vmin_residual = residual_median - residual_std * num_std
    vmax_residual = residual_median + residual_std * num_std
    print('Data', "vmin, vmax = {0:.2f}, {1:.2f}".format(vmin, vmax))
    # plot dynamic-spectrum data and band-averaged timeseries.
    panel2d_data.imshow(
        spectrum_downsampled[idx_good_freq], origin = 'lower', aspect="auto", interpolation="nearest", 
        extent=[min_time, max_time, freq_initial, freq_final], vmin=vmin, vmax=vmax
    )

    timeseries =  np.nanmean(spectrum_downsampled[idx_good_freq], axis=0) * np.sqrt(
        np.count_nonzero(~np.isnan(np.nansum(spectrum_downsampled[idx_good_freq], axis=-1)))
    )
    panel1d_data.plot(times_plot, timeseries)

    # extract y-axis limits from data panel for model and residual panels.
    y_min, y_max = panel1d_data.get_ylim()
    panel1d_model.set_ylim(y_min, y_max)
    panel1d_residual.set_ylim(y_min, y_max)
    print('Model', "vmin, vmax = {0:.2f}, {1:.2f}".format(vmin, vmax))
    # plot model panel.
    if model is not None:
        panel2d_model.imshow(
            model_downsampled[idx_good_freq], origin = 'lower', aspect="auto", interpolation="nearest",
            extent=[min_time, max_time, freq_initial, freq_final], vmin=vmin, vmax=vmax
        )

        panel1d_model.plot(
            times_plot, np.nanmean(model_downsampled[idx_good_freq], axis=0) * np.sqrt(
        np.count_nonzero(~np.isnan(np.nansum(model_downsampled[idx_good_freq], axis=-1)))
    )
        )

    # plot residual panel.
    if residuals is not None:
        panel2d_residual.imshow(
            residuals_downsampled[idx_good_freq], origin = 'lower', aspect="auto", cmap="bwr", interpolation="nearest",
            extent=[min_time, max_time, freq_initial, freq_final], vmin=vmin, vmax=vmax
        )

        panel1d_residual.plot(
            times_plot,  np.nanmean(residuals_downsampled[idx_good_freq], axis=0) * np.sqrt(
        np.count_nonzero(~np.isnan(np.nansum(residuals_downsampled[idx_good_freq], axis=-1)))
    )
        )

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
    
    plt.savefig(output_name, dpi=150, bbox_inches="tight")      
    
    if show:
        plt.show()
    
    plt.clf()
    plt.figure(figsize = (7,5))
    plt.plot(
            times_plot, timeseries, color = 'k', alpha = 0.3, lw=4
        )
    plt.plot(
            times_plot,  np.nanmean(model_downsampled[idx_good_freq], axis=0) * np.sqrt(
        np.count_nonzero(~np.isnan(np.nansum(model_downsampled[idx_good_freq], axis=-1)))
    ), c = 'r'
        )
    plt.xlabel('Time [ms]')
    #plt.savefig(output_name, dpi=150, bbox_inches="tight")      
    
    if show:
        plt.show()

    plt.clf()
