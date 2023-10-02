#! /usr/bin/env python
from fitburst.backend.generic import DataReader
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

if __name__=='__main__':

    parser=argparse.ArgumentParser(description="====A python script to read .npz files and find the peak values for the given data====\n \
                                                Include the name of file with full path if you are out of the file directory."
                                                )
    parser.add_argument('file_name', 
                        type=str,
                        metavar='',
                        help='A Numpy state file containing data and metadata.')

    parser.add_argument('-r','--shift_rms',type=float,metavar='',
                        help='By what percent would you like to increase RMS value?')

    args=parser.parse_args()

    input_file=args.file_name  #Store the input file in input_file variable
    rms_shift=args.shift_rms   #Store the shift rms percentage in rms_shift variable.
    #Read the input file.
    input_data=DataReader(input_file)

    # Load the data into the memory for processing.
    input_data.load_data()
    input_data.downsample(64,1)
    freq=input_data.freqs
    time=input_data.times
    data_full=input_data.data_full

class FindPeak:
    """
    A Python Class Find Peaks in the Loaded Data.
    """

    #This is for creating the name of output files same as the input files.
    #name_of_file=str(input_file).replace('.npz','')
    
    def __init__(self, data: float, time: float, freq : float, rms: float = None):
        """
        Initializes FindPeak class with data(data is input file provided),time,freq as a parameters.

        Parameters:
        ----------
        data : numpy.ndarray
            a matrix of spectrum data, with dimenions that match those of the times 
            and freqs arrays

        time : numpy.ndarray
            an array of values corresponding to observing times

        freq : numpy.ndarray
            an array of observing frequencies at which to evaluate spectrum

        rms : a floating value ranging from 0 to infinity
            this is the percentage by which we want to shift(increase) the original rms value 

        """     
        self.data = data
        self.freq = freq
        self.time = time
        self.rms = rms
        
    def find_peak(self, distance: int = 5):
        """
        Finds peak positions in a spectrum and the approximate temporal width of each peak. Prints out
        the peak positions and corresponding temporal width in a data frame.

        Returns: None

        """
        self.mean_intensity_freq=self.data.mean(axis=0)
        plt.plot(self.time*1000,self.mean_intensity_freq)
        peaks_location=find_peaks(self.mean_intensity_freq, distance=distance)
        
        self.peak_times=self.time[peaks_location[0]]
        self.peak_mean_intensities=self.mean_intensity_freq[peaks_location[0]]
        
        #Find the rms value of intensities
        self.rms_intensity=np.sqrt(np.mean((self.mean_intensity_freq)**2))
        print(f"The rms intensity is {self.rms_intensity} \n")

        #Shift RMS line by given percentage if asked in the input.
        if self.rms is not None:
            #Increase the value of rms intensity by given rms shift percent.
            #shifted_rms_intensity=self.rms_intensity+(self.rms/100)*self.rms_intensity
            shifted_rms_intensity = self.rms * np.max(self.mean_intensity_freq)

            #Find the peaks greater than shifted_rms values.
            index_peaks_greater_rms=np.where(self.peak_mean_intensities>shifted_rms_intensity)
            self.peaks_greater_rms=self.peak_mean_intensities[index_peaks_greater_rms]
            self.times_peaks_greater_rms=self.peak_times[index_peaks_greater_rms]
        else:  
            #Find the peaks greater than rms values.
            index_peaks_greater_rms=np.where(self.peak_mean_intensities>self.rms_intensity)
            self.peaks_greater_rms=self.peak_mean_intensities[index_peaks_greater_rms]
            self.times_peaks_greater_rms=self.peak_times[index_peaks_greater_rms]
        
        #Now find the width of each burst
        for each_peak in index_peaks_greater_rms[0]:

            # These are the indices of times just below and just above the peaks.
            below_peak_index, above_peak_index = each_peak - 1, each_peak + 1

            if below_peak_index < 0:
                below_peak_index = 0

            if above_peak_index >= len(self.peak_times):
                above_peak_index = each_peak
            
            # Now find the times for above and below peak
            below_peak_time=self.peak_times[below_peak_index]*1000
            above_peak_time=self.peak_times[above_peak_index]*1000

            # Print the Burst Width for each burst
            self.burst_widths=above_peak_time-below_peak_time
            self.time_of_arrivals=self.times_peaks_greater_rms*1000


        dictionary_of_outputs={"Times of Arrivals":self.time_of_arrivals,"Burst_Widths":self.burst_widths}
        self.df=pd.DataFrame(data=dictionary_of_outputs)
        print(self.df)      #This displays output in dataframe format.

    def create_csv_files_and_plot(self):
        """
        This method creates a csv file and a plot in .png format for the peaks and corresponding 
        temporal widths. If you call this method then you don't have to call find_peak method 
        because find_peak method is already called here.

        Returns: None
        ------
        """
        self.find_peak()
        self.df.to_csv(f"{self.name_of_file}_peaks",index=False,)

        #Now create plots.
        fig = plt.figure(figsize=(15,12))
        gs = gridspec.GridSpec(2, 1,hspace=0.0, wspace=0.1)
        panel_2d=plt.subplot(gs[1])
        panel_1d=plt.subplot(gs[0])
        panel_2d.imshow(self.data,origin='lower',aspect='auto')
        panel_1d.plot(self.time*1000,self.mean_intensity_freq)
        panel_1d.plot(self.time*1000,self.rms_intensity*np.ones(len(self.time)),linestyle='dashed',
        label='RMS')

        if self.rms is not None:
            panel_1d.plot(self.time*1000,(self.rms_intensity+self.rms/100*self.rms_intensity)*np.ones(len(self.time)),
            linestyle='dashed',label='Shifted RMS')

        #Add or Remove labels
        plt.setp(panel_1d.get_xticklabels(), visible=False)
        panel_2d.set_xlabel('Time(ms)',fontsize=20)
        panel_2d.set_ylabel('Frequency(MHz)',fontsize=20)
        panel_1d.set_ylabel('Mean Intensity',fontsize=20)
        panel_1d.legend()

        panel_1d.set_xlim(self.time.min()*1000,self.time.max()*1000)
        panel_1d.scatter(self.peak_times*1000,self.peak_mean_intensities,c='red')
        panel_1d.scatter(self.times_peaks_greater_rms*1000,self.peaks_greater_rms,c='green',s=45)
        plt.savefig(f"{self.name_of_file}_peaks.png")
    
    def get_parameters_dict(self, original_dict: dict, update_width=False):
        """
        This method returns peak values and burst width in dictionary which is compatible with fitburst.

        Returns: Dictionary with values in List
        --------
        """

        #self.find_peak()
        mul_factor=len(self.time_of_arrivals)
        burst_parameters = {}

        for current_key in original_dict.keys():
            if current_key == "arrival_time":
                burst_parameters[current_key] = (self.time_of_arrivals / 1000.).tolist()

            elif current_key == "burst_width" and update_width:
                burst_parameters[current_key] = (self.burst_widths / 1000.).tolist()

            else:
                burst_parameters[current_key] = [original_dict[current_key][0]] * mul_factor

        #burst_parameters={
        #    "amplitude"             :   original_dict*mul_factor,
        #    "arrival_time"          :   [self.time_of_arrivals],
        #    "burst_width"           :   [self.burst_widths],
        #    "dm"                    :   [557.0]*mul_factor,
        #    "dm_index"              :   [-2.0]*mul_factor,
        #    "ref_freq"              :   [600.0]*mul_factor,
        #    "scattering_index"      :   [-4.0]*mul_factor,
        #    "scattering_timescale"  :   [0.0]*mul_factor,
        #    "freq_mean"             :   [450.0]*mul_factor,
        #    "freq_width"            :   [43.0]*mul_factor,
        #}

        return burst_parameters
        

if __name__=='__main__':
    display_peaks=FindPeak(data_full,time,freq,rms_shift)
    display_peaks.create_csv_files_and_plot()


