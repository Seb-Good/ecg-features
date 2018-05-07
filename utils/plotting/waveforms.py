"""
waveforms.py
--------------------
This module provides functions for plotting ECG waveforms.
--------------------
By: Sebastian D. Goodfellow, Ph.D., 2017
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import numpy as np
import scipy.io as sio
import matplotlib.pylab as plt
from ipywidgets import interact, fixed


def plot_waveform(index, labels, waveform_path, fs):

    # Get file name
    file_name = labels.loc[index, 'file_name']

    # Get label
    label = labels.loc[index, 'label']

    # Get waveform
    time, waveform = load_waveform(path=os.path.join(waveform_path, file_name + '.mat'), fs=fs)

    # Setup plot
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(hspace=0.25)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.set_title('File Name: ' + file_name + '\nLabel: ' + label, fontsize=20)

    # Plot waveform
    ax1.plot(time, waveform, '-k')

    # Configure axes
    ax1.set_xlabel('Time, seconds', fontsize=25)
    ax1.set_ylabel('Amplitude, mV', fontsize=25)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=18)


def plot_waveforms(labels, waveform_path, fs):

    # Launch widget
    interact(
        plot_waveform,
        index=(0, labels.shape[0] - 1, 1),
        labels=fixed(labels),
        waveform_path=fixed(waveform_path),
        fs=fixed(fs)
    )


def load_waveform(path, fs):

    # Load .mat file
    waveform = sio.loadmat(path)['val'][0] / 1000.

    return np.arange(len(waveform)) * 1 / fs, waveform
