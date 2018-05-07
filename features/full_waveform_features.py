"""
feature_extractor.py
--------------------
This module provides a class and methods for extracting full waveform features from ECG signals.
--------------------
By: Sebastian D. Goodfellow, Ph.D., 2017
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import pywt
import numpy as np
import scipy as sp
from scipy import signal

# Local imports
from utils.tools.higuchi_fractal_dimension import hfd


class FullWaveformFeatures:

    """
    Generate a dictionary of full waveform statistics for one ECG signal.

    Parameters
    ----------
    ts : numpy array
        Full waveform time array.
    signal_raw : numpy array
        Raw full waveform.
    signal_filtered : numpy array
        Filtered full waveform.
    rpeaks : numpy array
        Array indices of R-Peaks
    templates_ts : numpy array
        Template waveform time array
    templates : numpy array
        Template waveforms
    fs : int, float
        Sampling frequency (Hz).

    Returns
    -------
    full_waveform_features : dictionary
        Full waveform features.
    """

    def __init__(self, ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates, fs):

        # Set parameters
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs

        # Feature dictionary
        self.full_waveform_features = dict()

    def get_full_waveform_features(self):
        return self.full_waveform_features

    def extract_full_waveform_features(self):
        self.full_waveform_features.update(self.calculate_basic_features())
        self.full_waveform_features.update(self.calculate_stationary_wavelet_transform_features())

    def calculate_basic_features(self):

        # Empty dictionary
        basic_features = dict()

        # Calculate statistics
        basic_features['full_waveform_min'] = np.min(self.signal_filtered)
        basic_features['full_waveform_max'] = np.max(self.signal_filtered)
        basic_features['full_waveform_mean'] = np.mean(self.signal_filtered)
        basic_features['full_waveform_median'] = np.median(self.signal_filtered)
        basic_features['full_waveform_std'] = np.std(self.signal_filtered)
        basic_features['full_waveform_skew'] = sp.stats.skew(self.signal_filtered)
        basic_features['full_waveform_kurtosis'] = sp.stats.kurtosis(self.signal_filtered)
        basic_features['full_waveform_duration'] = np.max(self.ts)

        return basic_features

    def calculate_stationary_wavelet_transform_features(self):

        # Empty dictionary
        stationary_wavelet_transform_features = dict()

        # Decomposition level
        decomp_level = 4

        # Stationary wavelet transform
        swt = self.stationary_wavelet_transform(self.signal_filtered, wavelet='db4', level=decomp_level)

        # Set frequency band
        freq_band_low = (3, 10)
        freq_band_med = (10, 30)
        freq_band_high = (30, 45)

        """Frequency Domain"""
        for level in range(len(swt)):

            """Detail"""
            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=swt[level]['d'], fs=self.fs)

            # Get frequency band
            freq_band_low_index = np.logical_and(fxx >= freq_band_low[0], fxx < freq_band_low[1])
            freq_band_med_index = np.logical_and(fxx >= freq_band_med[0], fxx < freq_band_med[1])
            freq_band_high_index = np.logical_and(fxx >= freq_band_high[0], fxx < freq_band_high[1])

            # Calculate maximum power
            max_power_low = np.max(pxx[freq_band_low_index])
            max_power_med = np.max(pxx[freq_band_med_index])
            max_power_high = np.max(pxx[freq_band_high_index])

            # Calculate average power
            mean_power_low = np.trapz(y=pxx[freq_band_low_index], x=fxx[freq_band_low_index])
            mean_power_med = np.trapz(y=pxx[freq_band_med_index], x=fxx[freq_band_med_index])
            mean_power_high = np.trapz(y=pxx[freq_band_high_index], x=fxx[freq_band_high_index])

            # Calculate max/mean power ratio
            stationary_wavelet_transform_features['swt_d_' + str(level+1) + '_low_power_ratio'] = \
                max_power_low / mean_power_low
            stationary_wavelet_transform_features['swt_d_' + str(level+1) + '_med_power_ratio'] = \
                max_power_med / mean_power_med
            stationary_wavelet_transform_features['swt_d_' + str(level+1) + '_high_power_ratio'] = \
                max_power_high / mean_power_high

            """Approximation"""
            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=swt[level]['a'], fs=self.fs)

            # Get frequency band
            freq_band_low_index = np.logical_and(fxx >= freq_band_low[0], fxx < freq_band_low[1])
            freq_band_med_index = np.logical_and(fxx >= freq_band_med[0], fxx < freq_band_med[1])
            freq_band_high_index = np.logical_and(fxx >= freq_band_high[0], fxx < freq_band_high[1])

            # Calculate maximum power
            max_power_low = np.max(pxx[freq_band_low_index])
            max_power_med = np.max(pxx[freq_band_med_index])
            max_power_high = np.max(pxx[freq_band_high_index])

            # Calculate average power
            mean_power_low = np.trapz(y=pxx[freq_band_low_index], x=fxx[freq_band_low_index])
            mean_power_med = np.trapz(y=pxx[freq_band_med_index], x=fxx[freq_band_med_index])
            mean_power_high = np.trapz(y=pxx[freq_band_high_index], x=fxx[freq_band_high_index])

            # Calculate max/mean power ratio
            stationary_wavelet_transform_features['swt_a_' + str(level+1) + '_low_power_ratio'] = \
                max_power_low / mean_power_low
            stationary_wavelet_transform_features['swt_a_' + str(level+1) + '_med_power_ratio'] = \
                max_power_med / mean_power_med
            stationary_wavelet_transform_features['swt_a_' + str(level+1) + '_high_power_ratio'] = \
                max_power_high / mean_power_high

        """Non-Linear"""
        for level in range(len(swt)):

            """Detail"""
            # Log-energy entropy
            stationary_wavelet_transform_features['swt_d_' + str(level+1) + '_energy_entropy'] = \
                np.sum(np.log10(np.power(swt[level]['d'], 2)))

            # Higuchi_fractal
            stationary_wavelet_transform_features['swt_d_' + str(level+1) + '_higuchi_fractal'] = \
                hfd(swt[level]['d'], k_max=10)

            """Approximation"""
            # Log-energy entropy
            stationary_wavelet_transform_features['swt_a_' + str(level+1) + '_energy_entropy'] = \
                np.sum(np.log10(np.power(swt[level]['a'], 2)))

            # Higuchi_fractal
            stationary_wavelet_transform_features['swt_a_' + str(level+1) + '_higuchi_fractal'] = \
                hfd(swt[level]['a'], k_max=10)

        return stationary_wavelet_transform_features

    @staticmethod
    def calculate_decomposition_level(waveform_length, level):

        # Set starting multiplication factor
        factor = 0

        # Set updated waveform length variable
        waveform_length_updated = None

        # If waveform is not the correct length for proposed decomposition level
        if waveform_length % 2**level != 0:

            # Calculate remainder
            remainder = waveform_length % 2**level

            # Loop through multiplication factors until minimum factor found
            while remainder != 0:

                # Update multiplication factor
                factor += 1

                # Update waveform length
                waveform_length_updated = factor * waveform_length

                # Calculate updated remainder
                remainder = waveform_length_updated % 2**level

            return waveform_length_updated

        # If waveform is the correct length for proposed decomposition level
        else:
            return waveform_length

    @staticmethod
    def add_padding(waveform, waveform_length_updated):

        # Calculate required padding
        pad_count = np.abs(len(waveform) - waveform_length_updated)

        # Calculate before waveform padding
        pad_before = int(np.floor(pad_count / 2.0))

        # Calculate after waveform padding
        pad_after = pad_count - pad_before

        # Add padding to waveform
        waveform_padded = np.append(np.zeros(pad_before), np.append(waveform, np.zeros(pad_after)))

        return waveform_padded, pad_before, pad_after

    def stationary_wavelet_transform(self, waveform, wavelet, level):

        # Calculate waveform length
        waveform_length = len(waveform)

        # Calculate minimum waveform length for SWT of certain decomposition level
        waveform_length_updated = self.calculate_decomposition_level(waveform_length, level)

        # Add necessary padding to waveform
        waveform_padded, pad_before, pad_after = self.add_padding(waveform, waveform_length_updated)

        # Compute stationary wavelet transform
        swt = pywt.swtn(waveform_padded, wavelet=wavelet, level=level, start_level=0)

        # Loop through decomposition levels and remove padding
        for lev in range(len(swt)):

            # Approximation
            swt[lev]['a'] = swt[lev]['a'][pad_before:len(waveform_padded) - pad_after]

            # Detail
            swt[lev]['d'] = swt[lev]['d'][pad_before:len(waveform_padded) - pad_after]

        return swt
