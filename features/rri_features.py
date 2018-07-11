"""
rri_features.py
--------------------
This module provides a class and methods for extracting RRI features from ECG signals.
--------------------
By: Sebastian D. Goodfellow, Ph.D., 2017
"""


# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import scipy as sp
from scipy import signal
from scipy import interpolate
from pyentrp import entropy as ent

# Local imports
from utils.tools.higuchi_fractal_dimension import hfd


class RRIFeatures:

    """
    Generate a dictionary of RRI features for one ECG signal.

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
    template_before : float, seconds
            Time before R-Peak to start template.
    template_after : float, seconds
        Time after R-Peak to end template.

    Returns
    -------
    rri_features : dictionary
        RRI features.
    """

    def __init__(self, ts, signal_raw, signal_filtered, rpeaks,templates_ts,
                 templates, fs, template_before, template_after):

        # Set parameters
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs
        self.template_before_ts = template_before
        self.template_after_ts = template_after

        # Set attributes
        self.template_before_sp = int(self.template_before_ts * self.fs)
        self.template_after_sp = int(self.template_after_ts * self.fs)
        self.rri = None
        self.rri_ts = None
        self.diff_rri = None
        self.diff_rri_ts = None
        self.diff2_rri = None
        self.diff2_rri_ts = None
        self.templates_good = None
        self.templates_bad = None
        self.median_template = None
        self.median_template_good = None
        self.median_template_bad = None
        self.rpeaks_good = None
        self.rpeaks_bad = None
        self.templates_secondary = None
        self.median_template_secondary = None
        self.rpeaks_secondary = None

        # Calculate median template
        self.median_template = np.median(self.templates, axis=1)

        # R-Peak calculations
        self.template_rpeak_sp = self.template_before_sp

        # Correct R-Peak picks
        self.r_peak_check(correlation_threshold=0.9)

        # RR interval calculations
        self.rpeaks_ts = self.ts[self.rpeaks]
        self.calculate_rr_intervals(correlation_threshold=0.9)

        # Get secondary templates
        # self.get_secondary_templates(correlation_threshold=0.9)

        # Feature dictionary
        self.rri_features = dict()

    """
    Compile Features
    """
    def get_rri_features(self):
        return self.rri_features

    def calculate_heart_rate_variability_statistics(self):
        """Group features"""
        self.rri_features.update(
            self.calculate_rri_temporal_features(self.rri, self.diff_rri, self.diff2_rri)
        )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_rri_nonlinear_statistics(self.rri, self.diff_rri, self.diff2_rri)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_pearson_correlation_statistics(self.rri)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_spearmanr_correlation_statistics(self.rri)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_kendalltau_correlation_statistics(self.rri)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_pointbiserialr_correlation_statistics(self.rri)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_poincare_statistics(self.rri)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_rri_spectral_statistics(self.rri, self.rri_ts)
        # )
        # self.heart_rate_variability_statistics.update(
        #     self.calculate_rri_fragmentation_statistics(self.diff_rri, self.diff_rri_ts)
        # )
        # self.heart_rate_variability_statistics.update(self.calculate_rpeak_detection_statistics())
        # self.heart_rate_variability_statistics.update(self.calculate_rri_cluster_statistics())

    """
    Pre Processing
    """
    @staticmethod
    def normalize_series(series, method='median'):

        if method == 'median':
            return series / np.median(series)
        if method == 'mean':
            return series / np.mean(series)

    @staticmethod
    def is_outlier(points, thresh=8.0):

        if len(points.shape) == 1:
            points = points[:, None]

        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def rri_physiological_filter(self):

        # Define physiologically possible interbeat interval range
        rri_max = 3.0   # 20 bpm
        rri_min = 0.25  # 240 bpm

        # get indices of physiologically impossible values
        possible = np.nonzero((self.rri <= rri_max) & (self.rri >= rri_min))[0]

    def get_secondary_templates(self, correlation_threshold=0.9):

        # Set rpeaks
        rpeaks = self.rpeaks_bad.astype(float)

        # If bad templates exist
        if self.templates_bad is not None:

            # Calculate median template
            self.median_template_secondary = np.median(self.templates_bad, axis=1)

            # Set counter
            count = 0

            # Loop through bad templates
            for template_id in range(self.templates_bad.shape[1]):

                # Calculate correlation coefficient
                correlation_coefficient = np.corrcoef(
                    self.median_template_secondary[self.template_rpeak_sp - 50:self.template_rpeak_sp + 50],
                    self.templates_bad[self.template_rpeak_sp - 50:self.template_rpeak_sp + 50, template_id]
                )

                # Check correlation
                if correlation_coefficient[0, 1] < correlation_threshold:

                    # Remove rpeak
                    rpeaks[template_id] = np.nan

                else:

                    # Update counter
                    count += 1

            if count >= 2:

                # Get good and bad rpeaks
                self.rpeaks_secondary = self.rpeaks_bad[np.isfinite(rpeaks)]

                # Get good and bad
                self.templates_secondary = self.templates_bad[:, np.where(np.isfinite(rpeaks))[0]]

                # Get median templates
                self.median_template_secondary = np.median(self.templates_secondary, axis=1)

    def r_peak_check(self, correlation_threshold=0.9):

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

        # Loop through rpeaks
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Compute cross correlation
                cross_correlation = signal.correlate(
                    self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                    self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
                )

                # Correct rpeak
                rpeak_corrected = \
                    self.rpeaks[template_id] - \
                    (np.argmax(cross_correlation) -
                     len(self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]))

                # Check to see if shifting the R-Peak improved the correlation coefficient
                if self.check_improvement(rpeak_corrected, correlation_threshold):

                    # Update rpeaks array
                    self.rpeaks[template_id] = rpeak_corrected

        # Re-extract templates
        self.templates, self.rpeaks = self.extract_templates(self.rpeaks)

        # Re-compute median template
        self.median_template = np.median(self.templates, axis=1)

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

    def extract_templates(self, rpeaks):

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(self.signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - self.template_before_sp
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + self.template_after_sp
            if b > length:
                break

            # Append template list
            templates.append(self.signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def check_improvement(self, rpeak_corrected, correlation_threshold):

        # Before R-Peak
        a = rpeak_corrected - self.template_before_sp

        # After R-Peak
        b = rpeak_corrected + self.template_after_sp

        if a >= 0 and b < len(self.signal_filtered):

            # Update template
            template_corrected = self.signal_filtered[a:b]

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                template_corrected[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]
            )

            # Check new correlation
            if correlation_coefficient[0, 1] >= correlation_threshold:
                return True
            else:
                return False
        else:
            return False

    def calculate_rr_intervals(self, correlation_threshold=0.9):

        # Get rpeaks is floats
        rpeaks = self.rpeaks.astype(float)

        # Loop through templates
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Remove rpeak
                rpeaks[template_id] = np.nan

        # RRI
        rri = np.diff(rpeaks) * 1 / self.fs
        rri_ts = rpeaks[0:-1] / self.fs + rri / 2

        # RRI Velocity
        diff_rri = np.diff(rri)
        diff_rri_ts = rri_ts[0:-1] + diff_rri / 2

        # RRI Acceleration
        diff2_rri = np.diff(diff_rri)
        diff2_rri_ts = diff_rri_ts[0:-1] + diff2_rri / 2

        # Drop rri, diff_rri, diff2_rri outliers
        self.rri = rri[np.isfinite(rri)]
        self.rri_ts = rri_ts[np.isfinite(rri_ts)]
        self.diff_rri = diff_rri[np.isfinite(diff_rri)]
        self.diff_rri_ts = diff_rri_ts[np.isfinite(diff_rri_ts)]
        self.diff2_rri = diff2_rri[np.isfinite(diff2_rri)]
        self.diff2_rri_ts = diff2_rri_ts[np.isfinite(diff2_rri_ts)]

        # Get good and bad rpeaks
        self.rpeaks_good = self.rpeaks[np.isfinite(rpeaks)]
        self.rpeaks_bad = self.rpeaks[~np.isfinite(rpeaks)]

        # Get good and bad
        self.templates_good = self.templates[:, np.where(np.isfinite(rpeaks))[0]]
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.templates_bad = self.templates[:, np.where(~np.isfinite(rpeaks))[0]]

        # Get median templates
        self.median_template_good = np.median(self.templates_good, axis=1)
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.median_template_bad = np.median(self.templates_bad, axis=1)

    @staticmethod
    def safe_check(value):
        """Check for finite value and replace with np.nan if does not exist."""
        try:
            if np.isfinite(value):
                return value
            else:
                return np.nan()
        except ValueError:
            return np.nan

    def calculate_rri_temporal_features(self, rri, diff_rri, diff2_rri):

        # Empty dictionary
        rri_temporal_features = dict()

        # RR interval statistics
        if len(rri) > 0:
            rri_temporal_features['rri_min'] = np.min(rri)
            rri_temporal_features['rri_max'] = np.max(rri)
            rri_temporal_features['rri_mean'] = np.mean(rri)
            rri_temporal_features['rri_median'] = np.median(rri)
            rri_temporal_features['rri_std'] = np.std(rri, ddof=1)
            rri_temporal_features['rri_skew'] = sp.stats.skew(rri)
            rri_temporal_features['rri_kurtosis'] = sp.stats.kurtosis(rri)
            rri_temporal_features['rri_rms'] = np.sqrt(np.mean(np.power(rri, 2)))
        else:
            rri_temporal_features['rri_min'] = np.nan
            rri_temporal_features['rri_max'] = np.nan
            rri_temporal_features['rri_mean'] = np.nan
            rri_temporal_features['rri_median'] = np.nan
            rri_temporal_features['rri_std'] = np.nan
            rri_temporal_features['rri_skew'] = np.nan
            rri_temporal_features['rri_kurtosis'] = np.nan
            rri_temporal_features['rri_rms'] = np.nan

        # Differences between successive RR interval differences statistics
        if len(diff_rri) > 0:
            rri_temporal_statistics['diff_rri_min' + suffix] = np.min(diff_rri)
            rri_temporal_statistics['diff_rri_max' + suffix] = np.max(diff_rri)
            rri_temporal_statistics['diff_rri_mean' + suffix] = np.mean(diff_rri)
            rri_temporal_statistics['diff_rri_median' + suffix] = np.median(diff_rri)
            rri_temporal_statistics['diff_rri_std' + suffix] = np.std(diff_rri, ddof=1)
            rri_temporal_statistics['diff_rri_skew' + suffix] = sp.stats.skew(diff_rri)
            rri_temporal_statistics['diff_rri_kurtosis' + suffix] = sp.stats.kurtosis(diff_rri)
            rri_temporal_statistics['diff_rri_rms' + suffix] = np.sqrt(np.mean(np.power(diff_rri, 2)))
        else:
            rri_temporal_statistics['diff_rri_min' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_max' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_mean' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_median' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_std' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_skew' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_kurtosis' + suffix] = np.nan
            rri_temporal_statistics['diff_rri_rms' + suffix] = np.nan

        # Differences between successive RR intervals statistics
        if len(diff2_rri) > 0:
            rri_temporal_statistics['diff2_rri_min' + suffix] = np.min(diff2_rri)
            rri_temporal_statistics['diff2_rri_max' + suffix] = np.max(diff2_rri)
            rri_temporal_statistics['diff2_rri_mean' + suffix] = np.mean(diff2_rri)
            rri_temporal_statistics['diff2_rri_median' + suffix] = np.median(diff2_rri)
            rri_temporal_statistics['diff2_rri_std' + suffix] = np.std(diff2_rri, ddof=1)
            rri_temporal_statistics['diff2_rri_kurtosis' + suffix] = sp.stats.kurtosis(diff2_rri)
            rri_temporal_statistics['diff2_rri_rms' + suffix] = np.sqrt(np.mean(np.power(diff2_rri, 2)))
        else:
            rri_temporal_statistics['diff2_rri_min' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_max' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_mean' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_median' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_std' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_kurtosis' + suffix] = np.nan
            rri_temporal_statistics['diff2_rri_rms' + suffix] = np.nan

        # pNN statistics
        if len(diff_rri) > 0:
            rri_temporal_statistics['pnn01' + suffix] = self.pnn(diff_rri, 0.001)
            rri_temporal_statistics['pnn10' + suffix] = self.pnn(diff_rri, 0.01)
            rri_temporal_statistics['pnn20' + suffix] = self.pnn(diff_rri, 0.02)
            rri_temporal_statistics['pnn30' + suffix] = self.pnn(diff_rri, 0.03)
            rri_temporal_statistics['pnn40' + suffix] = self.pnn(diff_rri, 0.04)
            rri_temporal_statistics['pnn50' + suffix] = self.pnn(diff_rri, 0.05)
            rri_temporal_statistics['pnn60' + suffix] = self.pnn(diff_rri, 0.06)
            rri_temporal_statistics['pnn70' + suffix] = self.pnn(diff_rri, 0.07)
            rri_temporal_statistics['pnn80' + suffix] = self.pnn(diff_rri, 0.08)
            rri_temporal_statistics['pnn90' + suffix] = self.pnn(diff_rri, 0.09)
            rri_temporal_statistics['pnn100' + suffix] = self.pnn(diff_rri, 0.1)
            rri_temporal_statistics['pnn200' + suffix] = self.pnn(diff_rri, 0.2)
            rri_temporal_statistics['pnn400' + suffix] = self.pnn(diff_rri, 0.4)
            rri_temporal_statistics['pnn600' + suffix] = self.pnn(diff_rri, 0.6)
            rri_temporal_statistics['pnn800' + suffix] = self.pnn(diff_rri, 0.8)

        else:
            rri_temporal_statistics['pnn01' + suffix] = np.nan
            rri_temporal_statistics['pnn10' + suffix] = np.nan
            rri_temporal_statistics['pnn20' + suffix] = np.nan
            rri_temporal_statistics['pnn30' + suffix] = np.nan
            rri_temporal_statistics['pnn40' + suffix] = np.nan
            rri_temporal_statistics['pnn50' + suffix] = np.nan
            rri_temporal_statistics['pnn60' + suffix] = np.nan
            rri_temporal_statistics['pnn70' + suffix] = np.nan
            rri_temporal_statistics['pnn80' + suffix] = np.nan
            rri_temporal_statistics['pnn90' + suffix] = np.nan
            rri_temporal_statistics['pnn100' + suffix] = np.nan
            rri_temporal_statistics['pnn200' + suffix] = np.nan
            rri_temporal_statistics['pnn400' + suffix] = np.nan
            rri_temporal_statistics['pnn600' + suffix] = np.nan
            rri_temporal_statistics['pnn800' + suffix] = np.nan

        return rri_temporal_statistics

    @staticmethod
    def consecutive_count(random_list):

        retlist = []
        count = 1
        # Avoid IndexError for  random_list[i+1]
        for i in range(len(random_list) - 1):
            # Check if the next number is consecutive
            if random_list[i] + 1 == random_list[i+1]:
                count += 1
            else:
                # If it is not append the count and restart counting
                retlist = np.append(retlist, count)
                count = 1
        # Since we stopped the loop one early append the last count
        retlist = np.append(retlist, count)

        return retlist

    def calculate_rri_nonlinear_statistics(self, rri, diff_rri, diff2_rri, suffix=''):

        # Empty dictionary
        rri_nonlinear_statistics = dict()

        # Non-linear RR statistics
        if len(rri) > 1:
            rri_nonlinear_statistics['rri_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(rri, M=2, R=0.1*np.std(rri)))
            rri_nonlinear_statistics['rri_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['rri_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['rri_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(rri, m=2, delay=1))
            rri_nonlinear_statistics['rri_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(rri, m=2, delay=1, scale=1)[0])
            rri_nonlinear_statistics['rri_fisher_info' + suffix] = fisher_info(rri, tau=1, de=2)
            hjorth_parameters = hjorth(rri)
            rri_nonlinear_statistics['rri_activity' + suffix] = hjorth_parameters[0]
            rri_nonlinear_statistics['rri_complexity' + suffix] = hjorth_parameters[1]
            rri_nonlinear_statistics['rri_morbidity' + suffix] = hjorth_parameters[2]
            rri_nonlinear_statistics['rri_hurst_exponent' + suffix] = pfd(rri)
            rri_nonlinear_statistics['rri_svd_entropy' + suffix] = svd_entropy(rri, tau=2, de=2)
            rri_nonlinear_statistics['rri_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(rri)
        else:
            rri_nonlinear_statistics['rri_approximate_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_sample_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_multiscale_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_multiscale_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_fisher_info' + suffix] = np.nan
            rri_nonlinear_statistics['rri_activity' + suffix] = np.nan
            rri_nonlinear_statistics['rri_complexity' + suffix] = np.nan
            rri_nonlinear_statistics['rri_morbidity' + suffix] = np.nan
            rri_nonlinear_statistics['rri_hurst_exponent' + suffix] = np.nan
            rri_nonlinear_statistics['rri_svd_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['rri_petrosian_fractal_dimension' + suffix] = np.nan

        # Non-linear RR difference statistics
        if len(diff_rri) > 1:
            rri_nonlinear_statistics['diff_rri_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(diff_rri, M=2, R=0.1*np.std(rri)))
            rri_nonlinear_statistics['diff_rri_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(diff_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff_rri_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(diff_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff_rri_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(diff_rri, m=2, delay=1))
            rri_nonlinear_statistics['diff_rri_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(diff_rri, m=2, delay=1, scale=1)[0])
            rri_nonlinear_statistics['diff_rri_fisher_info' + suffix] = fisher_info(diff_rri, tau=1, de=2)
            hjorth_parameters = hjorth(diff_rri)
            rri_nonlinear_statistics['diff_rri_activity' + suffix] = hjorth_parameters[0]
            rri_nonlinear_statistics['diff_rri_complexity' + suffix] = hjorth_parameters[1]
            rri_nonlinear_statistics['diff_rri_morbidity' + suffix] = hjorth_parameters[2]
            rri_nonlinear_statistics['diff_rri_hurst_exponent' + suffix] = pfd(diff_rri)
            rri_nonlinear_statistics['diff_rri_svd_entropy' + suffix] = svd_entropy(diff_rri, tau=2, de=2)
            rri_nonlinear_statistics['diff_rri_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(diff_rri)
        else:
            rri_nonlinear_statistics['diff_rri_approximate_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_sample_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_multiscale_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_multiscale_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_fisher_info' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_activity' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_complexity' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_morbidity' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_hurst_exponent' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_svd_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff_rri_petrosian_fractal_dimension' + suffix] = np.nan

        # Non-linear RR difference difference statistics
        if len(diff2_rri) > 1:
            rri_nonlinear_statistics['diff2_rri_shannon_entropy' + suffix] = \
                self.safe_check(ent.shannon_entropy(diff2_rri))
            rri_nonlinear_statistics['diff2_rri_approximate_entropy' + suffix] = \
                self.safe_check(pyeeg.ap_entropy(diff2_rri, M=2, R=0.1*np.std(rri)))
            rri_nonlinear_statistics['diff2_rri_sample_entropy' + suffix] = \
                self.safe_check(ent.sample_entropy(diff2_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff2_rri_multiscale_entropy' + suffix] = \
                self.safe_check(ent.multiscale_entropy(diff2_rri, sample_length=2, tolerance=0.1*np.std(rri))[0])
            rri_nonlinear_statistics['diff2_rri_permutation_entropy' + suffix] = \
                self.safe_check(ent.permutation_entropy(diff2_rri, m=2, delay=1))
            rri_nonlinear_statistics['diff2_rri_multiscale_permutation_entropy' + suffix] = \
                self.safe_check(ent.multiscale_permutation_entropy(diff2_rri, m=2, delay=1, scale=1)[0])
            rri_nonlinear_statistics['diff2_rri_fisher_info' + suffix] = fisher_info(diff2_rri, tau=1, de=2)
            hjorth_parameters = hjorth(diff2_rri)
            rri_nonlinear_statistics['diff2_rri_activity' + suffix] = hjorth_parameters[0]
            rri_nonlinear_statistics['diff2_rri_complexity' + suffix] = hjorth_parameters[1]
            rri_nonlinear_statistics['diff2_rri_morbidity' + suffix] = hjorth_parameters[2]
            rri_nonlinear_statistics['diff2_rri_hurst_exponent' + suffix] = pfd(diff2_rri)
            rri_nonlinear_statistics['diff2_rri_svd_entropy' + suffix] = svd_entropy(diff2_rri, tau=2, de=2)
            rri_nonlinear_statistics['diff2_rri_petrosian_fractal_dimension' + suffix] = pyeeg.pfd(diff2_rri)
        else:
            rri_nonlinear_statistics['diff2_rri_shannon_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_approximate_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_sample_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_multiscale_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_multiscale_permutation_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_fisher_info' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_activity' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_complexity' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_morbidity' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_hurst_exponent' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_svd_entropy' + suffix] = np.nan
            rri_nonlinear_statistics['diff2_rri_petrosian_fractal_dimension' + suffix] = np.nan

        return rri_nonlinear_statistics

    @staticmethod
    def pnn(diff_rri, time):

        # Count number of rri diffs greater than the specified time
        nn = sum(abs(diff_rri) > time)

        # Compute pNN
        pnn = nn / len(diff_rri) * 100

        return pnn

    @staticmethod
    def calculate_pearson_correlation_statistics(rri, suffix=''):

        # Empty dictionary
        pearson_correlation_statistics = dict()

        # Calculate Pearson correlation
        pearson_coeff_p1, pearson_p_value_p1 = sp.stats.pearsonr(rri[0:-2], rri[1:-1])
        pearson_coeff_p2, pearson_p_value_p2 = sp.stats.pearsonr(rri[0:-3], rri[2:-1])
        pearson_coeff_p3, pearson_p_value_p3 = sp.stats.pearsonr(rri[0:-4], rri[3:-1])

        # Get features
        pearson_correlation_statistics['rri_p1_pearson_coeff' + suffix] = pearson_coeff_p1
        pearson_correlation_statistics['rri_p1_pearson_p_value' + suffix] = pearson_p_value_p1
        pearson_correlation_statistics['rri_p2_pearson_coeff' + suffix] = pearson_coeff_p2
        pearson_correlation_statistics['rri_p2_pearson_p_value' + suffix] = pearson_p_value_p2
        pearson_correlation_statistics['rri_p3_pearson_coeff' + suffix] = pearson_coeff_p3
        pearson_correlation_statistics['rri_p3_pearson_p_value' + suffix] = pearson_p_value_p3

        return pearson_correlation_statistics

    @staticmethod
    def calculate_rri_spectral_statistics(rri, rri_ts, suffix=''):

        # Empty dictionary
        rri_spectral_statistics = dict()

        if len(rri) > 3:

            # Zero the time array
            rri_ts = rri_ts - rri_ts[0]

            # Set resampling rate
            fs = 10  # Hz

            # Generate new resampling time array
            rri_ts_interp = np.arange(rri_ts[0], rri_ts[-1], 1 / float(fs))

            # Setup interpolation function
            tck = interpolate.splrep(rri_ts, rri, s=0)

            # Interpolate rri on new time array
            rri_interp = interpolate.splev(rri_ts_interp, tck, der=0)

            # Set frequency band limits [Hz]
            vlf_band = (0, 0.04)    # Very low frequency
            lf_band = (0.04, 0.15)  # Low frequency
            hf_band = (0.15, 0.6)   # High frequency
            vhf_band = (0.6, 2)     # High frequency

            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=rri_interp, fs=fs)

            # Get frequency band indices
            vlf_index = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
            lf_index = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
            hf_index = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])
            vhf_index = np.logical_and(fxx >= vhf_band[0], fxx < vhf_band[1])

            # Compute power in each frequency band
            vlf_power = np.trapz(y=pxx[vlf_index], x=fxx[vlf_index])
            lf_power = np.trapz(y=pxx[lf_index], x=fxx[lf_index])
            hf_power = np.trapz(y=pxx[hf_index], x=fxx[hf_index])
            vhf_power = np.trapz(y=pxx[vhf_index], x=fxx[vhf_index])

            # Compute total power
            total_power = vlf_power + lf_power + hf_power + vhf_power

            # Compute spectral ratios
            rri_spectral_statistics['rri_low_high_spectral_ratio' + suffix] = lf_power / hf_power
            rri_spectral_statistics['rri_low_very_high_spectral_ratio' + suffix] = lf_power / vhf_power
            rri_spectral_statistics['rri_low_frequency_power' + suffix] = (lf_power / total_power) * 100
            rri_spectral_statistics['rri_high_frequency_power' + suffix] = (hf_power / total_power) * 100
            rri_spectral_statistics['rri_very_high_frequency_power' + suffix] = (vhf_power / total_power) * 100
            rri_spectral_statistics['rri_freq_max_frequency_power' + suffix] = \
                fxx[np.argmax(pxx[np.logical_and(fxx >= lf_band[0], fxx < vhf_band[1])])]
            rri_spectral_statistics['rri_power_max_frequency_power' + suffix] = \
                np.max(pxx[np.logical_and(fxx >= lf_band[0], fxx < vhf_band[1])])

        else:
            # Compute spectral ratios
            rri_spectral_statistics['rri_low_high_spectral_ratio' + suffix] = np.nan
            rri_spectral_statistics['rri_low_very_high_spectral_ratio' + suffix] = np.nan
            rri_spectral_statistics['rri_low_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_high_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_very_high_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_freq_max_frequency_power' + suffix] = np.nan
            rri_spectral_statistics['rri_power_max_frequency_power' + suffix] = np.nan

        return rri_spectral_statistics

    def calculate_rpeak_detection_statistics(self):

        # Empty dictionary
        rpeak_detection_statistics = dict()

        # Get median rri
        if len(self.rri) > 0:

            # Compute median rri
            rri_avg = np.median(self.rri)

        else:

            # Define possible rri's
            th1 = 1.5  # 40 bpm
            th2 = 0.3  # 200 bpm

            # Compute mean rri
            rri_avg = (th1 + th2) / 2

        # Calculate waveform duration in seconds
        time_duration = np.max(self.ts)

        # Calculate theoretical number of expected beats
        beat_count_theory = np.ceil(time_duration / rri_avg)

        # Calculate percentage of observed beats to theoretical beats
        rpeak_detection_statistics['detection_success'] = len(self.rpeaks) / beat_count_theory

        # Calculate percentage of bad rpeaks
        if self.rpeaks_bad is None:
            rpeak_detection_statistics['rpeaks_rejected'] = 0.0
        else:
            rpeak_detection_statistics['rpeaks_rejected'] = len(self.rpeaks_bad) / len(self.rpeaks)

        return rpeak_detection_statistics
