import mne
import glob
import os
import re
import numpy as np
import pandas as pd
from mne.time_frequency import psd_multitaper


class Data:
    """

    """

    def __init__(self, files, indexes, path, drop_chan, classes, fr_bands):
        """

        Parameters
        ----------
        files
        indexes
        path
        drop_chan
        classes
        fr_bands
        """
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        for it, (file_name, ind) in enumerate(zip(tqdm(files), indexes)):
            paths = glob.glob(path + '/{0}/Reals/*.edf'.format(file_name))
            epochs_list = self._create_epochs(classes, paths)
            for teta in range(len(epochs_list)):
                new_names = dict(
                    (ch_name,
                     ch_name.replace('-', '').replace('Chan ', 'E').replace('CAR', '').replace('EEG ', '')
                     .replace('CA', '').replace(' ', ''))
                    for ch_name in epochs_list[teta].ch_names)
                epochs_list[teta].rename_channels(new_names).set_montage(montage).drop_channels(drop_chan)
            for i in range(len(epochs_list)):
                data = np.concatenate(([epochs_list[i].get_data() for i in range(len(classes))]), axis=0)
            if it == 0:
                self.data = data
            else:

                self.data = np.concatenate((self.data, data), axis=0)
            # self.__setitem__ (ind, epochs_list)
        self.chan_names = epochs_list[0].ch_names
        self.fr_bands = fr_bands

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    @staticmethod
    def _create_epochs(events_list, paths):
        """

        Parameters
        ----------
        events_list
        paths

        Returns
        -------

        """
        epochs_list = []
        for event_ind in events_list:
            j = 0
            for i in [x for x in paths if '{0}'.format(event_ind) in x]:
                event_id = dict(a=event_ind)
                raw = mne.io.read_raw_edf(i, preload=True, verbose='ERROR')
                raw = raw.resample(250)
                raw.filter(3, 25., fir_design='firwin')
                if len(raw.times) // 500 < 10:
                    continue
                new_events = mne.make_fixed_length_events(raw, id=event_ind, start=5, duration=2)
                if j == 1:
                    epochs = mne.concatenate_epochs([mne.Epochs(raw, new_events, event_id=event_id, tmin=0,
                                                                tmax=1.99, baseline=None, flat=dict(eeg=1e-20),
                                                                preload=True, verbose=False), epochs])
                else:
                    epochs = mne.Epochs(raw, new_events, event_id=event_id, tmin=0, tmax=1.99, baseline=None,
                                        flat=dict(eeg=1e-20), preload=True, verbose=False)
                    j += 1
            epochs_list.append(epochs)
        return epochs_list

    def _eeg_power_band(self, key):
        """

        Parameters
        ----------
        key

        Returns
        -------

        """
        fr_bands = self.fr_bands
        fin_table, fin_feat = [], []
        for beta in range(len(self[key])):
            psds, freqs = psd_multitaper(self[key][beta])
            psds_table = np.mean(psds, axis=0)
            psds /= psds.sum(axis=-1)[..., None]
            psds_table /= psds_table.sum(axis=-1)[..., None]
            psd_table_list, psd_features_list = [], []
            for fmin, fmax in fr_bands.values():
                freq_mask = (fmin < freqs) & (freqs < fmax)
                data_table, data_feat = psds_table[..., freq_mask].mean(axis=-1), psds[..., freq_mask].mean(axis=-1)
                psd_features_list.append(data_feat)
                psd_table_list.append(data_table)
            fin_table.append(psd_table_list)
            fin_feat.append(psd_features_list)
        return fin_feat, fin_table
