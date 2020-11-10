import mne
import glob
import os
import re
import numpy as np
import pandas as pd


class Data:
    def __init__(self, del_chan, classes, fr_bands):

        global epochs_list
        subj_list_table, subj_list_features = [], []
        epochs_list_fin = []
        for file_name in files:
            paths = glob.glob(path + '/{0}/Reals/*.edf'.format(file_name))
            epochs_list = self._create_epochs(events_list, paths)
            for teta in range(len(epochs_list)):
                new_names = dict(
                    (ch_name,
                     ch_name.replace('-', '').replace('Chan ', 'E').replace('CAR', '').replace('EEG ', '')
                     .replace('CA', '').replace(' ', ''))
                    for ch_name in epochs_list[teta].ch_names)
                epochs_list[teta].rename_channels(new_names)
                epochs_list[teta].set_montage(montage)
                epochs_list[teta].drop_channels(chan_drop)
            feat_list, tabl_list = self._eeg_power_band(epochs_list)
            subj_list_table.append(tabl_list)
            subj_list_features.append(feat_list)
            epochs_list_fin.append(epochs_list)
        self.chan_names = epochs_list[0].ch_names
        self.dictionary = {'epochs': epochs_list_fin, 'powers':subj_list_table, 'mean_powers': }

    def __getitem__(self, key):
        return self.dictionary[key]

    def __iter__(self):
        return iter(self.dictionary)

    def _create_epochs(self, events_list, paths):
        epochs_list = []
        for event_ind in events_list:
            j = 0
            for i in [x for x in paths if '{0}'.format(event_ind) in x]:
                event_id = dict(a=event_ind)
                raw = mne.io.read_raw_edf(i, preload=True, verbose=False)
                raw.filter(3, 25., fir_design='firwin')
                if len(raw.times) / / 500 < 10:
                    continue
                new_events = mne.make_fixed_length_events(raw, id=event_ind, start=5, duration=1)
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

    def _eeg_power_band(self, epochs_list):
        fin_table, fin_feat = [], []
        for beta in range(len(epochs_list)):
            psds, freqs = psd_multitaper(epochs_list[beta])
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
