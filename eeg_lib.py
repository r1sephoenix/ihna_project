from typing import Dict, Tuple
import mne
import datetime as dt
from tqdm.notebook import tqdm
from settings import EEGSettings
import os
import numpy as np
import pandas as pd
from mne.time_frequency import psd_multitaper


class Data:
    """
    Data class (powers/eeg time series for each subject)

    Parameters
    ----------
    features : str
        type of features to return
    settings : EEGSettings
        settings with preprocessing parameters

    Attributes
    __________
    info: dict
        return info about channels positions
    ch_names: list of string
        names of channels
    """

    def __init__(self, settings, features='powers'):
        montage = mne.channels.make_standard_montage(settings.montage_name)
        dict_times: Dict[str, Dict[int, Tuple[int, int]]] = {}
        epochs_list_fin =[]
        for it, ind in enumerate(tqdm(settings.order_indexes)):
            epochs_list, dict_times[ind] = self._create_epochs(settings.events, settings.files[ind])
            for teta in range(len(epochs_list)):
                new_names = dict(
                    (ch_name,
                     ch_name.replace('-', '').replace('Chan ', 'E').replace('CAR', '').replace('EEG ', '')
                     .replace('CA', '').replace(' ', ''))
                    for ch_name in epochs_list[teta].ch_names)
                epochs_list[teta].rename_channels(new_names).set_montage(montage). \
                    drop_channels(settings.channels_to_drop)
            if features == 'powers':
                pass
            else:
                epochs_list_fin.append(epochs_list)
        if features == 'epochs':
            data = np.concatenate(([epochs_list[i].get_data()
                                  for i in range(len(settings.events))]), axis=0)

                if it == 0:
                    self._data = data
                else:
                    self._data = np.concatenate((self.data, data), axis=0)
            else:
                raise ValueError('please type powers or epochs')
            # self.__setitem__ (ind, epochs_list)
        self.chan_names = epochs_list[0].ch_names

    def __getitem__(self, key):
        if key == 'mat_group':
            ret = self

        return self.__dict__[key]

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
        epochs_list, shapes_dict = [], {}
        for event_n, event_ind in enumerate(events_list):
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
            if event_n == 0:
                shapes_dict[event_ind] = (0, j-1)
            else:
                new_start = shapes_dict[events_list[event_n-1]][1]
                shapes_dict[event_ind] = (new_start+1, new_start+j)
            epochs_list.append(epochs)
        return epochs_list, shapes_dict

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


def create_results_folder(path_res='/Users/ilamiheev/Downloads/results_ihna'):
    """
    function that creates new folder
    Parameters
    ----------
    path_res: str
        path for new folder
    """
    path_res = path_res + f'{dt.datetime.now().strftime("%m%d%H:%M:%S")}'
    path_subj = os.path.join(path_res, 'subjects_classification')
    path_unite_subj = os.path.join(path_res, 'unite_subjects')
    path_unite_topo = os.path.join(path_unite_subj, 'unite_topo')
    path_group = os.path.join(path_res, 'group_classification')
    path_group_topo = os.path.join(path_group, 'group_topo')
    for path in [path_res, path_subj, path_unite_subj, path_unite_topo, path_group, path_group_topo]:
        os.makedirs(path)
