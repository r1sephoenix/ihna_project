import datetime as dt
import os

import mne
import numpy as np
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt, cm
from mne.decoding import LinearModel, get_coef
from mne.time_frequency import psd_multitaper
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def eeg_power_band(epochs_list, fr_bands):
    """
    function for evaluation of powers in specific frequency bands
    Parameters
    ----------
    epochs_list
    fr_bands

    Returns
    -------
    tuple
        (features for models, features for stat tests)
    """
    fin_table, fin_feat = [], []
    for ind in range(len(epochs_list)):
        psds, freqs = psd_multitaper(epochs_list[ind])
        psds_table = np.mean(psds, axis=0)
        psds /= psds.sum(axis=-1)[..., None]
        psds_table /= psds_table.sum(axis=-1)[..., None]
        psd_bands_table_list, psd_bands_features_list = [], []
        for fmin, fmax in fr_bands.values():
            freq_mask = (fmin < freqs) & (freqs < fmax)
            data_table, data_feat = psds_table[..., freq_mask].mean(axis=-1), psds[..., freq_mask].mean(axis=-1)
            psd_bands_features_list.append(data_feat)
            psd_bands_table_list.append(data_table)
        fin_table.append(psd_bands_table_list)
        fin_feat.append(psd_bands_features_list)
    return fin_feat, fin_table


def predict_lm(data, eeg_param):
    """
    function for logistic regression model
    Parameters
    ----------
    data: tuple
    eeg_param: tuple
       (fr_bands, n_channels)
    
    Returns
    -------
    tuple
        (ac, roc_auc, interp_tpr, coefs)
    """
    mean_fpr = np.linspace(0, 1, 100)
    coefs = np.zeros((2, len(eeg_param[0]), eeg_param[1]))
    model = make_pipeline(StandardScaler(),
                          LinearModel(LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10))), penalty='l2',
                                                           scoring='roc_auc', random_state=0, max_iter=10000,
                                                           fit_intercept=True, solver='newton-cg',
                                                           class_weight='balanced', tol=10)))
    model.fit(data[0], data[2])
    y_predict, y_predict_pr = model.predict(data[1]), model.predict_proba(data[1])
    ac = balanced_accuracy_score(data[3], y_predict)
    fpr, tpr, _ = roc_curve(data[3], y_predict_pr[:, 1])
    roc_auc, interp_tpr = auc(fpr, tpr), np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    for name, j in zip(['patterns_', 'filters_'], [0, 1]):
        coefs_n = get_coef(model, name, inverse_transform=True)
        coefs[j, ...] = coefs_n.reshape(len(eeg_param[0]), -1)
    return ac, roc_auc, interp_tpr, coefs, y_predict_pr


def predict_lgbm(data, eeg_param):
    """
    function for LightGBM model
    Parameters
    ----------
    data: tuple
    eeg_param: tuple
        (fr_bands, n_channels)
    Returns
    -------   
    tuple
        (ac, roc_auc, interp_tpr, feature_importances)
    """
    mean_fpr = np.linspace(0, 1, 100)
    model = LGBMClassifier(objective='binary')
    model.fit(data[0], data[2])
    feature_importances = ((model.feature_importances_ / sum(model.feature_importances_)) * 100).reshape((eeg_param[0],
                                                                                                          eeg_param[1]))
    y_predict, y_predict_pr = model.predict(data[1]), model.predict_proba(data[1])
    ac = balanced_accuracy_score(data[3], y_predict)
    fpr, tpr, _ = roc_curve(data[3], y_predict_pr[:, 0], pos_label=0)
    roc_auc, interp_tpr = auc(fpr, tpr), np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return ac, roc_auc, interp_tpr, feature_importances


def plot_patterns(data, eeg_param):
    """
    function that plot filters and patterns on topo
    Parameters
    ----------
    data: ndarray
    eeg_param: tuple
        (fr_bands, info)
    Returns
    -------   
    figure 
    """
    vmax = 1
    vmin = -1
    fig, axes = plt.subplots(nrows=2, ncols=len(eeg_param[0]), figsize=(30, 20))
    for name, pos, plot_name, ind in zip(('patterns_', 'filters_'), (0.82, 0.5),
                                         ('Patterns', 'Filters'), (0, 1)):
        for i, key in enumerate(list(eeg_param[0].keys())):
            mne.viz.plot_topomap(data[ind, i, :], eeg_param[1], vmin=vmin, vmax=vmax, axes=axes[ind, i],
                                 show=False)
            axes[ind, i].set_title(label=f'{eeg_param[0][key][0]}-{eeg_param[0][key][1]} Hz',
                                   fontdict={'fontsize': 40, 'fontweight': 'semibold'})
            mne.viz.tight_layout()
        plt.figtext(0.5, pos, f'{plot_name}', va='center', ha='center', size=44, fontweight='semibold')
    m = cm.ScalarMappable(cmap='RdBu_r')
    m.set_array(np.array([vmin, vmax]))
    cax = fig.add_axes([1, 0.3, 0.03, 0.38])
    cb = fig.colorbar(m, cax)
    cb.ax.tick_params(labelsize=40)
    return fig


def plot_feature_importance(data, eeg_param):
    """
    function that plot feature importances on topo
    Parameters
    ----------
    data: ndarray
    eeg_param: tuple
        (fr_bands, info)
    Returns
    -------   
    figure     
    """
    vmin = 0
    vmax = np.amax(data)
    fig, axes = plt.subplots(nrows=1, ncols=len(eeg_param[0]), figsize=(30, 20))
    num = 0
    for i, key in enumerate(list(eeg_param[0].keys())):
        if i in range(7):
            evoked = mne.EvokedArray(data[i, :].reshape(-1, 1),
                                     eeg_param[1], tmin=0.)
            evoked.plot_topomap(axes=axes[num], time_format=None, colorbar=False, times=[0], cmap='hot',
                                size=5, show_names=False, vmin=0, show=False)
            axes[num].set_title(label=f'{eeg_param[0][key][0]}-{eeg_param[0][key][1]} Hz',
                                fontdict={'fontsize': 40, 'fontweight': 'semibold'})
            mne.viz.tight_layout()
            num += 1
    m = cm.ScalarMappable(cmap='hot')
    m.set_array(np.array([vmin, vmax]))
    cax = fig.add_axes([1, 0.3, 0.03, 0.38])
    cb = fig.colorbar(m, cax)
    cb.ax.tick_params(labelsize=40)
    plt.figtext(0.5, 0.7, 'Features importance', va='center', ha='center', size=60, fontweight='semibold')
    return fig


def plot_clusters(data, cl_param):
    """
    function that plot significant clusters on topo
    Parameters
    ----------
    data: ndarray
    cl_param: tuple
        (fr_bands, info, sign_cl)
    Returns
    -------
    figure
    """
    vmin = np.amin(data)
    vmax = np.amax(data)
    fig, axes = plt.subplots(nrows=1, ncols=len(cl_param[2]), figsize=(30, 20))
    num = 0
    for i, key in enumerate(list(cl_param[0].keys())):
        if i in cl_param[2]:
            evoked = mne.EvokedArray(data[i, :].reshape(-1, 1),
                                     cl_param[1], tmin=0.)
            evoked.plot_topomap(axes=axes[num], time_format=None, colorbar=False, times=[0], size=5, show_names=True,
                                show=False)
            axes[num].set_title(label=f'{cl_param[0][key][0]}-{cl_param[0][key][1]} Hz',
                                fontdict={'fontsize': 40, 'fontweight': 'semibold'})
            mne.viz.tight_layout()
            num += 1
    m = cm.ScalarMappable(cmap='Reds')
    m.set_array(np.array([vmin, vmax]))
    cax = fig.add_axes([1, 0.3, 0.03, 0.38])
    cb = fig.colorbar(m, cax)
    cb.ax.tick_params(labelsize=40)
    plt.figtext(0.5, 1.0, 'Significant clusters', va='center', ha='center', size=60, fontweight='semibold')
    return fig


def plot_roc_curves(tprs, aucs, list_predictions, list_y_true, method_name, plot_all=True):
    """

    Parameters
    ----------
    tprs: list
    aucs: list
    list_predictions: list
        predictions
    list_y_true: list
        true target values
    method_name: str
        name of ml algorithm
    plot_all: bool
        Plot or not to plot roc for each fold. Default True
    Returns
    -------
    figure
    """
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(12, 8))
    if plot_all:
        for i, (y_true, pred) in enumerate(zip(list_y_true, list_predictions)):
            RocCurveDisplay.from_predictions(y_true=y_true, y_pred=pred.tolist(), name=f"ROC fold {i + 1}",
                                             pos_label=0, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8, fontdict={'fontsize': 35})
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc, std_auc = auc(mean_fpr, mean_tpr), np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
    ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})', lw=2,
            alpha=0.8, fontdict={'fontsize': 35})
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='\u00B1  1 std. dev.',
                    fontdict={'fontsize': 35})
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f'Receiver operating characteristic {method_name}',
           fontdict={'fontsize': 40, 'fontweight': 'semibold'})
    ax.legend(loc='lower right')
    return fig


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
