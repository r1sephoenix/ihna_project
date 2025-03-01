import pickle
import mne
import numpy as np
from itertools import compress
import h5io
from joblib import Parallel, delayed
from tqdm import tqdm

from ihna_project.settings import EEGSettings
from scipy import stats
from matplotlib import pyplot as plt, cm
from mne.decoding import LinearModel, get_coef
from mne.stats import permutation_cluster_test
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale
from lightgbm import LGBMClassifier as lgbm


def eeg_power_band(result: dict, fr_bands: dict[str, list[int]]) -> dict:
    """
    Function for evaluation of relative spectra power in specific frequency bands
    Parameters
    ----------
    epochs_list
    fr_bands

    Returns
    -------
    :rtype: tuple
    :return: features for models, features for stat tests
    """
    name = ""
    with open(f"preprocessed_data/{name}.pkl", "rb") as file:
        result[name] = pickle.load(file)
    dict_spectra = {"mean_spectra": [], "spectra_feat": []}
    for ind in range(len(result["epochs"])):
        psds, freqs = (
            result["epochs"]
            .compute_psd(method="multitaper", verbose=False)
            .get_data(return_freqs=True)
        )
        psds_mean_spectra = np.mean(psds, axis=0)
        psds /= psds.sum(axis=-1)[..., None]
        psds_mean_spectra /= psds_mean_spectra.sum(axis=-1)[..., None]
        psd_m, psd_f = [], []
        for fmin, fmax in fr_bands.values():
            freq_mask = (fmin < freqs) & (freqs < fmax)
            data_mean_spectra, data_feat = psds_mean_spectra[..., freq_mask].mean(
                axis=-1
            ), psds[..., freq_mask].mean(axis=-1)
            psd_f.append(data_feat)
            psd_m.append(data_mean_spectra)
        dict_spectra.update(
            {
                key: val.append(new_val)
                for key, val in dict_spectra.items()
                for new_val in [psd_m, psd_f]
            }
        )
    return dict_spectra


def subj_proc(settings: EEGSettings, paths: list[str], s_ind, info=True) -> dict:
    """

    Returns
    -------

    """
    epochs_list, result = [], {}
    montage = mne.channels.make_standard_montage(settings.montage_name)
    for event_ind in settings.events:
        epochs, j = None, 0
        for i in [x for x in paths if "{0}".format(event_ind) in x]:
            raw = mne.io.read_raw_edf(i, verbose="ERROR")
            if len(raw.times) // 500 < 10:
                continue
            new_events = mne.make_fixed_length_events(
                raw, id=event_ind, start=5, duration=2, overlap=0
            )
            if j == 1:
                epochs = mne.concatenate_epochs(
                    [
                        mne.Epochs(
                            raw,
                            new_events,
                            event_id=event_ind,
                            tmin=0,
                            tmax=2,
                            baseline=None,
                            flat=dict(eeg=1e-20),
                            preload=True,
                            verbose="ERROR",
                        ),
                        epochs,
                    ],
                    verbose="ERROR",
                )
            else:
                epochs = mne.Epochs(
                    raw,
                    new_events,
                    event_id=event_ind,
                    tmin=0,
                    tmax=2,
                    baseline=None,
                    flat=dict(eeg=1e-20),
                    preload=True,
                    verbose="ERROR",
                )
                j += 1
        if epochs is not None:
            epochs_list.append(epochs.copy())
    if not epochs_list:
        raise ValueError(
            f"{s_ind} subject data is too short to be processed. Please check files manually"
        )
    for epoch in range(len(epochs_list)):
        new_names = {
            (
                ch_name,
                ch_name.replace("-", "")
                .replace("Chan ", "E")
                .replace("CAR", "")
                .replace("EEG ", "")
                .replace("CA", "")
                .replace(" ", ""),
            )
            for ch_name in epochs_list[epoch].ch_names
        }
        epochs_list[epoch].rename_channels(new_names, verbose="ERROR").set_montage(
            montage, verbose="ERROR"
        ).drop_channels(settings.channels_to_drop)
    result["epochs"] = epochs_list
    if info:
        result["chan_names"] = epochs_list[0].ch_names
        result["info"] = mne.create_info(
            epochs_list[0].info.ch_names, ch_types="eeg", sfreq=250
        ).set_montage(montage)
    return result


def create_dataset(
    settings,
    res=("raw_epochs", "spectra_feat"),
    save=True,
    ret=True,
    save_names="default",
    rest=False,
):
    """
    function for dataset creation
    Parameters
    ----------
    settings: EEGSettings
        settings
    res: tuple[str]
        specify raw epochs/spectra features or both. for saving on disc choose default value
    save: bool
        save on disc option
    ret: bool
        option to return values
    save_names: str or dict
        names of files
    rest: bool
        for resting state data
    Returns
    -------
    :rtype: dict
    :return: only if argument ret is True. Return dict with keys in ('raw_epochs', 'spectra_feat', 'mean_spectra',
    'chan_names')
    """
    if len(res) != 2 and save:
        raise ValueError("Please, use default res attribute")
    if (save_names != "default" and len(save_names) != 4) and save:
        raise ValueError(
            "Please, specify all files names - for epochs, spectra features, mean spectra features and "
            "ch_names"
        )
    if save_names == "default":
        n_l = [
            "raw_epochs",
            "spectra_feat",
            "mean_spectra",
            "chan_names",
            "info_object",
        ]
        save_dict = dict(zip(n_l, n_l))
    else:
        save_dict = save_names
    (
        subj_list_mean_spectra,
        subj_list_features,
        e_list,
        l_res,
        res_names,
        s,
        chan_names,
        info,
    ) = ([], [], [], [], [], 0, None, None)
    result = {}
    n_subj = len(settings.files)
    for subj_paths, s_ind in tqdm(
        zip(settings.files.values(), settings.files.keys()),
        total=n_subj,
        desc=f"Creation of dataset for {n_subj} subjects",
        position=0,
    ):
        paths, epochs_list = subj_paths, []
        if s == 0:
            info = True
            s += 1
        results = subj_proc(settings, paths, s_ind, info=info)
        for name in res:
            if name == "":
                pass
        s += 1
        if "raw_epochs" in res:
            e_list.append(epochs_list)
        if "spectra_feat" in res:
            feat_list, tabl_list = eeg_power_band(epochs_list, settings.fr_bands)
            subj_list_mean_spectra.append(tabl_list)
            subj_list_features.append(feat_list)

    if "raw_epochs" in res:
        l_res.append(e_list)
        res_names.append("raw_epochs")
    if "spectra_feat" in res:
        spectra_feat = [
            [np.stack(subj_list_features[i][j], axis=1) for j in range(3)]
            for i in range(len(settings.files))
        ]
        l_res.extend([spectra_feat, np.stack(subj_list_mean_spectra)])
        res_names.extend(["spectra_feat", "mean_spectra"])
    res_names.extend(["chan_names", "info_object"])
    l_res.extend([chan_names, info])
    if save:
        save_folder = "preprocessed_data"
        for r, name in tqdm(
            zip(l_res, list(save_dict.values())), total=len(save_dict), desc=f"Saving"
        ):
            with open(f"{save_folder}/{name}.pkl", "wb") as file:
                pickle.dump(r, file)
    if ret:
        return dict(zip(res_names, l_res))


def estimate_features(data, feat, n_jobs):
    cfg = ""
    condition = ""
    subjects = []
    for i in feat:
        if i == "spectra":
            features = Parallel(n_jobs=n_jobs)(
                delayed(eeg_power_band)(sub, cfg=cfg, condition=condition)
                for sub in subjects
            )
            out_fname = ""
        elif i == "cov":
            pass
        else:
            raise ValueError
        h5io.write_hdf5(out_fname, features, overwrite=True)


def load_data(load_names):
    """
    function for load files
    Parameters
    ----------
    load_names: list[str]
        names of files to load

    Returns
    -------
    :rtype: dict
    :return: dictionary with files
    """
    result = dict()
    for name in tqdm(load_names, desc="Loading files", total=len(load_names)):
        with open(f"preprocessed_data/{name}.pkl", "rb") as file:
            result[name] = pickle.load(file)
    return result


def predict_lm(data: list[np.ndarray], eeg_param: list[dict, list], ps=None) -> tuple:
    """
    Function for fitting logistic regression model
    Parameters
    ----------
    data: list[np.ndarray]
    eeg_param:
       [fr_bands, channels_names]

    Returns
    -------
    :rtype: tuple
    :return: ac, roc_auc, interp_tpr, coefs
    """
    mean_fpr = np.linspace(0, 1, 100)
    coefs = np.zeros((2, len(eeg_param[0]), len(eeg_param[1])))
    model = make_pipeline(
        StandardScaler(),
        LinearModel(
            LogisticRegressionCV(
                Cs=list(np.power(10.0, np.arange(-10, 10))),
                penalty="l2",
                scoring="roc_auc",
                random_state=0,
                max_iter=10000,
                fit_intercept=True,
                solver="newton-cg",
                cv=ps,
                class_weight="balanced",
                tol=10,
            )
        ),
    )
    model.fit(data[0], data[2])
    y_predict, y_predict_pr = model.predict(data[1]), model.predict_proba(data[1])
    ac = balanced_accuracy_score(data[3], y_predict)
    fpr, tpr, _ = roc_curve(data[3], y_predict_pr[:, 1])
    roc_auc, interp_tpr = auc(fpr, tpr), np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    for name, j in zip(["patterns_", "filters_"], [0, 1]):
        coefs_n = get_coef(model, name, inverse_transform=True)
        coefs[j, ...] = coefs_n.reshape(len(eeg_param[0]), -1)
    return ac, roc_auc, interp_tpr, coefs, y_predict_pr


def predict_lgbm(data, eeg_param, ps=None):
    """
    function for LightGBM model
    Parameters
    ----------
    data: list[np.ndarray]
    eeg_param: list[dict, list]
        [fr_bands, channels_names]
    Returns
    -------
    :rtype: tuple
    :return: ac, roc_auc, interp_tpr, feature_importances, y_predict_pr
    """
    mean_fpr = np.linspace(0, 1, 100)
    sc = StandardScaler()
    data[0] = sc.fit_transform(data[0])
    data[1] = sc.transform(data[1])
    fit_params = {"objective": "binary"}
    m = lgbm(**fit_params)
    param_grid = {
        "reg_alpha": list(np.power(10.0, np.arange(-5, 5))),
        "reg_lambda": list(np.power(10.0, np.arange(-5, 5))),
    }
    if ps is None:
        ps = 5
    model = GridSearchCV(m, param_grid, cv=ps)
    model.fit(data[0], data[2])
    feature_importance = (
        (
            model.best_estimator_.feature_importances_
            / sum(model.best_estimator_.feature_importances_)
        )
        * 100
    ).reshape((len(eeg_param[0]), len(eeg_param[1])))
    y_predict, y_predict_pr = model.predict(data[1]), model.predict_proba(data[1])
    ac = balanced_accuracy_score(data[3], y_predict)
    fpr, tpr, _ = roc_curve(data[3], y_predict_pr[:, 0], pos_label=0)
    roc_auc, interp_tpr = auc(fpr, tpr), np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return ac, roc_auc, interp_tpr, feature_importance, y_predict_pr


def c_bar(v_min, v_max, c_map, fig, coord=None):
    """
    add colorbar to figure
    Parameters
    ----------
    v_min
        min value
    v_max
        max value
    c_map
        color map
    fig: figure
        figure to add new axes
    coord: list
        position of new axes
    Returns
    -------

    """
    if coord is None:
        coord = [1, 0.3, 0.03, 0.38]
    m = cm.ScalarMappable(cmap=f"{c_map}")
    m.set_array(np.array([v_min, v_max]))
    cax = fig.add_axes(coord)
    cb = fig.colorbar(m, cax)
    cb.ax.tick_params(labelsize=40)


def plot_patterns(coefs_list, eeg_param, subfig=None):
    """
    function that plot filters and patterns on topo
    Parameters
    ----------
    data: list[np.ndarray]
    eeg_param: tuple
        (fr_bands, info)
    subfig
    Returns
    -------
    :rtype:matplotlib.figure.Figure
    """
    min_max, s, vmax, vmin, ret, fig = [], coefs_list[0].shape, 1, -1, False, None
    if subfig is None:
        ret = True
        fig, axes = plt.subplots(nrows=2, ncols=len(eeg_param[0]), figsize=(30, 20))
    else:
        axes = subfig.subplots(nrows=2, ncols=len(eeg_param[0]))
    for i in range(len(coefs_list)):
        min_max.append(
            minmax_scale(
                coefs_list[i].reshape(2, -1), feature_range=(vmin, vmax), axis=1
            ).reshape(s)
        )
    data = np.mean(min_max, axis=0)
    for name, pos, plot_name, ind in zip(
        ("patterns_", "filters_"), (0.82, 0.5), ("Patterns", "Filters"), (0, 1)
    ):
        for i, key in enumerate(list(eeg_param[0].keys())):
            mne.viz.plot_topomap(
                data[ind, i, :],
                eeg_param[1],
                vmin=vmin,
                vmax=vmax,
                axes=axes[ind, i],
                show=False,
            )
            axes[ind, i].set_title(
                label=f"{eeg_param[0][key][0]}-{eeg_param[0][key][1]} Hz",
                fontdict={"fontsize": 55, "fontweight": "semibold"},
            )
            mne.viz.tight_layout()
    if ret:
        if fig is not None:
            c_bar(vmin, vmax, "RdBu_r", fig)
        return fig
    else:
        return axes


def plot_feature_importance(data, eeg_param, subfig=None):
    """
    function that plot feature importances on topo
    Parameters
    ----------
    data: list[np.ndarray]
    eeg_param: tuple
        (fr_bands, info)
    subfig
    Returns
    -------
    figure
    """
    vmin, vmax, min_max, s, ret, fig = 0, 1, [], data[0].shape, False, None
    if subfig is None:
        fig, axes = plt.subplots(nrows=1, ncols=len(eeg_param[0]), figsize=(30, 20))
        ret = True
    else:
        axes = subfig.subplots(nrows=1, ncols=len(eeg_param[0]))
    for i in range(len(data)):
        min_max.append(
            minmax_scale(data[i].reshape(1, -1), feature_range=(0, 1), axis=1).reshape(
                s
            )
        )
    data = np.mean(min_max, axis=0)
    for i, key in enumerate(list(eeg_param[0].keys())):
        mne.viz.plot_topomap(
            data[i, :],
            eeg_param[1],
            vmin=0,
            vmax=1,
            axes=axes[i],
            cmap="hot",
            show=False,
        )
        axes[i].set_title(
            label=f"{eeg_param[0][key][0]}-{eeg_param[0][key][1]} Hz",
            fontdict={"fontsize": 55, "fontweight": "semibold"},
        )
        mne.viz.tight_layout()
    if ret:
        if fig is not None:
            c_bar(vmin, vmax, "hot", fig)
        return fig
    else:
        return axes


def plot_clusters(data, l_ind, cl_param):
    """
    function that plot significant clusters on topo
    Parameters
    ----------
    data: np.ndarray
        samples
    l_ind: list[list]
        groups indexes
    cl_param: tuple
        (fr_bands, info)
    Returns
    -------
    :rtype:matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(nrows=3, ncols=len(cl_param[0]), figsize=(30, 20))
    vmin, thr, res_ar = 0, -stats.t.ppf(0.05 / 2, 1), np.zeros(data[0, 0].shape)
    for i, co, name in zip(range(3), [0.1, 0.38, 0.66], ["A", "B", "C"]):
        t_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
            [data[l_ind[0], i, ...], data[l_ind[1], i, ...]],
            threshold=thr,
            adjacency=None,
            out_type="mask",
            n_permutations=1024,
            verbose=False,
        )
        t_obs_plot = np.nan * np.ones_like(t_obs)
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= 0.05:
                t_obs_plot[c] = t_obs[c]
        np.nan_to_num(t_obs_plot, copy=False)
        if np.count_nonzero(t_obs_plot) > 0:
            res_ar = t_obs_plot
        mr_ar = res_ar > 0
        vmax = np.amax(res_ar)
        for j, key in enumerate(list(cl_param[0].keys())):
            mne.viz.plot_topomap(
                res_ar[j],
                cl_param[1],
                vmin=vmin,
                vmax=vmax,
                axes=axes[i, j],
                show=False,
                cmap="hot",
                mask=mr_ar[j].reshape(-1, 1),
            )
            axes[i, j].set_title(
                label=f"{cl_param[0][key][0]}-{cl_param[0][key][1]} Hz",
                fontdict={"fontsize": 50, "fontweight": "semibold"},
            )
            mne.viz.tight_layout()
            c_bar(vmin, vmax, "hot", fig, coord=[1, co, 0.03, 0.25])
            axes[i, 0].text(
                -0.2, 0, f"{name}", fontdict={"fontsize": 50, "fontweight": "semibold"}
            )
    return fig


def plot_roc_curves(tprs, aucs, list_predictions, list_y_true, plot_all=True, ax=None):
    """
    function that plot roc curves
    Parameters
    ----------
    tprs: list[float]
    aucs: list[float]
    list_predictions: list[np.ndarray[float]]
        predictions
    list_y_true: list[list[int]]
        true target values
    plot_all: bool
        Plot or not to plot roc for each fold. Default True
    ax:
    Returns
    -------
    :rtype:matplotlib.figure.Figure
    """
    mean_fpr, ret, fig = np.linspace(0, 1, 100), False, None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ret = True
    if plot_all:
        for i, (y_true, pred) in enumerate(zip(list_y_true, list_predictions)):
            RocCurveDisplay.from_predictions(
                y_true=y_true,
                y_pred=pred[:, 1].tolist(),
                name=f"ROC fold {i + 1}",
                ax=ax,
            )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc, std_auc = auc(mean_fpr, mean_tpr), np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(
        mean_tpr - std_tpr, 0
    )
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
        lw=2,
        alpha=0.8,
    )
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label="\u00B1  1 std. dev.",
    )
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.tick_params(axis="both", labelsize=65)
    ax.set_xlabel("True Positive Rate", fontsize=70)
    ax.set_ylabel("False Positive Rate", fontsize=70)
    ax.legend(loc="lower right", fontsize=55)
    if ret:
        return fig
    else:
        pass


def merge_fig(
    fig_type, data, s_ind, metrics, results, settings, inf, roc_plot_all=False, pl=True
):
    """
    merge several figures

    Parameters
    ----------
    fig_type
    data: ndarray
    s_ind
    metrics: dict
    results: ndarray
    settings
    inf
    roc_plot_all: bool
    pl: bool

    Returns
    -------
    :rtype:matplotlib.figure.Figure
    """
    if fig_type not in [1, 2]:
        raise ValueError(
            "Please type correct value: 1 for filters/patterns, 2 for feature importance"
        )
    fig = plt.figure(constrained_layout=True, figsize=(75, 60))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    axs_left, axs_right, bars_l = (
        subfigs[0].subfigures(3, 1),
        subfigs[1].subfigures(3, 1),
        [],
    )
    for side, pos in zip(["left", "right"], [1, 2]):
        bars_l.append(
            tqdm(
                range(3),
                total=3,
                desc=f"Draw 3 {side} plots",
                leave=False,
                position=pos,
            )
        )
    for n, (ax, name) in enumerate(zip(axs_left, ["A", "B", "C"])):
        bars_l[0].update(1)
        ax.text(
            -0.05, 0.5, f"{name}", fontdict={"fontsize": 90, "fontweight": "semibold"}
        )
        if fig_type == 1:
            ax_bar = plot_patterns(
                list(compress(data[s_ind, n, ...], results[s_ind, n, 1] > 0.5)),
                [settings.fr_bands, inf],
                ax,
            )
            if n == 2:
                m = cm.ScalarMappable(cmap="RdBu_r")
                m.set_array(np.array([-1, 1]))
                cb = subfigs[0].colorbar(m, ax=ax_bar, shrink=0.6, location="bottom")
                cb.ax.tick_params(labelsize=80)
        if fig_type == 2:
            ax_bar = plot_feature_importance(
                list(compress(data[s_ind, n, ...], results[s_ind, n, 1] > 0.5)),
                [settings.fr_bands, inf],
                ax,
            )
            if n == 2:
                m = cm.ScalarMappable(cmap="hot")
                m.set_array(np.array([0, 1]))
                cb = subfigs[0].colorbar(
                    m, ax=ax_bar, shrink=0.6, pad=0.3, location="bottom"
                )
                cb.ax.tick_params(labelsize=80)
    for n, (ax, name) in enumerate(zip(axs_right, ["D", "E", "F"])):
        bars_l[1].update(1)
        ax.text(
            -0.05, 0.5, f"{name}", fontdict={"fontsize": 90, "fontweight": "semibold"}
        )
        ax = ax.subplots(nrows=1, ncols=1)
        plot_roc_curves(
            [metrics["tprs"][i][n] for i in s_ind],
            [metrics["aucs"][i][n] for i in s_ind],
            [metrics["pr_v"][i][n] for i in s_ind],
            [metrics["true_v"][i][n] for i in s_ind],
            plot_all=roc_plot_all,
            ax=ax,
        )
    if pl:
        plt.show()
    else:
        plt.close(fig)
    return fig
