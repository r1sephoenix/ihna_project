from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import transformers
import importlib
import random
import settings
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

importlib.reload(transformers)
CSP = transformers.CSP
CSPCv = transformers.CSPCv
PCS = transformers.ProjCommonSpaceCV
test_size = 10

results = np.zeros((test_size, len(settings.dict_cls), 2))
coefs = np.zeros((len(index_new_mat), len(settings.dict_cls), 2, len(settings.fr_bands), len(data['chan_names'])))
metrics = {k: [] for k in ['tprs', 'aucs', 'true_v', 'pr_v']}


def csp_pipeline(cov_data, settings, index_new_mat, index_new_not_mat):
    for count, (ind_mat, ind_not_mat) in tqdm(enumerate(zip(index_new_mat, index_new_not_mat)),
                                              total=len(index_new_mat),
                                              desc=f'Evaluation for {len(index_new_mat)} folds', position=0):
        metrics_loop = {k: [] for k in range(4)}
        for ind in tqdm(range(3), total=3, position=1):
            # subjects for cv
            ind_mat_cv, ind_not_mat_cv = random.choice([i for i in index_new_mat if i != ind_mat]), random.choice(
                [i for i in index_new_not_mat if i != ind_not_mat])
            y_train = [1] * sum([cov_data[i][ind][0].shape[0] for i in index_mat if i not in [ind_mat_cv, ind_mat]]) + [
                0] * sum(
                [cov_data[i][ind][0].shape[0] for i in index_not_mat if i not in [ind_not_mat_cv, ind_not_mat]])
            x_train = [cov_data[i][ind] for i in index_all if
                       i not in [ind_mat_cv, ind_not_mat_cv, ind_mat, ind_not_mat]]
            x_train = np.concatenate([np.stack(x_train[i]) for i in range(len(x_train))], axis=1).transpose(1, 0, 2, 3)
            csp_l = [CSPCv(metric='euclid', log=True) for n in range(7)]
            f = [csp_l[n].fit(x_train[:, n, ...], y_train).filters_ for n in range(7)]
            p = [csp_l[n].fit(x_train[:, n, ...], y_train).patterns_ for n in range(7)]
            y_val = [1] * cov_data[ind_mat_cv][ind][0].shape[0] + [0] * cov_data[ind_not_mat_cv][ind][0].shape[0]
            x_val = [cov_data[i][ind] for i in [ind_mat_cv, ind_not_mat_cv]]
            x_val = np.concatenate([np.stack(x_val[i]) for i in range(len(x_val))], axis=1).transpose(1, 0, 2, 3)
            y_test = [1] * cov_data[ind_mat][ind][0].shape[0] + [0] * cov_data[ind_not_mat][ind][0].shape[0]
            x_test = [cov_data[i][ind] for i in [ind_mat, ind_not_mat]]
            x_test = np.concatenate([np.stack(x_test[i]) for i in range(len(x_test))], axis=1).transpose(1, 0, 2, 3)
            ps = PredefinedSplit(np.concatenate([np.zeros(len(x_train)) - 1, np.ones(len(x_val))]))
            x_train, y_train = np.concatenate((x_train, x_val), axis=0), y_train + y_val
            lr_params = {'penalty': 'l2', 'random_state': 0, 'max_iter': 10000, 'fit_intercept': True,
                         'solver': 'newton-cg', 'class_weight': 'balanced', 'tol': 10}
            pipe = Pipeline(steps=[('csp', CSP(csp_f=f)), ('scaler', StandardScaler()),
                                   ('logistic', LogisticRegression(**lr_params))])
            param_grid = {'csp__nfilter': [5], "logistic__C": [0.001]}
            search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=ps, n_jobs=2, verbose=10)
            search.fit(x_train, y_train)
            y_predict, y_predict_pr = search.predict(x_test), search.predict_proba(x_test)
            ac = balanced_accuracy_score(y_test, y_predict)
            fpr, tpr, _ = roc_curve(y_test, y_predict_pr[:, 1])
            mean_fpr = np.linspace(0, 1, 100)
            roc_auc, interp_tpr = auc(fpr, tpr), np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            results[count, ind, ...] = ac, roc_auc
            for k, v in zip(metrics_loop.keys(), [interp_tpr, roc_auc, y_test, y_predict_pr]):
                metrics_loop[k].append(v)
    for k, v in zip(metrics.keys(), metrics_loop.values()):
        metrics[k].append(v)
