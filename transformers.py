# changed transformers from https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/spatialfilters.py
# for cross validation purposes
import numpy as np
from pyriemann.utils.ajd import ajd_pham
from pyriemann.utils.mean import mean_covariance, _check_mean_method
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.tangentspace import tangent_space, untangent_space
import numpy as np
from scipy.linalg import eigh, pinv
from sklearn.base import BaseEstimator, TransformerMixin


def shrink(cov, alpha):
    n = len(cov)
    shrink_cov = (1 - alpha) * cov + alpha * np.trace(cov) * np.eye(n) / n
    return shrink_cov


def fstd(y):
    y = y.astype(np.float32)
    y -= y.mean(axis=0)
    y /= y.std(axis=0)
    return y


def _get_scale(X, scale):
    if scale == 'auto':
        scale = 1 / np.mean([np.trace(x) for x in X])
    return scale


def _check_x_df(x):
    if hasattr(x, 'values'):
        x = np.array(list(np.squeeze(x`))).astype(float)
        if x.ndim == 2:  # deal with single sample
            assert x.shape[0] == x.shape[1]
            x = x[np.newaxis, :, :]
    return x


class CSPCv(BaseEstimator, TransformerMixin):
    """CSP projection
    """

    def __init__(self, nfilter=4, metric='euclid', log=True):
        """Init."""
        self.filters_ = None
        self.patterns_ = None
        self.nfilter = nfilter
        self.metric = metric
        self.log = log

    def fit(self, x, y):
        """Train CSP spatial filters.
        Parameters
        ----------
        x : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.
        Returns
        -------
        self : CSP instance
              The CSP instance.
        """
        if not isinstance(self.nfilter, int):
            raise TypeError('nfilter must be an integer')
        _check_mean_method(self.metric)
        if not isinstance(self.log, bool):
            raise TypeError('log must be a boolean')

        if not isinstance(x, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be an array.')
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_channels')
        if len(y) != len(x):
            raise ValueError('X and y must have the same length.')
        if np.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')
        n_trials, n_channels, _ = x.shape
        classes = np.unique(y)
        # estimate class means
        c = []
        for i in classes:
            c.append(mean_covariance(x[y == i], self.metric))
        c = np.array(c)

        # Switch between binary and multiclass
        if len(classes) == 2:
            evals, evecs = eigh(c[1], c[0] + c[1])
            # sort eigenvectors
            ix = np.argsort(np.abs(evals - 0.5))[::-1]
        elif len(classes) > 2:
            evecs, d = ajd_pham(c)
            ctot = mean_covariance(c, self.metric)
            evecs = evecs.T
            # normalize
            for i in range(evecs.shape[1]):
                tmp = evecs[:, i].T @ ctot @ evecs[:, i]
                evecs[:, i] /= np.sqrt(tmp)

            mutual_info = []
            # class probability
            pc = [np.mean(y == c) for c in classes]
            for j in range(evecs.shape[1]):
                a = 0
                b = 0
                for i, c in enumerate(classes):
                    tmp = evecs[:, j].T @ c[i] @ evecs[:, j]
                    a += pc[i] * np.log(np.sqrt(tmp))
                    b += pc[i] * (tmp ** 2 - 1)
                mi = - (a + (3.0 / 16) * (b ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]
        else:
            raise ValueError("Number of classes must be >= 2.")

        # sort eigenvectors
        evecs = evecs[:, ix]

        # spatial patterns
        pat = np.linalg.pinv(evecs.T)

        self.filters_ = evecs.T
        self.patterns_ = pat.T
        return self

    def transform(self, x, nfilter=None, filters=None):
        """Apply spatial filters.
        Parameters
        ----------
        filters
        nfilter
        x : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        Returns
        -------
        Xf : ndarray, shape (n_trials, n_filters) or \
                ndarray, shape (n_trials, n_filters, n_filters)
            Set of spatialy filtered log-variance or covariance, depending on
            the 'log' input parameter.
        """
        if not isinstance(x, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if nfilter is None:
            nfilter = self.nfilter
        if filters is None:
            filters = self.filters_
        if x[0].shape[1] != filters.shape[1]:
            raise ValueError("Data and filters dimension must be compatible.")
        x_filt = filters[0:nfilter, :] @ x @ filters[0:nfilter, :].T
        if self.log:
            out = np.zeros(x_filt.shape[:2])
            for i, x in enumerate(x_filt):
                out[i] = np.log(np.diag(x))
            return out
        else:
            return x_filt


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, csp_f=None, nfilter=5):
        self.nfilter = nfilter
        self.csp_f = csp_f

    def fit(self, x_train, y_train):
        return self

    def transform(self, x_test):
        l = len(x_test)
        x = np.stack([CSPCv(nfilter=4, metric='euclid', log=True).transform(x_test[:, n], nfilter=self.nfilter,
                                                                            filters=self.csp_f[n]) for n in range(7)],
                     axis=1) \
            .reshape(l, -1)
        return x


class TangentSpace(BaseEstimator, TransformerMixin):
    """
    Tangent space projection
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.reference_ = None
        self.metric_map = None
        self.metric_mean = None
        self.metric = metric
        self.tsupdate = tsupdate

    def fit(self, x, y=None, sample_weight=None):
        """Fit (estimates) the reference point.
        Parameters
        ----------
        x : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.
        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)

        self.reference_ = mean_covariance(
            x,
            metric=self.metric_mean,
            sample_weight=sample_weight
        )
        return self

    @staticmethod
    def _check_metric(metric):

        if isinstance(metric, str):
            metric_mean = metric
            metric_map = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'map']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "map"')

            metric_mean = metric['mean']
            metric_map = metric['map']

        else:
            raise TypeError('metric must be dict or str')

        return metric_mean, metric_map

    @staticmethod
    def _check_data_dim(self, X):
        """Check data shape and return the size of SPD matrix."""
        shape_X = X.shape
        if len(X.shape) == 2:
            n_channels = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if n_channels != int(n_channels):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(n_channels)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)
        self._check_reference_points(X)

        if self.tsupdate:
            Cr = mean_covariance(X, metric=self.metric_mean)
        else:
            Cr = self.reference_
        return tangent_space(X, Cr, metric=self.metric_map)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.
        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)

        self.reference_ = mean_covariance(
            X,
            metric=self.metric_mean,
            sample_weight=sample_weight
        )
        return tangent_space(X, self.reference_, metric=self.metric_map)

    def inverse_transform(self, X, y=None):
        """Inverse transform.
        Project back a set of tangent space vector in the manifold.
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_ts)
            Set of tangent space projections of the matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        Returns
        -------
        cov : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices corresponding to each of tangent vector.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)
        self._check_reference_points(X)
        return untangent_space(X, self.reference_, metric=self.metric_map)


class ProjCommonSpaceCV(BaseEstimator, TransformerMixin):
    def __init__(self, scale='auto', n_compo=10, reg=1e-7):
        self.patterns_ = None
        self.filters_ = None
        self.scale_ = None
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, x, y=None):
        self.n_compo = len(x[0]) if self.n_compo == 'full' else self.n_compo
        self.scale_ = _get_scale(x, self.scale)
        c = x.mean(axis=0)
        eigvals, eigvecs = eigh(c)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs.T
        self.filters_ = evecs
        self.patterns_ = pinv(self.filters_).T
        return self

    def transform(self, x, n_compo=None, filters=None):
        if n_compo is None:
            n_compo = self.n_compo
        if filters is None:
            filters = self.filters_
        self.scale_ = _get_scale(x, self.scale)
        x = self.scale_ * x
        x_filt = filters[0:n_compo, :] @ x @ filters[0:n_compo, :].T
        x_filt += self.reg * np.eye(n_compo)
        return x_filt


class PCS(BaseEstimator, TransformerMixin):
    def __init__(self, filters=None, ncompo=5):
        self.ncompo = ncompo
        self.filters = filters

    def fit(self, x_train, y_train):
        return self

    def transform(self, x_test):
        pcv = ProjCommonSpaceCV(scale='auto', n_compo=10, reg=1e-7)
        x = np.stack(
            [TangentSpace().fit_transform(pcv.transform(x_test[:, n], n_compo=self.ncompo, filters=self.filters[n])) for
             n in range(x_test.shape[1])], axis=1)
        return x.reshape(len(x_test), -1)
