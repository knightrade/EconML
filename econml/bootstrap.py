# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bootstrap sampling."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from scipy.stats import norm
from collections import OrderedDict
import pandas as pd


class BootstrapInferenceResults:
    """
    Results class for bootstrap inference.

    Parameters
    ----------
    pred_dist : array-like, shape (b, m, d_y, d_t) or (b, m, d_y)
        the raw predictions of the metric using b times bootstrap.
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
    kind : 'percentile' or 'pivot'
        Whether to use percentile or pivot-based intervals
    d_t: int
        Number of treatments
    d_y: int
        Number of outputs
    inf_type: string
        The type of inference result.
        It could be either 'effect', 'coefficient' or 'intercept'.
    fname_transformer: None or predefined function
        The transform function to get the corresponding feature names from featurizer
    """

    def __init__(self, pred_dist, kind, d_y, d_t, inf_type, fname_transformer):
        self.pred_dist = pred_dist
        self.kind = kind
        self.d_t = d_t
        self.d_y = d_y
        self.inf_type = inf_type
        self.fname_transformer = fname_transformer

    @property
    def point_estimate(self):
        """
        Get the point estimate of each treatment on each outcome for each sample X[i].

        Returns
        -------
        prediction : array-like, shape (m, d_y, d_t) or (m, d_y)
            The point estimate of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return np.mean(self.pred_dist, axis=0)

    @property
    def stderr(self):
        """
        Get the standard error of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
            The standard error of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return np.std(self.pred_dist, axis=0)

    @property
    def var(self):
        """
        Get the variance of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        var : array-like, shape (m, d_y, d_t) or (m, d_y)
            The variance of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.stderr**2

    def conf_int(self, alpha=0.1):
        """
        Get the confidence interval of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple of arrays, shape (m, d_y, d_t) or (m, d_y)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        lower = alpha / 2
        upper = (1 - alpha) / 2
        if self.kind == 'percentile':
            return np.percentile(self.pred_dist, lower, axis=0), np.percentile(self.pred_dist, upper, axis=0)
        elif self.kind == 'pivot':
            est = self.point_estimate
            return (2 * est - np.percentile(self.pred_dist, upper, axis=0),
                    2 * est - np.percentile(self.pred_dist, lower, axis=0))
        else:
            raise ValueError("Unrecognized bootstrap kind; valid kinds are 'percentile' and 'pivot'")

    def pvalue(self, value=0):
        """
        Get the p value of the each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        pvalue : array-like, shape (m, d_y, d_t) or (m, d_y)
            The p value of of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """

        if self.kind == 'percentile':
            dist = self.pred_dist
        elif self.kind == 'pivot':
            est = np.mean(self.pred_dist, axis=0)
            dist = 2 * est - pred_dist
        else:
            raise ValueError("Unrecognized bootstrap kind; valid kinds are 'percentile' and 'pivot'")
        return min((dist < value).sum(), (dist > value).sum()) / dist.shape[0]

    def zstat(self, value=0):
        """
        Get the z statistic of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        zstat : array-like, shape (m, d_y, d_t) or (m, d_y)
            The z statistic of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return (self.point_estimate - value) / self.stderr

    def summary_frame(self, alpha=0.1, value=0, decimals=3, feat_name=None):
        """
        Output the dataframe for all the inferences above.

        Parameters
        ----------
        alpha: optional float in [0, 1] (default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.
        feat_name: optional list of strings or None (default is None)
            The input of the feature names

        Returns
        -------
        output: pandas dataframe
            The output dataframe includes point estimate, standard error, z score, p value and confidence intervals
            of the estimated metric of each treatment on each outcome for each sample X[i]
        """
        ci_mean = self.conf_int(alpha=alpha)
        to_include = OrderedDict()
        to_include['point_estimate'] = self._array_to_frame(self.d_t, self.d_y, self.point_estimate)
        to_include['stderr'] = self._array_to_frame(self.d_t, self.d_y, self.stderr)
        to_include['zstat'] = self._array_to_frame(self.d_t, self.d_y, self.zstat(value))
        to_include['pvalue'] = self._array_to_frame(self.d_t, self.d_y, self.pvalue(value))
        to_include['ci_lower'] = self._array_to_frame(self.d_t, self.d_y, ci_mean[0])
        to_include['ci_upper'] = self._array_to_frame(self.d_t, self.d_y, ci_mean[1])
        res = pd.concat(to_include, axis=1, keys=to_include.keys()).round(decimals)
        if self.d_t == 1:
            res.columns = res.columns.droplevel(1)
        if self.d_y == 1:
            res.index = res.index.droplevel(1)
        if self.inf_type == 'coefficient':
            if feat_name is not None and self.fname_transformer:
                ind = self.fname_transformer(feat_name)
            else:
                ct = res.shape[0] // self.d_y
                ind = ['X' + str(i) for i in range(ct)]

            if self.d_y > 1:
                res.index = res.index.set_levels(ind, level=0)
            else:
                res.index = ind
        elif self.inf_type == 'intercept':
            if self.d_y > 1:
                res.index = res.index.set_levels(['intercept'], level=0)
            else:
                res.index = ['intercept']
        return res

    def population_summary(self, alpha=0.1, value=0, decimals=3, tol=0.001):
        """
        Output the object of population summary results.

        Parameters
        ----------
        alpha: optional float in [0, 1] (default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.
        tol:  optinal float (default=0.001)
            The stopping criterion. The iterations will stop when the outcome is less than ``tol``

        Returns
        -------
        PopulationSummaryResults: object
            The population summary results instance contains the different summary analysis of point estimate
            for sample X on each treatment and outcome.
        """
        if self.inf_type == 'effect':
            return PopulationSummaryResults(pred=self.point_estimate, pred_stderr=self.stderr,
                                            d_t=self.d_t, d_y=self.d_y,
                                            alpha=alpha, value=value, decimals=decimals, tol=tol)
        else:
            raise AttributeError(self.inf_type + " inference doesn't support population_summary function!")

    def _array_to_frame(self, d_t, d_y, arr):
        if np.isscalar(arr):
            arr = np.array([arr])
        if self.inf_type == 'coefficient':
            arr = np.moveaxis(arr, -1, 0)
        arr = arr.reshape((-1, d_y, d_t))
        df = pd.concat([pd.DataFrame(x) for x in arr], keys=np.arange(arr.shape[0]))
        df.index = df.index.set_levels(['Y' + str(i) for i in range(d_y)], level=1)
        df.columns = ['T' + str(i) for i in range(d_t)]
        return df


class BootstrapEstimator:
    """Estimator that uses bootstrap sampling to wrap an existing estimator.

    This estimator provides a `fit` method with the same signature as the wrapped estimator.

    The bootstrap estimator will also wrap all other methods and attributes of the wrapped estimator,
    but return the average of the sampled calculations (this will fail for non-numeric outputs).

    It will also provide a wrapper method suffixed with `_interval` for each method or attribute of
    the wrapped estimator that takes two additional optional keyword arguments `lower` and `upper` specifiying
    the percentiles of the interval, and which uses `np.percentile` to return the corresponding lower
    and upper bounds based on the sampled calculations.  For example, if the underlying estimator supports
    an `effect` method with signature `(X,T) -> Y`, this class will provide a method `effect_interval`
    with pseudo-signature `(lower=5, upper=95, X, T) -> (Y, Y)` (where `lower` and `upper` cannot be
    supplied as positional arguments).

    Parameters
    ----------
    wrapped : object
        The basis for the clones used for estimation.
        This object must support a `fit` method which takes numpy arrays with consistent first dimensions
        as arguments.

    n_bootstrap_samples : int
        How many draws to perform.

    n_jobs: int, default: None
        The maximum number of concurrently running jobs, as in joblib.Parallel.

    compute_means : bool, default: True
        Whether to pass calls through to the underlying collection and return the mean.  Setting this
        to ``False`` can avoid ambiguities if the wrapped object itself has method names with an `_interval` suffix.

    prefer_wrapped: bool, default: False
        In case a method ending in '_interval' exists on the wrapped object, whether
        that should be preferred (meaning this wrapper will compute the mean of it).
        This option only affects behavior if `compute_means` is set to ``True``.

    bootstrap_type: 'percentile', 'pivot', or 'normal', default 'percentile'
        Bootstrap method used to compute results.  'percentile' will result in using the empiracal CDF of
        the replicated copmutations of the statistics.   'pivot' will also use the replicates but create a pivot
        interval that also relies on the estimate over the entire dataset.  'normal' will instead compute an interval
        assuming the replicates are normally distributed.
    """

    def __init__(self, wrapped, n_bootstrap_samples=1000, n_jobs=None, compute_means=True, prefer_wrapped=False,
                 bootstrap_type='percentile'):
        self._instances = [clone(wrapped, safe=False) for _ in range(n_bootstrap_samples)]
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs
        self._compute_means = compute_means
        self._prefer_wrapped = prefer_wrapped
        self._bootstrap_type = bootstrap_type
        self._wrapped = wrapped

    # TODO: Add a __dir__ implementation?

    @staticmethod
    def __stratified_indices(arr):
        assert 1 <= np.ndim(arr) <= 2
        unique = np.unique(arr, axis=0)
        indices = []
        for el in unique:
            ind, = np.where(np.all(arr == el, axis=1) if np.ndim(arr) == 2 else arr == el)
            indices.append(ind)
        return indices

    def fit(self, *args, **named_args):
        """
        Fit the model.

        The full signature of this method is the same as that of the wrapped object's `fit` method.
        """
        from .cate_estimator import BaseCateEstimator  # need to nest this here to avoid circular import

        index_chunks = None
        if isinstance(self._instances[0], BaseCateEstimator):
            index_chunks = self._instances[0]._strata(*args, **named_args)
            if index_chunks is not None:
                index_chunks = self.__stratified_indices(index_chunks)
        if index_chunks is None:
            n_samples = np.shape(args[0] if args else named_args[(*named_args,)[0]])[0]
            index_chunks = [np.arange(n_samples)]  # one chunk with all indices

        indices = []
        for chunk in index_chunks:
            n_samples = len(chunk)
            indices.append(chunk[np.random.choice(n_samples,
                                                  size=(self._n_bootstrap_samples, n_samples),
                                                  replace=True)])

        indices = np.hstack(indices)

        def fit(x, *args, **kwargs):
            x.fit(*args, **kwargs)
            return x  # Explicitly return x in case fit fails to return its target

        def convertArg(arg, inds):
            return np.asarray(arg)[inds] if arg is not None else None

        self._instances = Parallel(n_jobs=self._n_jobs, prefer='threads', verbose=3)(
            delayed(fit)(obj,
                         *[convertArg(arg, inds) for arg in args],
                         **{arg: convertArg(named_args[arg], inds) for arg in named_args})
            for obj, inds in zip(self._instances, indices)
        )
        return self

    def __getattr__(self, name):
        """
        Get proxy attribute that wraps the corresponding attribute with the same name from the wrapped object.

        Additionally, the suffix "_interval" is supported for getting an interval instead of a point estimate.
        """

        # don't proxy special methods
        if name.startswith('__'):
            raise AttributeError(name)

        def proxy(make_call, name, summary):
            def summarize_with(f):
                return summary(np.array(Parallel(n_jobs=self._n_jobs, prefer='threads', verbose=3)(
                    (f, (obj, name), {}) for obj in self._instances)), f(self._wrapped, name))
            if make_call:
                def call(*args, **kwargs):
                    return summarize_with(lambda obj, name: getattr(obj, name)(*args, **kwargs))
                return call
            else:
                return summarize_with(lambda obj, name: getattr(obj, name))

        def get_mean():
            # for attributes that exist on the wrapped object, just compute the mean of the wrapped calls
            return proxy(callable(getattr(self._instances[0], name)), name, lambda arr, _: np.mean(arr, axis=0))

        def get_std():
            prefix = name[: - len('_std')]
            return proxy(callable(getattr(self._instances[0], prefix)), prefix,
                         lambda arr, _: np.std(arr, axis=0))

        def get_interval():
            # if the attribute exists on the wrapped object once we remove the suffix,
            # then we should be computing a confidence interval for the wrapped calls
            prefix = name[: - len("_interval")]

            def call_with_bounds(can_call, lower, upper):
                def percentile_bootstrap(arr, _):
                    return np.percentile(arr, lower, axis=0), np.percentile(arr, upper, axis=0)

                def pivot_bootstrap(arr, _):
                    # TODO: do we want the central estimate to be the average of all bootstrap estimates,
                    #       or the original estimate over the entire non-bootstrapped population?
                    est = np.mean(arr, axis=0)
                    return 2 * est - np.percentile(arr, upper, axis=0), 2 * est - np.percentile(arr, lower, axis=0)

                def normal_bootstrap(arr, _):
                    est = np.mean(arr, axis=0)
                    std = np.std(arr, axis=0)
                    return est - norm.ppf(upper / 100) * std, est - norm.ppf(lower / 100) * std

                # TODO: studentized bootstrap? this would be more accurate in most cases but can we avoid
                #       second level bootstrap which would be prohibitive computationally?

                fn = {'percentile': percentile_bootstrap,
                      'normal': normal_bootstrap,
                      'pivot': pivot_bootstrap}[self._bootstrap_type]
                return proxy(can_call, prefix, fn)

            can_call = callable(getattr(self._instances[0], prefix))
            if can_call:
                # collect extra arguments and pass them through, if the wrapped attribute was callable
                def call(*args, lower=5, upper=95, **kwargs):
                    return call_with_bounds(can_call, lower, upper)(*args, **kwargs)
                return call
            else:
                # don't pass extra arguments if the wrapped attribute wasn't callable to begin with
                def call(lower=5, upper=95):
                    return call_with_bounds(can_call, lower, upper)
                return call

        def get_inference():
            # can't import from econml.inference at top level without creating mutual dependencies
            from .inference import InferenceResults

            prefix = name[: - len("_inference")]
            if prefix in ['const_marginal_effect', 'effect']:
                inf_type = 'effect'
            elif prefix == 'coef_':
                inf_type = 'coefficient'
            elif prefix == 'intercept_':
                inf_type = 'intercept'
            else:
                raise AttributeError("Unsupported inference: " + name)

            d_t = self._wrapped._d_t[0] if self._wrapped._d_t else 1
            d_t = 1 if prefix == 'effect' else d_t
            d_y = self._wrapped._d_y[0] if self._wrapped._d_y else 1

            def get_inference_nonparametric(kind):
                return proxy(callable(getattr(self._instances[0], prefix)), prefix,
                             lambda arr, _: BootstrapInferenceResults(pred_dist=arr, kind=kind,
                                                                      d_t=d_t, d_y=d_y, inf_type=inf_type,
                                                                      fname_transformer=None))

            def get_inference_parametric():
                pred = getattr(self._wrapped, prefix)
                stderr = getattr(self, prefix + '_std')
                return InferenceResults(d_t=d_t, d_y=d_y, pred=pred,
                                        pred_stderr=stderr, inf_type=inf_type,
                                        pred_dist=None, fname_transformer=None)

            return {'normal': get_inference_parametric,
                    'percentile': lambda: get_inference_nonparametric('percentile'),
                    'pivot': lambda: get_inference_nonparametric('pivot')}[self._bootstrap_type]

        caught = None
        m = None
        if name.endswith("_interval"):
            m = get_interval
        elif name.endswith("_std"):
            m = get_std
        elif name.endswith("_inference"):
            m = get_inference
        if self._compute_means and self._prefer_wrapped:
            try:
                return get_mean()
            except AttributeError as err:
                caught = err
            if m is not None:
                m()
        else:
            # try to get interval/std first if appropriate,
            # since we don't prefer a wrapped method with this name
            if m is not None:
                try:
                    return m()
                except AttributeError as err:
                    caught = err
            if self._compute_means:
                return get_mean()

        raise (caught if caught else AttributeError(name))
