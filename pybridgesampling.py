from scipy.linalg import lstsq
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.ar_model import ar_select_order
from scipy.stats import multivariate_normal as mvnorm
from scipy.special import logsumexp

import numpy as np
import pymc as pm
import arviz as az

def iterative_scheme(q11, q12, q21, q22, r0, neff, N1, N2, tol, maxiter, criterion):
    '''
    Iterative scheme as proposed by Meng and Wong (1996) to estimate marginal
    likelihood. Code copied from
    https://gist.github.com/junpenglao/4d2669d69ddfe1d788318264cdcf0583
    '''
    l1 = q11 - q12
    l2 = q21 - q22
    lstar = np.median(l1) # To increase numerical stability,
                          # subtracting the median of l1 from l1 & l2 later
    s1 = neff/(neff + N2)
    s2 = N2/(neff + N2)
    r = r0
    r_vals = [r]
    logml = np.log(r) + lstar
    criterion_val = 1 + tol

    i = 0
    while (i <= maxiter) & (criterion_val > tol):
        rold = r
        logmlold = logml
        log_numi = (l2 - lstar) - np.logaddexp(np.log(s1) + l2 - lstar, np.log(s2) + np.log(r))
        numi = np.exp(log_numi)
        log_deni = -np.logaddexp(np.log(s1) + l1 - lstar, np.log(s2) + np.log(r))
        deni = np.exp(log_deni)
        if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
            warnings.warn("""Infinite value in iterative scheme, returning NaN.
            Try rerunning with more samples.""")
        r = (N1/N2) * np.exp(logsumexp(log_numi) - logsumexp(log_deni))
        r_vals.append(r)
        logml = np.log(r) + lstar
        i += 1
        if criterion=='r':
            criterion_val = np.abs((r - rold)/r)
        elif criterion=='logml':
            criterion_val = np.abs((logml - logmlold)/logml)

    if i >= maxiter:
        return dict(logml = np.NaN, niter = i, r_vals = np.asarray(r_vals))
    else:
        return dict(logml = logml, niter = i)


class Reshaper:

    def __init__(self, model, trace):
        '''
        On instantiation, we extract the variable names that will be used for bridge
        sampling and verify that they are present in the trace. Namely, we want the
        versions of the variables that have been transformed to have range [-inf, inf]
        rather than the original scale when applicable, and we don't want to include
        deterministics or observed.
        '''
        self.varnames = [model.rvs_to_values[v].name for v in model.free_RVs]
        for v in self.varnames:
            try: # making sure that everything you need is in the trace
                assert(v in list(trace.posterior.data_vars.keys()))
            except:
                raise Exception('%s missing from trace. Did you set `idata_kwargs = dict(include_transformed = True)` in `pm.sample`?'%v)

    def to_array(self, trace):
        '''
        Aggregates samples from posterior trace into one big array
        where samples have been flattened. This will help us fit
        one, MVNormal proposal distribution for bridge sampling.

        Parameters
        ----------
        trace : arviz.InferenceData

        Returns
        ---------
        posterior : np.array of shape (chains, draws, variables)
        '''
        X = []
        mapping = dict()
        for v in self.varnames:
            x = trace.posterior[v].values
            if len(x.shape) > 2:
                mapping[v] = dict(orig_shape = x.shape[2:])
                x = x.reshape(*x.shape[:2], -1) # flatten all but chain/draw dims
            else:
                mapping[v] = dict(orig_shape = ())
                assert(len(x.shape) == 2)
                x = x[..., np.newaxis] # now everything is 3D
            if len(X) != 0:
                idx_start = X.shape[-1]
                X = np.concatenate([X, x], axis = -1)
            else:
                idx_start = 0
                X = x
            idx_end = X.shape[-1]
            mapping[v]['idxs'] = (idx_start, idx_end)
        self.inverse_mapping = mapping
        return X

    def extract_variable(self, sample, varname):
        '''
        extracts a single variable from a single flattened sample
        '''
        idxs = self.inverse_mapping[varname]['idxs']
        x = sample[idxs[0]:idxs[1]]
        orig_shape = self.inverse_mapping[varname]['orig_shape']
        if orig_shape:
            return x.reshape(*self.inverse_mapping[varname]['orig_shape'])
        else:
            return x[0]

    def extract_variables(self, sample):
        '''
        extracts all variables from a single sample into dictionary format
        expected by pymc model's log-likelihood
        '''
        return {v: self.extract_variable(sample, v) for v in self.varnames}

def spectrum0_ar(x):
    '''
    Port of spectrum0.ar from R's coda::spectrum0.ar,
    written by J. Lao in [3].
    '''
    z = np.arange(1, len(x)+1)
    z = z[:, np.newaxis]**[0, 1]
    p, res, rnk, s = lstsq(z, x)
    residuals = x - np.matmul(z, p)

    if residuals.std() == 0:
        spec = order = 0
    else:
        order = ar_select_order(x, maxlag = 15, ic = 'aic', trend = 'c')
        ar_out = AR(x, lags = order.ar_lags).fit()
        spec = np.var(ar_out.resid)/(1 - np.sum(ar_out.params[1:]))**2

    return spec, order

def error_measures(logml):
    """
    Port of the error_measures.R in bridgesampling
    https://github.com/quentingronau/bridgesampling/blob/master/R/error_measures.R

    Returns
    -------
    rel_mse : float
        Expected relative-mean squared error
    coef_of_variation : float
        Coefficient of variation (i.e., the ratio of the standard deviation
        and the mean [1]) under assumption that birdge sampling estimator is an
        unbiased estimator of the marginal likelihood.

    References
    -----------
    [1] Brown, C. E. (2012). Applied multivariate statistics in geohydrology
        and related sciences. Springer Science & Business Media.
    """
    ml = np.exp(logml['logml'])
    g_p = np.exp(logml['q12'])
    g_g = np.exp(logml['q22'])
    priorTimesLik_p = np.exp(logml['q11'])
    priorTimesLik_g = np.exp(logml['q21'])
    p_p = priorTimesLik_p/ml
    p_g = priorTimesLik_g/ml

    N1 = len(p_p)
    N2 = len(g_g)
    s1 = N1/(N1 + N2)
    s2 = N2/(N1 + N2)

    f1 = p_g/(s1*p_g + s2*g_g)
    f2 = g_p/(s1*p_p + s2*g_p)
    rho_f2, _ = spectrum0_ar(f2)

    term1 = 1/N2 * np.var( f1 ) / np.mean( f1 )**2
    term2 = rho_f2/N1 * np.var( f2 ) / np.mean( f2 )**2

    re2 = term1 + term2

    # convert to coefficient of variation (assumes that bridge estimate is unbiased)
    cv = np.sqrt(re2)

    return re2, cv

def bridge_sample(model, trace, maxiter = 1000, r0 = 0.5, tol1 = 1e-10, tol2 = 1e-4, random_seed = None):
    '''
    Bridge sampling estimator of marginal likelihood using optimal bridge function
    and iterative scheme given by [1]. More digestible descriptions provided by [2].
    Code based on [3], updated for newer pymc versions/arviz compatibility and
    automatic handling of transforms and determinsitics.

    Parameters
    -----------
    model : pymc.Model
        Model object that was used for sampling of `trace`. It should be
        fine to just reinstantiate this from the same model definition,
        if you're loading a stored trace.
    trace : arivz.InferenceData
        `pymc.sample` must have been run with
        `idata_kwargs = dict(include_transformed = True)`
    maxiter : int
        Maximum number of iterations to reach convergence

    Returns
    ---------
    logml_dict : dict
        Key `logml` contains log of model's marginal likelihood.

    References
    -----------
    [1] Meng, X. L., & Wong, W. H. (1996). Simulating ratios of
        normalizing constants via a simple identity: a theoretical
        exploration. Statistica Sinica, 831-860.

    [2] Gronau, Q. F., Sarafoglou, A., Matzke, D., Ly, A.,
        Boehm, U., Marsman, M., ... & Steingroever, H. (2017).
        A tutorial on bridge sampling.
        Journal of mathematical psychology, 81, 80-97.

    [3] Lao, J. (2017), Marginal Likelihood in Python and PyMC3.
        https://junpenglao.xyz/Blogs/posts/2017-11-22-Marginal_likelihood_in_PyMC3.html

    '''
    reshaper = Reshaper(model, trace)
    X = reshaper.to_array(trace) # chains x draws x variables

    ## allocate half the draws to fitting proposal distribution
    n1_draws = X.shape[1] // 2
    X_fit = X[:, :n1_draws, :]
    n1_eff = np.median([az.ess(X_fit[..., i]) for i in range(X_fit.shape[-1])])
    X_fit = X_fit.reshape(-1, *X_fit.shape[2:]) # samples x variables
    n1 = X_fit.shape[0]

    ## then we'll use other half of draws for iterative scheme
    X_iter = X[:, n1_draws:, :]
    X_iter = X_iter.reshape(-1, *X_iter.shape[2:])
    n2 = X_iter.shape[0]

    # fit proposal distribution and  generate samples
    m, cov = mvnorm.fit(X_fit)
    prop_samps = mvnorm.rvs(m, cov, size = n2, random_state = random_seed)
    if len(prop_samps.shape) < 2:
        prop_samps = prop_samps[:, np.newaxis]

    # likelihood of posterior and proposal samples under proposal
    q12 = mvnorm.logpdf(X_iter, m, cov)
    q22 = mvnorm.logpdf(prop_samps, m, cov)
    # likelihood of posterior and proposal samples under un-normalized posterior
    logp = model.compile_logp()
    q11 = np.array([logp(reshaper.extract_variables(s)) for s in X_iter])
    q21 = np.array([logp(reshaper.extract_variables(s)) for s in prop_samps])

    # run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r0, n1_eff, n1, n2, tol1, maxiter, 'r')
    if ~np.isfinite(tmp['logml']):
        warnings.warn("""logml could not be estimated within maxiter, rerunning with
                      adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, n1_eff, n1, n2, tol2, maxiter, 'logml')
    logml = dict(
        logml = tmp['logml'],
        niter = tmp['niter'],
        method = 'normal',
        q11 = q11, q12 = q12, q21 = q21, q22 = q22
    )
    # add error measures before returning
    rel_mse, coef_of_var = error_measures(logml)
    logml['relative_mean_squared_error'] = rel_mse
    logml['coefficient_of_variation'] = coef_of_var
    return logml
