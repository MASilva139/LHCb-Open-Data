#################################################################
#           Modelos de ajuste y estimación de fondo             #
#################################################################
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit

from .config import (
    MASS_MIN,
    MASS_MAX,
    MASS_CENTER,
    N_BINS,
    BIN_WIDTH
)

######################### Utilidades internas #########################
def _safe_norm(value: float) -> float:
    return value if abs(value) > 1e-30 else 1e-30

def _prepare_hist(masses_array):
    masses = np.asarray(masses_array, dtype = float)
    masses = masses[np.isfinite(masses) & (masses >= MASS_MIN) & (masses <= MASS_MAX)]
    counts, edges = np.histogram(masses, bins=N_BINS, range=(MASS_MIN, MASS_MAX))
    centers = (edges[:-1] + edges[1:])/2.0
    uncert = np.sqrt(np.maximum(counts, 1.0))
    N_total = counts.sum()
    if N_total == 0:
        raise ValueError('Sin eventos en el intervalo')
    return masses, counts, edges, centers, uncert, N_total

############################### Modelos ###############################
# Modelo exponencial
def exponential_pdf(x, slope):
    lower = MASS_MIN - MASS_CENTER
    upper = MASS_MAX - MASS_CENTER
    if abs(slope) < 1e-10:
        return np.ones_like(x)/(MASS_MAX - MASS_MIN)
    norm = (np.exp(slope * upper) - np.exp(slope * lower))/slope
    return np.exp(slope*(x - MASS_CENTER))/_safe_norm(norm)

# Modelo de Chebyshev
def chebyshev2_pdf(x, c0, c1, c2):
    t = 2 * (x - MASS_MIN)/(MASS_MAX - MASS_MIN) - 1
    pdf = c0 + c1*t + c2*(2*t**2 - 1)
    pdf = np.clip(pdf, 0, None)         # Asegurar positividad
    norm = np.trapezoid(pdf, x)
    return pdf/_safe_norm(norm)

# Función gaussiana
def gauss_pdf(x, mu, sigma):
    return stats.norm.pdf(x, loc=mu, scale=sigma)

# Modelo de Crystal Ball
def crystalball_pdf(x, beta, m, mu, sigma):
    norm = (
        stats.crystalball.cdf(MASS_MAX, beta, m, loc=mu, scale=sigma)
        - stats.crystalball.cdf(MASS_MIN, beta, m, loc=mu, scale=sigma)
    )
    return (stats.crystalball.pdf(x, beta, m, loc=mu, scale=sigma)/_safe_norm(norm))

# Modelo lineal
def linear_pdf(x, slope):
    scaled_x = (x - MASS_CENTER)/(0.5*(MASS_MAX - MASS_MIN))
    pdf = 1.0 + slope *scaled_x
    pdf = np.clip(pdf, 0, None)
    norm = np.trapezoid(pdf, x)
    return pdf/_safe_norm(norm)

# Modelo sigmoide
def sigmoid_shape(x, amplitude, midpoint, width):
    return amplitude*expit(-(x - midpoint)/width)

########################## Modelos combinados ##########################
# Gaussiana + fondo exponencial
def model_gauss_exp(x, n_sig, mu, sigma, n_bg, slope):
    return BIN_WIDTH * (
        n_sig*gauss_pdf(x, mu, sigma) + n_bg*exponential_pdf(x, slope)
    )

# Gaussiana + fondo chebyshev
def model_gauss_cheb(x, n_sig, mu, sigma, n_bg, c0, c1, c2):
    return BIN_WIDTH*(
        n_sig*gauss_pdf(x, mu, sigma) + n_bg*chebyshev2_pdf(x, c0, c1, c2)
    )

# Cristal Ball +  fondo lineal
def model_cb_linear(x, n_sig, beta_cb, m_cb, mu, sigma, n_bg, slope):
    signal_pdf = crystalball_pdf(x, beta_cb, m_cb, mu, sigma)
    background_pdf = linear_pdf(x, slope)
    return BIN_WIDTH*(n_sig*signal_pdf + n_bg*background_pdf)

# Gaussiana + lineal +  sigmoide
def gaussian_counts(x, amplitude, mu, sigma):
    return amplitude*np.exp(-0.5*((x - mu)/sigma)**2)

def linear_counts(x, alpha, beta):
    return alpha*(x - MASS_CENTER) + beta

def model_gauss_linear_sigmoid(x, amp_sig, mu, sigma, alpha, beta, amp_sigmoid, m, width):
    return (
        gaussian_counts(x, amp_sig, mu, sigma) 
        + linear_counts(x, alpha, beta) 
        + sigmoid_shape(x, amp_sigmoid, m, width)
    )

####################### Configuración de modelos #######################
def _get_model_configuration(
        model: str,
        counts: np.ndarray,
        N_total: int
) -> dict:
    max_count = max(float(counts.max()), 1.0)
    median_count = max(float(np.median(counts)), 1.0)
    if model == 'gauss_exp':
        return {
            'function': model_gauss_exp,
            'p0': [0.65*N_total, 5279.0, 18.0, 0.35*N_total, -0.002],
            'lower': [0.0, 5240.0, 3.0,  0.0,   -0.02],
            'upper': [2.0*N_total, 5320.0, 60.0, 2.0*N_total, 0.02],
            'signal_mode': 'yield',
            'signal_index': 0,
            'parameter_names': ['n_sig', 'mu', 'sigma', 'n_bg', 'slope']
        }
    if model == 'gauss_cheb':
        return {
            'function': model_gauss_cheb,
            'p0': [0.50*N_total, 5279.0, 18.0, 0.50*N_total, 1.0, 0.0, 0.0],
            'lower': [0, 5240, 3.0,  0.0,  0.0, -2.0, -2.0],
            'upper': [2.0*N_total, 5320.0, 60.0, 2.0*N_total, 1e6, 2.0, 2.0],
            'signal_mode': 'yield',
            'signal_index': 0,
            'parameter_names': ['n_sig', 'mu', 'sigma', 'n_bg', 'c0', 'c1', 'c2']
        }
    if model == 'cb_linear':
        return {
            'function': model_cb_linear,
            'p0': [0.65*N_total, 2.0, 3.0, 5279.0, 18.0, 0.35*N_total, 0.0],
            'lower': [0.0, 0.3, 1.01, 5240.0, 3.0, 0.0, -0.95],
            'upper': [2.0*N_total, 15.0, 30.0, 5320.0, 60.0, 2.0**N_total, 0.95],
            'signal_mode': 'yield',
            'signal_index': 0,
            'parameter_names': ['n_sig', 'beta_cb', 'm_cb', 'mu', 'sigma', 'n_bg', 'slope']
        }
    if model == 'gauss_linear_sigmoid':
        return {
            'function': model_gauss_linear_sigmoid,
            'p0': [max_count, 5280.0, 18.0, -0.05, median_count, 0.20*max_count, 5170.0, -15.0],
            'lower': [0.0, 5240.0, 3.0, -10.0, -1e5, -1e5, 5050.0, -150.0],
            'upper': [10.0*max_count, 5320.0, 60.0, 10.0, 1e5, 1e5, 5600.0, -1.0],
            'signal_mode': 'amplitude',
            'signal_index': 0,
            'parameter_names': ['amp_sig', 'mu', 'sigma', 'alpha', 'beta', 'amp_sigmoid', 'm', 'width']
        }
    valid_models = [
        'gauss_exp',
        'gauss_cheb',
        'cb_linear',
        'gauss_linear_sigmoid'
    ]
    raise ValueError(
        'Modelo no reconocido. '
        f"Modelos válidos: {', '.join(valid_models)}"
    )

def _extract_signa_yield(popt: np.ndarray, pcov: np.ndarray, config: dict) -> tuple[float, float]:
    signal_mode = config['signal_mode']
    if signal_mode == 'yield':
        idx = config['signal_index']
        n_signal = popt[idx]
        n_signal_error = np.sqrt(np.clip(pcov[idx, idx], 0, None))
        return n_signal, n_signal_error
    if signal_mode == 'amplitude':
        amp = popt[0]
        sigma = popt[2]
        n_signal = (amp*sigma*np.sqrt(2.0*np.pi)/BIN_WIDTH)
        amp_error = np.sqrt(np.clip(pcov[0, 0], 0, None))
        sigma_error = np.sqrt(np.clip(pcov[2, 2], 0, None))
        if amp <= 0 or sigma <= 0:
            return n_signal, np.nan
        n_signal_error = abs(n_signal)*np.sqrt(
            (amp_error/amp)**2 + (sigma_error/sigma)**2
        )
        return n_signal, n_signal_error
    raise ValueError(f'signal_mode no soportado: {signal_mode}')

########################## Función de ajuste ###########################
def fit_mass(
    masses_array, 
    model='gauss_exp', 
    verbose: bool = True
) -> dict:
    (masses, counts, edges, centers, uncert, N_total) = _prepare_hist(masses_array) 
    config = _get_model_configuration(model, counts, N_total)
    fit_function = config['function']
    try:
        popt, pcov = curve_fit(
            fit_function, 
            centers, 
            counts,
            p0=config['p0'], 
            sigma=uncert, 
            absolute_sigma=True,
            bounds=(config['lower'], config['upper']), 
            maxfev=300000
        )
        converged = True
        warning = None
    except (RuntimeError, ValueError) as error:
        print(f'  [WARNING] Ajuste no convergió: {error}')
        popt = np.asarray(config['p0'], dtype=float)
        pcov = np.diag(np.ones(len(config['p0'])))
        converged = False
        warning = str(error)
    expected = fit_function(centers, *popt)
    chi2 = np.sum(((counts - expected) / uncert)**2)
    ndf = len(counts) - len(popt)
    n_signal, n_signal_error = _extract_signa_yield(popt, pcov, config)
    parameter_values = dict(zip(config['parameter_names'], popt))
    if verbose:
        print(f'Modelo  = {model}')
        print(f'N_señal = {n_signal:.0f} ± {n_signal_error:.0f}')
        print(f'χ²/ndf  = {chi2:.1f} / {ndf}')
        if 'mu' in parameter_values:
            print(f"Media   = {parameter_values['mu']:.2f} [MeV/c²]")
        if 'sigma' in parameter_values:
            print(f"  σ     = {parameter_values['sigma']:.2f} [MeV/c²]")
    return {
        'model_name': model,
        'model_fn': fit_function,
        'parameter_names': config['parameter_names'],
        'parameters': parameter_values,
        'popt': popt, 
        'pcov': pcov,
        'n_signal': n_signal, 
        'n_signal_error': n_signal_error,
        'n_bg': parameter_values.get('n_bg', np.nan), 
        'chi2': chi2, 
        'ndf': ndf,
        'counts': counts, 
        'centers': centers, 
        'edges': edges,
        'uncertainties': uncert,
        'expected': expected,
        'converged': converged,
        'warning': warning
    }

def evaluate_fit_components(x, fit_result: dict) -> dict:
    model = fit_result['model_name']
    popt = fit_result['popt']
    if model == 'gauss_exp':
        n_sig, mu, sigma, n_bg, slope = popt
        signal = BIN_WIDTH*n_sig*gauss_pdf(x, mu, sigma)
        background = BIN_WIDTH*n_bg*exponential_pdf(x, slope)
    elif model == 'gauss_cheb':
        n_sig, mu, sigma, n_bg, c0, c1, c2 = popt
        signal = BIN_WIDTH*n_sig*gauss_pdf(x, mu, sigma)
        background = BIN_WIDTH*n_bg*chebyshev2_pdf(x, c0, c1, c2)
    elif model == 'cb_linear':
        (n_sig, beta_cb, m_cb, mu, sigma, n_bg, slope) = popt
        signal = BIN_WIDTH*n_sig*crystalball_pdf(x, beta_cb, m_cb, mu, sigma)
        background = BIN_WIDTH*n_bg*linear_pdf(x, slope)
    elif model == 'gauss_linear_sigmoid':
        (amp_sig, mu, sigma, alpha, beta, amp_sigmoid, m, width) = popt
        signal = gaussian_counts(x, amp_sig, mu, sigma)
        background = (
            linear_counts(x, alpha, beta) + sigmoid_shape(x, amp_sigmoid, m, width)
        )
    else:
        raise ValueError(f'Modelo no reconocido: {model}')
    total = fit_result['model_fn'](x, *popt)
    return {
        'total': total,
        'signal': signal,
        'background': background
    }

# ── Sideband subtraction (para canales con fondo alto) ──────
def sideband_background_estimate(
    df: pd.DataFrame, 
    mass_col: str = 'B_M',
    signal_window: tuple = (5228, 5330),
    left_band: tuple = (5100, 5200),
    right_band: tuple = (5400, 5500),
    verbose: bool = True
) -> dict:
    n_sig_region = len(df.query(f'{signal_window[0]} <= {mass_col} < {signal_window[1]}'))
    n_left = len(df.query(f'{left_band[0]} <= {mass_col} < {left_band[1]}'))
    n_right = len(df.query(f'{right_band[0]} <= {mass_col} < {right_band[1]}'))
    # Factor de escala: relación de anchos
    w_sig = signal_window[1] - signal_window[0]
    w_bands = (left_band[1] - left_band[0] + right_band[1] - right_band[0]) / 2.0
    scale = w_sig/w_bands
    n_bg_est = ((n_left + n_right)/2)*scale
    n_sig_est = (n_sig_region - n_bg_est)
    sob = (n_sig_est/np.sqrt(n_bg_est + 1e-9))
    print(f'  Región señal     : {n_sig_region:,} eventos')
    print(f'  Banda izquierda  : {n_left:,} eventos')
    print(f'  Banda derecha    : {n_right:,} eventos')
    print(f'  Fondo estimado   : {n_bg_est:.1f} eventos')
    print(f'  Señal estimada   : {n_sig_est:.1f} eventos')
    print(f'  S/√B             : {sob:.2f}')
    return {
        'n_total': n_sig_region,
        'n_bg_estimate': n_bg_est,
        'n_signal_estimate': n_sig_est,
        'SoverSqrtB': sob,
        'n_left': n_left,
        'n_right': n_right,
        'scale': scale
    }

def compare_mass_models(
    masses_array,
    models: list[str] | tuple[str, ...] | None = None, # U+007C: |, U+2223: ∣
    verbose: bool = True
) -> dict:
    if models is None:
        models = [
            'gauss_exp',
            'gauss_cheb',
            'cb_linear',
            'gauss_linear_sigmoid'
        ]
    results = {}
    for model in models:
        if verbose:
            print(f'\n--- Ajustando modelo: {model} ---')
        try:
            result = fit_mass(masses_array, model= model, verbose=verbose)
            result['status'] = 'ok'
            results[model] = result
        except Exception as e:
            print(f'[ERROR] Falló el modelo {model}: {e}')
            results[model] = {
                'model': model,
                'status': 'error',
                'error': str(e),
                'n_signal': np.nan,
                'n_signal_error': np.nan,
                'chi2': np.nan,
                'ndf': np.nan,
                'converged': False
            }
    return results

def sumarize_model_comparison(comparison_results: dict) -> pd.DataFrame:
    rows = []
    for model, result in comparison_results.items():
        chi2 = result.get('chi2', np.nan)
        ndf = result.get('chi2', np.nan)
        if (np.isfinite(chi2) and np.isfinite(ndf) and ndf != 0):
            chi2_ndf = chi2/ndf
        else:
            chi2_ndf = np.nan
        rows.append({
            'modelo': model,
            'estado': result.get('status', 'ok'),
            'convergio': result.get('converged', False),
            'n_signal': result.get('n_signal', np.nan),
            'n_signal_error': result.get('n_signal_error', np.nan),
            'chi2': chi2,
            'ndf': ndf,
            'chi2_ndf': chi2_ndf
        })
    return pd.DataFrame(rows)

def compare_acp_model(comparison_plus: dict, comparison_minus: dict) -> pd.DataFrame:
    rows = []
    common_models = (set(comparison_plus.keys()) & set(comparison_minus.keys()))
    for model in common_models:
        result_plus = comparison_plus[model]
        result_minus = comparison_minus[model]
        n_plus = result_plus.get('n_signal', np.nan)
        n_minus = result_minus.get('n_signal', np.nan)
        if (np.isfinite(n_plus) and np.isfinite(n_minus) and (n_plus + n_minus) > 0):
            acp = (n_minus - n_plus)/(n_minus + n_plus)
            sigma_acp = np.sqrt((1 - acp**2)/(n_minus + n_plus))
            significance = (acp/sigma_acp if sigma_acp > 0 else np.nan)
        else:
            acp = np.nan
            sigma_acp = np.nan
            significance = np.nan
        rows.append({
            'modelo': model,
            'N_plus': n_plus,
            'N_minus': n_minus,
            'A_CP': acp,
            'sigma_A_CP': sigma_acp,
            'significancia': significance
        })
    return pd.DataFrame(rows)