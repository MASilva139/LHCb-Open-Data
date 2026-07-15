#################################################################
#                     Análisis de canales                       #
#################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import (DATA_DIR, MASS_MIN, MASS_MAX)
from .io import load_magnet_data
from .physics import (
    calc_B_mass,
    calc_dalitz_vars,
    compute_acp,
    print_acp
)
from .fitting import (
    fit_mass, 
    sideband_background_estimate
)
from .plotting import (
    plot_mass_fit,
    plot_probability_distributions,
    plot_dalitz_sumary,
    plot_dalitz_scatter,
    PlotStyle
)

def run_channel_analysis(
    channel_name: str,
    preselection: str,
    masses: tuple,
    fit_model: str = 'gauss_exp',
    charm_veto: bool = True,
    use_sideband: bool = False,
    mass_window: tuple = (5194, 5364)
) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  CANAL: {channel_name}")
    print(f"{'='*60}")
    
    ############################ Carga de datos ############################
    print('\n[1] Cargando datos...')
    df = load_magnet_data(DATA_DIR, preselection)
    print(f'Total tras preselección: {len(df):,} eventos')

    #################### Distribución de Probabilidades ####################
    plot_probability_distributions(df = df, channel_name=channel_name)

    ################### Reconstrucción de masa invariante ##################
    print(r'\n[2] Reconstruyendo masa invariante del $B^{\pm}$...')
    df = calc_B_mass(df, masses)

    ############################ Ajuste de masa ############################
    print('\n[3] Ajuste de la distribución de masa...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{channel_name} - Masa invariante del B', fontsize = 17, fontweight = 'bold', color='white')
    results_fit = {}
    for charge, label, ax in zip([1, -1], [r'$B^{+}$', r'$B^{-}$'], axes):
        masses_arr = df.loc[df['B_Charge'] == charge, 'B_M'].values
        print(f"{label}: {len(masses_arr):,} candidatos en [{MASS_MIN}, {MASS_MAX}] MeV/c²")
        result = fit_mass(masses_arr, model=fit_model, verbose=True)
        results_fit[charge] = result
        plot_mass_fit(ax, result, label)
        safe_name = (
            channel_name
            .replace('→', 'to')     # U+2192: →
            .replace(' ', '_')
            .replace('±', "pm")        # U+00B1: ±
            .replace('+', 'plus')
            .replace('-', 'minus')  
        )
        fig.tight_layout()
        PlotStyle.save_fig(fig, f"{safe_name}_masa")
        plt.show()

    ############### Estimación de fondo por bandas laterales ###############
    if use_sideband:
        print('\n[4] Estimación de fondo por sideband substraction...')
        for charge, label in [(1, r'$B^{+}$'), (-1, r'$B^{-}$')]:
            print(f'{label}:')
            sideband_background_estimate(df[df['B_Charge'] == charge])

    ########################## Asimetría CP global ##########################
    print('\n[5] Asimetría CP global...')
    Np = len(df.query('B_Charge == 1'))
    Nm = len(df.query('B_Charge == -1'))
    acp_simple = compute_acp(Np, Nm)
    print_acp(f'{channel_name} - conteo simple', acp_simple)
    n_plus = results_fit.get(1, {}).get('n_signal', np.nan)
    n_minus = results_fit.get(-1, {}).get('n_signal', np.nan)
    if (np.isfinite(n_plus) and np.isfinite(n_minus) and (n_plus + n_minus) > 0):
        acp_fit = compute_acp(int(n_plus), int(n_minus))
        print_acp(f'{channel_name} -del ajuste', acp_fit)
    else:
        acp_fit = None
    
    ######################## Dalitz y asimetría local ########################
    print('\n[6] Diagramas de Dalitz')
    df_sig = df.query(f'B_M > {mass_window[0]} & B_M < {mass_window[1]}').copy()
    df_sig = calc_dalitz_vars(df_sig)
    if charm_veto:
        charm_mask = (
            (
                (df_sig['m2_12'] < df_sig['m2_13']) & ((df_sig['m2_12'] < 1800**2) | (df_sig['m2_12'] > 2000**2))
            ) | (
                (df_sig['m2_12'] > df_sig['m2_13']) & ((df_sig['m2_13'] < 1800**2) | (df_sig['m2_13'] > 2000**2))
            )
        )
        df_sig = df_sig[charm_mask]
        print(f'Tras charm veto: {len(df_sig):,} eventos')
    Bp_sig = df_sig[df_sig['B_Charge'] ==  1]
    Bm_sig = df_sig[df_sig['B_Charge'] == -1]
    BINS_D = 15
    RANGE_X = [0.0, 15.0]   # [GeV²/c⁴]
    RANGE_Y = [0.0, 30.0]   # [GeV²/c⁴]
    hBp, xb, yb = np.histogram2d(
        Bp_sig['R0low']/1e6, Bp_sig['R0high']/1e6, bins=BINS_D, range=[RANGE_X, RANGE_Y]
    )
    hBm, _, _ = np.histogram2d(
        Bm_sig['R0low']/1e6, Bm_sig['R0high']/1e6, bins=BINS_D, range=[RANGE_X, RANGE_Y]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        tot = hBp + hBm
        A_map = np.where(tot > 0, (hBm - hBp)/tot, np.nan)
        sA_map = np.where(tot > 0, np.sqrt((1 - A_map**2)/tot), np.nan)
        S_map = np.where(sA_map > 0, A_map/sA_map, np.nan)
    plot_dalitz_sumary(hBp, hBm, A_map, sA_map, S_map, xb, yb, channel_name, charm_veto, safe_name)
    
    # Diagramas de Dalitz (Scatter)
    plot_dalitz_scatter(
        df = df_sig,
        channel_name = channel_name,
        safe_name = safe_name,
        s = 0.13,
        color = 'red',
        alpha = 0.45,
        rasterized = True
    )

    print(f'\nAnálisis completado para {channel_name}')
    return {
        'channel': channel_name,
        'n_events': len(df),
        'acp_simple': acp_simple,
        'acp_fit': acp_fit,
        'fit_results': results_fit,
        'data': df
    }