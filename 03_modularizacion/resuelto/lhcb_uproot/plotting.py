from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from .config import (
    DARK_BACKGROUND,
    LIGHT_TEXT,
    MASS_MIN,
    MASS_MAX
)
from .fitting import evaluate_fit_components
from .styles import PlotStyle

def plot_momentum(
        df,
        channel_name: str,
        safe_name: str | None = None,
        bins: int = 20,
        m_range: tuple = (float, float),
        save: bool = True
):
    momentum_comps = ['PX', 'PY', 'PZ', 'P']
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), squeeze=False)
    fig.suptitle(
        f'{channel_name} - Distribuciones de momentum',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for row_idx, hadron_id in enumerate([1, 2, 3]):
        h = f'H{hadron_id}'
        for col_idx, comp in enumerate(momentum_comps):
            ax = axes[row_idx, col_idx]
            column_name = f'{h}_{comp}'
            if column_name not in df.columns:
                raise KeyError(f'No existe la columna {column_name} en el DataFrame')
            values = df[column_name].dropna()
            ax.hist(
                values/1000,
                bins = bins,
                range = m_range,
                histtype = 'stepfilled',
                alpha = 0.75
            )
            PlotStyle.apply_dark_axes_style(
                fig, 
                ax, 
                title = column_name,
                xlabel = rf'${column_name}$ [GeV/$c^{2}$]',
                ylabel = 'Eventos'
            )
            ax.set_yscale('log')
            ax.grid(alpha = 0.18, linestyle = '--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
        PlotStyle.save_fig(fig, f'{safe_name}_momentum')
    plt.show()
    return fig, axes

def plot_mass_fit(ax, result:dict, label:str):
    x_dense = np.linspace(MASS_MIN, MASS_MAX)
    components = evaluate_fit_components(x_dense, result)
    ax.step(result['centers'], result['counts'], where='mid', color='steelblue', label='Datos', linewidth=1)
    ax.plot(x_dense, components['total'], color='red', linewidth='1.5', label='Ajuste total')
    ax.plot(x_dense, components['signal'], color='lime', linestyle='--', linewidth=1.2, label='Señal')
    ax.plot(x_dense, components['background'], color='orange', linestyle=':', linewidth=1.2, label='Fondo')
    ax.axvline(result['parameters']['mu'], color='white', linestyle='--', linewidth=1, alpha=0.8, label=f"μ = {result['parameters']['mu']:.1f} [MeV]")
    PlotStyle.apply_dark_axes_style(
        ax.figure,
        ax, 
        label,
        r'$M_{B}$ [MeV/$c^{2}$]',
        'Eventos/bin'
    )
    ax.set_xlim(MASS_MIN, MASS_MAX)
    ax.legend(fontsize=8, facecolor=DARK_BACKGROUND, edgecolor=LIGHT_TEXT, labelcolor=LIGHT_TEXT)

def plot_probability_distributions(
    df, 
    channel_name: str,
    bins: int = 50,
    gamma: float = 0.4 
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for hadron_id, ax in zip([1, 2, 3], axes):
        ax.set_facecolor(DARK_BACKGROUND)
        _, _, _, image = ax.hist2d(
            df[f'H{hadron_id}_ProbK'],
            df[f'H{hadron_id}_ProbPi'],
            bins = bins,
            range = [[0.5, 1.0], [0.0, 0.5]],
            cmap = 'coolwarm',
            norm = mcolors.PowerNorm(gamma = gamma)
        )
        ax.grid(color=LIGHT_TEXT, linestyle='--', alpha=0.12)
        PlotStyle.apply_dark_axes_style(
            fig, ax,
            rf'H{hadron_id}: Probabilidad $K$ vs. $\pi$',
            "Probabilidad de ser kaón",
            "Probabilidad de ser pión"
        )
        PlotStyle.add_dark_colorbar(fig, ax, image, label='Frecuencia')
    fig.suptitle(
        'Distribución de probabilidades',
        color = LIGHT_TEXT,
        fontsize = 18,
        fontweight = 'bold'
    )
    fig.tight_layout()
    safe_name = (
        channel_name
        .replace('→', 'to')     # U+2192: →
        .replace(' ', '_')
        .replace('±', "pm")        # U+00B1: ±
        .replace('+', 'plus')
        .replace('-', 'minus')  
    )
    PlotStyle.save_fig(fig, f'{safe_name}_dist_prob')
    plt.show()

def plot_dalitz_sumary(hBp, hBm, A_map, sA_map, S_map, xb, yb, channel_name, charm_veto, safe_name):
    fig = plt.figure(figsize=(25,8))
    fig.patch.set_facecolor(DARK_BACKGROUND)
    gs = gridspec.GridSpec(
        1, 2,
        figure = fig,
        width_ratios = [1.4, 1.45],
        wspace = 0.3
    )
    left_gs = gs[0].subgridspec(2, 2, wspace = 0.45, hspace = 0.35)
    right_gs = gs[1].subgridspec(1, 1)
    ax_bp = fig.add_subplot(left_gs[0, 0])
    ax_bm = fig.add_subplot(left_gs[0, 1])
    ax_A = fig.add_subplot(left_gs[1, 0])
    ax_sA = fig.add_subplot(left_gs[1, 1])
    ax_S = fig.add_subplot(right_gs[0, 0])
    fig.suptitle(
        f'{channel_name} - Dalitz ordenado (charm veto = {charm_veto})',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    dalitz_cmap = plt.get_cmap('RdBu_r').copy()
    dalitz_cmap.set_bad(DARK_BACKGROUND)
    asym_cmap = plt.get_cmap('RdBu_r').copy()
    asym_cmap.set_bad(DARK_BACKGROUND)
    # unc_cmap = plt.get_cmap('viridis').copy()
    unc_cmap = plt.get_cmap('RdBu_r').copy()
    unc_cmap.set_bad(DARK_BACKGROUND)
    sig_cmap = plt.get_cmap('RdBu_r').copy()
    sig_cmap.set_bad(DARK_BACKGROUND)
    ext = [xb[0], xb[-1], yb[0], yb[-1]]
    panels = [
        {
            'ax': ax_bp,
            'values': np.ma.masked_where(hBp.T == 0, hBp.T),
            'title': r'Dalitz $B^{+}$',
            'cmap': dalitz_cmap,
            'label': 'Eventos por bin',
            'vmin': None,
            'vmax': None
        },
        {
            'ax': ax_bm,
            'values': np.ma.masked_where(hBm.T == 0, hBm.T),
            'title': r'Dalitz $B^{-}$',
            'cmap': dalitz_cmap,
            'label': 'Eventos por bin',
            'vmin': None,
            'vmax': None
        },
        {
            'ax': ax_A,
            'values': np.ma.masked_invalid(A_map.T),
            'title': r'Asimetría local',
            'cmap': asym_cmap,
            'label': r'$A_{CP}^{local}$',
            'vmin': -1,
            'vmax': 1
        },
        {
            'ax': ax_sA,
            'values': np.ma.masked_invalid(sA_map.T),
            'title': r'Incertidumbre local',
            'cmap': unc_cmap,
            'label': r'$\sigma(A_{CP}^{local})$',
            'vmin': None,
            'vmax': None
        },
        {
            'ax': ax_S,
            'values': np.ma.masked_invalid(S_map.T),
            'title': r'Significancia de la asimetría local',
            'cmap': sig_cmap,
            'label': r'$A/\sigma_{A}$',
            'vmin': -5,
            'vmax': 5
        }
    ]
    for panel in panels:
        ax = panel['ax']
        image = ax.imshow(
            panel['values'],
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = panel['cmap'],
            vmin = panel['vmin'],
            vmax = panel['vmax'],
            interpolation = 'nearest'
        )
        PlotStyle.apply_dark_axes_style(
            fig, ax, panel['title'],
            r'$m^{2}_{\mathrm{Low}}(KK)\,[GeV^{2}/c^{4}]$',
            r'$m^{2}_{\mathrm{High}}(KK)\,[GeV^{2}/c^{4}]$'
        )
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        PlotStyle.add_dark_colorbar(fig, ax, image, label = panel['label'])
    fig.tight_layout()
    PlotStyle.save_fig(fig, f'{safe_name}_dalitz_completo')
    plt.show()

def plot_dalitz_scatter(
    df, 
    channel_name: str,
    safe_name: str | None = None,
    s: float = 0.13,
    color: str = 'red',
    alpha: float = 0.45,
    rasterized: bool = True,
    save: bool = True
):
    fig, axes = plt.subplots(1, 2, figsize = (14, 7))
    fig.suptitle(f'{channel_name} — Diagramas de Dalitz (Scatter)', fontsize=17, fontweight='bold')
    panels = [
        {
            'ax': axes[0],
            'title': 'Diagrama de Dalitz',
            'xlabel': r'$m_{12}^{2}$ [GeV$^2/c^4$]',
            'ylabel': r'$m_{13}^{2}$ [GeV$^2/c^4$]',
            'x': df['m2_12']/1e6,
            'y': df['m2_13']/1e6,
        }, 
        {
            'ax': axes[1],
            'title': 'Diagrama de Dalitz ordenado',
            'xlabel': r'$m_{\mathrm{Low}}^{2}$ [GeV$^2/c^4$]',
            'ylabel': r'$m_{\mathrm{High}}^{2}$ [GeV$^2/c^4$]',
            'x': df['R0low']/1e6,
            'y': df['R0high']/1e6
        }
    ]
    for panel in panels:
        ax = panel['ax']
        ax.scatter(
            panel['x'],
            panel['y'],
            s = s,
            color = color,
            alpha = alpha,
            rasterized = rasterized
        )
        ax.set_title(panel['title'])
        ax.set_xlabel(panel['xlabel'])
        ax.set_ylabel(panel['ylabel'])
        ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    if save:
        if safe_name is None:       
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
        PlotStyle.save_fig(fig, f'{safe_name}_dalitz_scatter')
    plt.show()
    return fig, axes