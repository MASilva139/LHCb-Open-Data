import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy import stats
from scipy.optimize import curve_fit
from histograma_f import Formato, DARK_BACKGROUND, LIGHT_TEXT

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Rutas de las carpetas
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'Data'
Path('GChanels').mkdir(exist_ok = True)

# Masa invariante [MeV/c²]
mK = 493.677
mPi = 139.570

# Parámetros de ajuste
MASS_MIN = 5100.0 # [MeV/c²]
MASS_MAX = 5500.0 # [MeV/c²]
N_BINS = 80
BIN_WIDTH = (MASS_MAX - MASS_MIN)/N_BINS
MASS_CENTER = (MASS_MAX + MASS_MIN)/2.0

# Funciones de utilidades geerales
class funcgeneral:
    @staticmethod
    def load_decay_tree(file_path: Path, cut: str | None = None) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f'No encontrado: {file_path}')
        with uproot.open(file_path) as f:
            if 'DecayTree' not in f:
                raise KeyError(f'Sin árbol DecayTree: {file_path.name}')
            tree = f['DecayTree']
            return tree.arrays(
                expressions = tree.keys(), 
                cut = cut, 
                library = 'pd'
            )

    @staticmethod
    def calc_B_mass(df: pd.DataFrame, masses: tuple) -> pd.DataFrame:
        df = df.copy()
        m1, m2, m3 = masses
        for i, m in zip((1, 2, 3), (m1, m2, m3)):
            h = f'H{i}'
            # E = sqrt(px² + py² + pz² + m²)
            df[f'{h}_E'] = np.sqrt(
                m**2 + df[f'{h}_PX']**2 + df[f'{h}_PY']**2 + df[f'{h}_PZ']**2
            )
        df['B_E'] = df['H1_E'] + df['H2_E'] + df['H3_E']
        df['B_PX'] = df['H1_PX'] + df['H2_PX'] + df['H3_PX']
        df['B_PY'] = df['H1_PY'] + df['H2_PY'] + df['H3_PY']
        df['B_PZ'] = df['H1_PZ'] + df['H2_PZ'] + df['H3_PZ']
        df['B_M'] = np.sqrt(np.clip(
            df['B_E']**2 - df['B_PX']**2 - df['B_PY']**2 - df['B_PZ']**2,
            0, None
        ))
        df['B_Charge'] = df['H1_Charge'] + df['H2_Charge'] + df['H3_Charge']
        return df

    @staticmethod
    def calc_dalitz_vars(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # m²(H1+H2) = (E1+E2)² - (px1+px2)² - (py1+py2)² - (pz1+pz2)²
        df['m2_12'] = (
            (df['H1_E'] + df['H2_E'])**2
            - (df['H1_PX'] + df['H2_PX'])**2
            - (df['H1_PY'] + df['H2_PY'])**2
            - (df['H1_PZ'] + df['H2_PZ'])**2
        )
        df['m2_13'] = (
            (df['H1_E'] + df['H3_E'])**2
            - (df['H1_PX'] + df['H3_PX'])**2
            - (df['H1_PY'] + df['H3_PY'])**2
            - (df['H1_PZ'] + df['H3_PZ'])**2
        )
        # Versión ordenada: R0low ≤ R0high (Dalitz plot "plegado")
        df['R0low'] = df[['m2_12', 'm2_13']].min(axis=1)
        df['R0high'] = df[['m2_12', 'm2_13']].max(axis=1)
        return df

    @staticmethod
    def compute_acp(Np: int, Nm: int) -> dict:
        total = Np + Nm
        if total == 0:
            return {
                'A': np.nan, 
                'sigma': np.nan,
                'significance': np.nan, 
                'Np': 0, 
                'Nm': 0
            }
        A = (Nm - Np)/total
        sigma = np.sqrt((1 - A**2)/total)
        sig = A/ sigma if sigma > 0 else np.nan
        return {
            'A': A, 
            'sigma': sigma, 
            'significance': sig, 
            'Np': Np, 
            'Nm': Nm
        }

    @staticmethod
    def print_acp(label: str, result: dict):
        print(f'\n=== {label} ===')
        print(f'  N+ = {result["Np"]:,}   N- = {result["Nm"]:,}')
        print(f'  A_CP = {result["A"]:+.4f} ± {result["sigma"]:.4f}')
        print(f'  Significancia = {result["significance"]:+.2f} σ')

# Funciones de ajuste y estimación de fondo
class models:
    ## Modelos
    @staticmethod
    def exponential_pdf(x, slope):
        lower = MASS_MIN - MASS_CENTER
        upper = MASS_MAX - MASS_CENTER
        if abs(slope) < 1e-10:
            return np.ones_like(x)/(MASS_MAX - MASS_MIN)
        norm = (np.exp(slope * upper) - np.exp(slope * lower))/slope
        return np.exp(slope*(x - MASS_CENTER))/norm

    @staticmethod
    def chebyshev2_pdf(x, c0, c1, c2):
        t = 2 * (x - MASS_MIN)/(MASS_MAX - MASS_MIN) - 1
        pdf = c0 + c1*t + c2*(2*t**2 - 1)
        pdf = np.clip(pdf, 0, None)     # Asegurar positividad
        norm = np.trapezoid(pdf, x)
        return pdf/(norm + 1e-30)# Asegurar positividad

    @staticmethod
    def gauss_pdf(x, mu, sigma):
        return stats.norm.pdf(x, loc=mu, scale=sigma)

    # ── Modelos combinados ──────────────────────────────────────
    @staticmethod
    def model_gauss_exp(x, n_sig, mu, sigma, n_bg, slope):
        return BIN_WIDTH * (
            n_sig*models.gauss_pdf(x, mu, sigma) + n_bg*models.exponential_pdf(x, slope)
        )

    @staticmethod
    def model_gauss_cheb(x, n_sig, mu, sigma, n_bg, c0, c1, c2):
        return BIN_WIDTH*(
            n_sig*models.gauss_pdf(x, mu, sigma) + n_bg*models.chebyshev2_pdf(x, c0, c1, c2)
        )

    # ── Función genérica de ajuste ──────────────────────────────
    @staticmethod
    def fit_mass(masses_array, model='gauss_exp', verbose=True):
        masses = np.asarray(masses_array, dtype=float)
        masses = masses[np.isfinite(masses) & (masses >= MASS_MIN) & (masses <= MASS_MAX)]
        counts, edges = np.histogram(masses, bins=N_BINS, range=(MASS_MIN, MASS_MAX))
        centers = (edges[:-1] + edges[1:]) / 2.0
        uncert  = np.sqrt(np.maximum(counts, 1.0))
        N_total = counts.sum()
        if model == 'gauss_exp':
            fn = models.model_gauss_exp
            p0 = [0.65*N_total, 5279, 18, 0.35*N_total, -0.002]
            lo = [0, 5240, 3,  0,   -0.02]
            hi = [2*N_total, 5320, 60, 2*N_total, 0.02]
            n_sig_idx = 0
        else:  # gauss_cheb
            fn = models.model_gauss_cheb
            p0 = [0.50*N_total, 5279, 18, 0.50*N_total, 1.0, 0.0, 0.0]
            lo = [0, 5240, 3,  0,  0, -2, -2]
            hi = [2*N_total, 5320, 60, 2*N_total, 1e6, 2, 2]
            n_sig_idx = 0
        try:
            popt, pcov = curve_fit(
                fn, 
                centers, 
                counts,
                p0=p0, 
                sigma=uncert, 
                absolute_sigma=True,
                bounds=(lo, hi), 
                maxfev=300000
            )
        except RuntimeError as e:
            print(f'  [WARNING] Ajuste no convergió: {e}')
            popt = p0
            pcov = np.diag(np.ones(len(p0)))
        expected = fn(centers, *popt)
        chi2 = np.sum(((counts - expected) / uncert)**2)
        ndf = len(counts) - len(popt)
        n_sig = popt[n_sig_idx]
        n_sig_err = np.sqrt(np.clip(pcov[n_sig_idx, n_sig_idx], 0, None))
        if verbose:
            print(f'N_señal = {n_sig:.0f} ± {n_sig_err:.0f}')
            print(f'χ²/ndf  = {chi2:.1f} / {ndf}')
            print(f'Media   = {popt[1]:.2f} [MeV/c²]')
            print(f'  σ     = {popt[2]:.2f} [MeV/c²]')
        return {
            'n_signal': n_sig, 
            'n_signal_error': n_sig_err,
            'n_bg': popt[3], 
            'popt': popt, 
            'pcov': pcov,
            'chi2': chi2, 
            'ndf': ndf,
            'counts': counts, 
            'centers': centers, 
            'edges': edges,
            'model_fn': fn, 
            'model_name': model
        }

    # ── Sideband subtraction (para canales con fondo alto) ──────
    @staticmethod
    def sideband_background_estimate(
        df, 
        mass_col: str = 'B_M',
        signal_window: tuple = (5228, 5330),
        left_band: tuple = (5100, 5200),
        right_band: tuple = (5400, 5500)
    ) -> dict:
        n_sig_region = len(df.query(f'{signal_window[0]} <= {mass_col} < {signal_window[1]}'))
        n_left = len(df.query(f'{left_band[0]} <= {mass_col} < {left_band[1]}'))
        n_right = len(df.query(f'{right_band[0]} <= {mass_col} < {right_band[1]}'))
        # Factor de escala: relación de anchos
        w_sig = signal_window[1] - signal_window[0]
        w_bands = (left_band[1] - left_band[0] + right_band[1] - right_band[0]) / 2
        scale = w_sig/w_bands
        n_bg_est = ((n_left + n_right)/2)*scale
        n_sig_est = n_sig_region - n_bg_est
        sob = n_sig_est/np.sqrt(n_bg_est + 1e-9)
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
            'SoverSqrtB': sob
        }

# Analisis por canal
class analisis:
    @staticmethod
    def run_channel_analysis(
        channel_name: str,
        preselection: str,
        masses: tuple,
        fit_model: str = 'gauss_exp',
        charm_veto: bool = True,
        use_sideband: bool = False,
        mass_window: tuple = (5194, 5364)
    ) -> dict | None:
        print(f'\n{"="*60}')
        print(f'  CANAL: {channel_name}')
        print(f'{"="*60}')
        # ── 1. Carga de datos ──────────────────────────────────
        print('\n[1] Cargando datos...')
        dfs = []
        for fname in ('B2HHH_MagnetUp.root', 'B2HHH_MagnetDown.root'):
            fpath = DATA_DIR / fname
            try:
                dfs.append(funcgeneral.load_decay_tree(fpath, cut=preselection))
                print(f'  {fname}: {len(dfs[-1]):,} eventos')
            except FileNotFoundError as e:
                print(f'  [ERROR] {e}')
                return None
        df = pd.concat(dfs, ignore_index=True)
        print(f'  Total tras preselección: {len(df):,} eventos')
        safe_name = channel_name.replace('→', 'to').replace(' ', '_')
        # -----------------------------------------------------------
        # ------------------ Histograma 2D
        # -----------------------------------------------------------
        dp_fig, dp_ax = plt.subplots(1, 3, figsize = (20, 4.5))
        images = []
        for hi, ax in zip([1, 2, 3], dp_ax):
            ax.set_facecolor(DARK_BACKGROUND)
            counts_2d, xedges, yedges, image = ax.hist2d(
                df[f'H{hi}_ProbK'],
                df[f'H{hi}_ProbPi'],
                bins = 50,
                range = [[0.5, 1.0], [0.0, 0.5]],
                cmap = 'coolwarm',
                norm = mcolors.PowerNorm(gamma = 0.4)
            )
            images.append(image)
            ax.grid(color='#000000', linestyle='--', alpha=0.25)
            Formato.add_dark_colorbar(dp_fig, ax, image, label='Frecuencia')
            Formato.apply_dark_axes_style(
                dp_fig,
                ax,
                f'H{hi}: Probabilidades $\kappa$ vs. $\pi$',
                'Probabilidad de ser kaón',
                'Probabilidad de ser pión'
            )
        dp_fig.suptitle(
            'Distribuición de Probabilidades',
            color = LIGHT_TEXT,
            fontsize = 18,
            fontweight = 'bold'
        )
        dp_fig.tight_layout()
        Formato.save_fig(dp_fig, f'GChanels/{channel_name}-dist_prob.png')
        plt.show()
        # -----------------------------------------------------------
        # ── 2. Reconstrucción de masa invariante ───────────────
        print('\n[2] Reconstruyendo masa invariante del B...')
        df = funcgeneral.calc_B_mass(df, masses)
        # ── 3. Histograma de masa y ajuste ────────────────────
        print('\n[3] Ajuste de la distribución de masa...')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{channel_name} — Masa invariante del B', fontsize=13, fontweight='bold')
        results_fit = {}
        for charge, label, ax in zip([1, -1], ['B⁺', 'B⁻'], axes):
            masses_arr = df.loc[df['B_Charge'] == charge, 'B_M'].values
            print(f'  {label}: {len(masses_arr):,} candidatos en [{MASS_MIN},{MASS_MAX}] [MeV/c²]')
            res = models.fit_mass(masses_arr, model=fit_model, verbose=True)
            results_fit[charge] = res
            x_dense = np.linspace(MASS_MIN, MASS_MAX, 1000)
            # ax.step(res['centers'], res['counts'], where='mid', color='steelblue', label='Datos', linewidth=1)
            ax.step(res['centers'], res['counts'], where='mid', color='#000000', label='Datos', linewidth=1)
            ax.plot(x_dense, res['model_fn'](x_dense, *res['popt']), 'r-', linewidth=1.5, label='Ajuste total')
            ax.axvline(res['popt'][1], color='green', linestyle='--', linewidth=1, label=f"μ = {res['popt'][1]:.1f} [MeV]")
            ax.set_xlabel(r'$M_B$ [MeV/$c^2$]')
            ax.set_ylabel('Eventos / bin')
            ax.set_title(label)
            ax.legend(fontsize=8)
            ax.set_xlim(MASS_MIN, MASS_MAX)
        plt.tight_layout()
        plt.savefig(f'GChanels/{safe_name}_masa.png', dpi=150, bbox_inches='tight')
        plt.show()
        # ── 4. Estimación de fondo por sideband (opcional) ────
        if use_sideband:
            print('\n[4] Estimación de fondo por sideband subtraction...')
            for charge, label in [(1, 'B⁺'), (-1, 'B⁻')]:
                print(f'  {label}:')
                models.sideband_background_estimate(df[df['B_Charge'] == charge])
        # ── 5. Asimetría CP global ─────────────────────────────
        print('\n[5] Asimetría CP global (conteo simple)...')
        # Primero con conteo simple
        Np = len(df.query('B_Charge == 1'))
        Nm = len(df.query('B_Charge == -1'))
        acp_simple = funcgeneral.compute_acp(Np, Nm)
        funcgeneral.print_acp(f'{channel_name} — conteo simple', acp_simple)
        # Asimetría con rendimiento del ajuste
        n_plus = results_fit.get( 1, {}).get('n_signal', np.nan)
        n_minus = results_fit.get(-1, {}).get('n_signal', np.nan)
        n_plus_err  = results_fit.get( 1, {}).get('n_signal_error', np.nan)
        n_minus_err = results_fit.get(-1, {}).get('n_signal_error', np.nan)
        if np.isfinite(n_plus) and np.isfinite(n_minus) and (n_plus + n_minus) > 0:
            acp_fit = funcgeneral.compute_acp(int(n_plus), int(n_minus))
            funcgeneral.print_acp(f'{channel_name} — del ajuste', acp_fit)
        else:
            acp_fit = None
        # ── 6. Diagramas de Dalitz y asimetría local ──────────
        print('\n[6] Diagramas de Dalitz...')
        df_sig = df.query(f'B_M > {mass_window[0]} & B_M < {mass_window[1]}').copy()
        df_sig = funcgeneral.calc_dalitz_vars(df_sig)
        if charm_veto:
            # Eliminar resonancias D0 (ventana ~1800–2000 MeV/c²)
            df_sig = df_sig.query(
                '((m2_12 < m2_13) & ((m2_12 < 1800**2) | (m2_12 > 2000**2))) |'
                '((m2_12 > m2_13) & ((m2_13 < 1800**2) | (m2_13 > 2000**2)))'
            )
            print(f'  Tras charm veto: {len(df_sig):,} eventos')
        Bp_sig = df_sig[df_sig['B_Charge'] ==  1]
        Bm_sig = df_sig[df_sig['B_Charge'] == -1]
        # Histogramas 2D para B⁺ y B⁻
        BINS_D = 15
        # XMAX_D = df_sig['R0low'].quantile(0.98) if len(df_sig) > 0 else 10e6
        # YMAX_D = df_sig['R0high'].quantile(0.98) if len(df_sig) > 0 else 30e6
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
        #-------------------------------------
        fig2 = plt.figure(figsize=(25, 8))
        fig2.patch.set_facecolor('#000000')
        gs = gridspec.GridSpec(
            2, 4,
            figure = fig2,
            width_ratios = [1.4, 1.4, 1.45, 1.45],
            wspace = 0.5,
            hspace = 0.4
        )
        ax_bp = fig2.add_subplot(gs[0, 0])
        ax_bm = fig2.add_subplot(gs[0, 1])
        ax_A = fig2.add_subplot(gs[1, 0])
        ax_sA = fig2.add_subplot(gs[1, 1])
        ax_S = fig2.add_subplot(gs[:, 2:])
        fig2.suptitle(
            f'{channel_name} - Dalitz ordenado (charm veto = {charm_veto})',
            fontsize = 14,
            fontweight = 'bold',
            color = 'white'
        )
        dalitz_cmap = plt.get_cmap('RdBu_r').copy()
        dalitz_cmap.set_bad('#000000')
        asym_cmap = plt.get_cmap('RdBu_r').copy()
        asym_cmap.set_bad('#000000')
        unc_cmap = plt.get_cmap('viridis').copy()
        unc_cmap.set_bad('#000000')
        sig_cmap = plt.get_cmap('RdBu_r').copy()
        sig_cmap.set_bad('#000000')
        ext = [xb[0], xb[-1], yb[0], yb[-1]]

        # Dalitz B+
        im_bp = ax_bp.imshow(
            np.ma.masked_where(hBp.T == 0, hBp.T),
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = dalitz_cmap,
            norm = mcolors.PowerNorm(gamma = 0.5)
        )
        ax_bp.set_title(r'Dalitz $B^{+}$', color = 'white')
        # ax_bp.set_facecolor('#000000')
        cbar_bp = fig2.colorbar(im_bp, ax = ax_bp, shrink = 0.95)
        cbar_bp.set_label('Eventos por bin', color = 'white')
        cbar_bp.ax.tick_params(colors = 'white')

        # Dalitz B-
        im_bm = ax_bm.imshow(
            np.ma.masked_where(hBm.T == 0, hBm.T),
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = dalitz_cmap,
            norm = mcolors.PowerNorm(gamma = 0.5)
        )
        ax_bm.set_title(r'Dalitz $B^{-}$', color = 'white')
        ax_bm.set_facecolor('#000000')
        cbar_bm = fig2.colorbar(im_bm, ax = ax_bm, shrink = 0.95)
        cbar_bm.set_label('Eventos por bin', color = 'white')
        cbar_bm.ax.tick_params(colors = 'white')

        # Dalitz asimetría local
        im_A = ax_A.imshow(
            np.ma.masked_invalid(A_map.T),
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = asym_cmap,
            vmin = -1,
            vmax = 1
        )
        ax_A.set_title(r'Asimetría local', color = 'white')
        ax_A.set_facecolor('#000000')
        cbar_A = fig2.colorbar(im_A, ax = ax_A, shrink = 0.95)
        cbar_A.set_label(r'$A_{CP}^{\mathrm{local}}$', color = 'white')
        cbar_A.ax.tick_params(colors = 'white')

        # Incertidumbre de la asimetría local
        im_sA = ax_sA.imshow(
            np.ma.masked_invalid(sA_map.T),
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = dalitz_cmap
        )
        ax_sA.set_title(r'Incertidumbre local', color = 'white')
        ax_sA.set_facecolor('#000000')
        cbar_sA = fig2.colorbar(im_sA, ax = ax_sA, shrink = 0.95)
        cbar_sA.set_label(r'$\sigma(A_{CP}^{\mathrm{local}})$', color = 'white')
        cbar_sA.ax.tick_params(colors = 'white')

        # Dalitz significancia de la asimetría local
        im_S = ax_S.imshow(
            np.ma.masked_invalid(S_map.T),
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = sig_cmap,
            vmin = -5,
            vmax = 5
        )
        ax_S.set_title(r'Significancia de la asimetría local', color = 'white')
        ax_S.set_facecolor('#000000')
        cbar_S = fig2.colorbar(im_S, ax = ax_S, shrink = 0.95)
        cbar_S.set_label(r'$A/\sigma_{A}$', color = 'white')
        cbar_S.ax.tick_params(colors = 'white')
        
        for ax in [ax_bp, ax_bm, ax_A, ax_sA, ax_S]:
            ax.set_xlabel(r'$m^{2}_{Low} (KK)\,[{GeV}^{2}/c^{4}]$', color = 'white')
            ax.set_ylabel(r'$m^{2}_{High} (KK)\,[{GeV}^{2}/c^{4}]$', color = 'white')
            ax.tick_params(
                axis = 'both',
                which = 'major',
                colors = 'white',
                bottom = True,
                left = True,
                labelbottom = True,
                labelleft = True
            )
            for spine in ax.spines.values():
                spine.set_color('white')
        plt.tight_layout()
        plt.savefig(
            f'GChanels/{safe_name}_dalitz.png', 
            dpi = 500, 
            bbox_inches = 'tight',
            facecolor = fig2.get_facecolor()
        )
        #-------------------------------------
        #-------------------------------------
        plt.show()
        # ------------------- 7. Grafica de Dalitz con scatter 
        fig3, axes3 = plt.subplots(1, 2, figsize = (14, 7))
        fig3.suptitle(f'{channel_name} — Dalitz ordenado (Scatter)', fontsize=12, fontweight='bold')
        axes3[0].scatter(
            df_sig['m2_12']/1e6,
            df_sig['m2_13']/1e6,
            s = 0.3,
            color = 'red',
            alpha = 0.45,
            rasterized = True,
        )
        axes3[0].set_title(r'Diagrama de Dalitz')
        axes3[0].set_xlabel(r"$m_{12}^{2}$ [GeV$^2/c^4$]")
        axes3[0].set_ylabel(r"$m_{13}^{2}$ [GeV$^2/c^4$]")

        axes3[1].scatter(
            df_sig['R0low']/1e6,
            df_sig['R0high']/1e6,
            s = 0.3, 
            color = 'red',
            alpha = 0.45,
            rasterized = True
        )
        axes3[1].set_title(r'Diagrama de Dalitz ordenado')
        axes3[1].set_xlabel(r"$m_{\mathrm{Low}}^{2}$ [GeV$^2/c^4$]")
        axes3[1].set_ylabel(r"$m_{\mathrm{High}}^{2}$ [GeV$^2/c^4$]")
        for ax in axes3:
            ax.grid(alpha = 0.2, linestyle = '--')
        plt.tight_layout()
        plt.savefig(f'GChanels/{safe_name}_dalitz_scatter.png', dpi=500, bbox_inches='tight')
        plt.show()

        print(f'\n  Análisis completado para {channel_name}.')
        return {
            'channel': channel_name,
            'n_events': len(df),
            'acp_simple': acp_simple,
            'acp_fit': acp_fit,
            'fit_results': results_fit,
        }