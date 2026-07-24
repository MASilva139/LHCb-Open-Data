import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import (DATA_DIR, MASS_MIN, MASS_MAX)
from .io import (
    load_magnet_data,
    load_md_data,
    load_mu_data,
    load_phase_data
)
from .physics import (
    calc_B_mass,
    calc_dalitz_vars,
    compute_acp,
    print_acp
)
from .plotting import (
    plot_momentum,
    plot_mass_fit,
    plot_probability_distributions,
    plot_dalitz_scatter,
    plot_dalitz_sumary
)

def normalize_fit_models(fit_model: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(fit_model, str):
        return [fit_model]
    return list(fit_model)

def run_channel_analysis(
    channel_name: str,
    preselection: str,
    masses: tuple,
    fit_model: str | list[str] | tuple[str, ...] = 'gauss_exp',
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