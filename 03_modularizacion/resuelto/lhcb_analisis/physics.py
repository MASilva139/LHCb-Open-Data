###############################################################
#                         Cálculos físicos                    #
###############################################################
import numpy as np
import pandas as pd

def calc_B_mass(
    df: pd.DataFrame, 
    masses: tuple[float, float, float]
) -> pd.DataFrame:
    df = df.copy()
    for i, mass in zip((1, 2, 3), masses):
        h = f'H{i}'
        # E = sqrt(px² + py² + pz² + m²)
        df[f'{h}_E'] = np.sqrt(
            mass**2 
            + df[f'{h}_PX']**2 
            + df[f'{h}_PY']**2 
            + df[f'{h}_PZ']**2
        )
    df['B_E'] = df['H1_E'] + df['H2_E'] + df['H3_E']
    df['B_PX'] = df['H1_PX'] + df['H2_PX'] + df['H3_PX']
    df['B_PY'] = df['H1_PY'] + df['H2_PY'] + df['H3_PY']
    df['B_PZ'] = df['H1_PZ'] + df['H2_PZ'] + df['H3_PZ']
    df['B_M'] = np.sqrt(np.clip(
        df['B_E']**2 
        - df['B_PX']**2 
        - df['B_PY']**2 
        - df['B_PZ']**2,
        0, 
        None
    ))
    df['B_Charge'] = df['H1_Charge'] + df['H2_Charge'] + df['H3_Charge']
    return df

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

def print_acp(label: str, result: dict) -> None:
    print(f'\n=== {label} ===')
    print(f'  N+ = {result["Np"]:,}   N- = {result["Nm"]:,}')
    print(f'  A_CP = {result["A"]:+.4f} ± {result["sigma"]:.4f}')
    print(f'  Significancia = {result["significance"]:+.2f} σ')