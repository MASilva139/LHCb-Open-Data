##############################################################
#                       Lectura de Datos                     #
##############################################################
from pathlib import Path
import pandas as pd
import uproot

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
    
def load_magnet_data(
    data_dir: Path,
    preselection: str | None = None
) -> pd.DataFrame:
    dfs = []
    filenames = ('B2HHH_MagnetUp.root', 'B2HHH_MagnetDown.root')
    for fname in filenames:
        file_path = data_dir / fname
        try:
            df = load_decay_tree(file_path, cut = preselection)
            dfs.append(df)
            print(f'{fname}: {len(df):,} eventos')
        except FileNotFoundError as e:
            print(f'[Error] {e}')
    return pd.concat(dfs, ignore_index=True)
