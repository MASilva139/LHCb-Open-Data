from pathlib import Path
import pandas as pd
import uproot

def find_tree(root_file) -> str:
    classnames = root_file.classnames()
    for key in classnames:
        clean_key = key.split(';')[0]
        if clean_key == 'DecayTree':
            return key
    available_trees = [key for key, class_name in classnames.items() if 'TTree' in class_name]
    if not available_trees:
        print('Contenido del archivo ROOT:')
        print(root_file.keys())
        print('\nClases encontradas:')
        print(classnames)
        raise KeyError('No se encontró ningún árbol TTree dento del archivo ROOT.')
    return available_trees[0]

def load_decay_tree(
    file_path: Path, 
    cut: str | None = None
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f'No encontrado: {file_path}')
    with uproot.open(file_path) as f:
        tree_name = find_tree(f)
        print(f'Árbol usado en {file_path.name}: {tree_name}')
        tree = f[tree_name]
        # if 'DecayTree' not in f:
        #     raise KeyError(f'Sin árbol DecayTree: {file_path.name}')
        # tree = f['DecayTree']
        return tree.arrays(
            expressions = tree.keys(), 
            cut = cut, 
            library = 'pd'
        )

# Método para un solo documento root
def load_root_data(
    data_dir: Path,
    file_name: str,
    preselection: str | None = None
) -> pd.DataFrame:
    file_path = data_dir / file_name
    df = load_decay_tree(file_path, cut = preselection)
    print(f'{file_name}: {len(df):,} eventos')
    return df

# Método para varios documentos root (concatenar archivos)
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

def load_phase_data(
    data_dir: Path,
    preselection: str | None = None
) -> pd.DataFrame:
    return load_root_data(
        data_dir = data_dir,
        file_name = 'PhaseSpaceSimulation.root',
        preselection = preselection
    )

def load_mu_data(
    data_dir: Path,
    preselection: str | None = None
) -> pd.DataFrame:
    return load_root_data(
        data_dir = data_dir,
        file_name = 'B2HHH_MagnetUp.root',
        preselection = preselection
    )

def load_md_data(
    data_dir: Path,
    preselection: str | None = None
) -> pd.DataFrame:
    return load_root_data(
        data_dir = data_dir,
        file_name = 'B2HHH_MagnetDown.root',
        preselection = preselection
    )