###########################################################
#                 Archivo de configuración                #
###########################################################
from pathlib import Path

# Rutas/Directorio
# BASE_DIR = Path.cwd()
# DATA_DIR = BASE_DIR / 'Data'
ROOT_DIR = Path(__file__).resolve().parents[3]  # Ubicación en la raiz
DATA_DIR = ROOT_DIR / 'Data'
# OUTPUT_DIR = ROOT_DIR / 'GChannels'
OUTPUT_DIR = Path.cwd() / 'assets/img_channels'
OUTPUT_DIR.mkdir(exist_ok=True)

# Masa invariante [MeV/c²]
mK = 493.677
mPi = 139.570

# Parámetros de ajuste
MASS_MIN = 5100.0 # [MeV/c²]
MASS_MAX = 5500.0 # [MeV/c²]
N_BINS = 80
BIN_WIDTH = (MASS_MAX - MASS_MIN)/N_BINS
MASS_CENTER = (MASS_MAX + MASS_MIN)/2.0

# Estilo Visual
DARK_BACKGROUND = "#030315"
LIGHT_TEXT = "white"
ACCENT_CYAN = "#62d9ff"
ACCENT_GREEN = "#55d98b"
ACCENT_MAGENTA = "#ff70d5"