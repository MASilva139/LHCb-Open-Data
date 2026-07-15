## Origen del material
Este repositorio toma como base material educativo del proyecto LHCb Open Data y lo adapta para uso didáctico. El objetivo es facilitar el análisis de datos abiertos de LHCb en entornos locales para
- Windows, sin requerir de ROOT, utilizando `uproot`, `pandas`, `numpy`, `matplotlib` y `scipy`.
- Ubuntu, empleando ROOT, utilizando `pyroot`.

El material se organiza en:
1. **[01_estudiante](01_estudiante)**: notebooks para completar durante la práctica.
2. **[02_resuelto](02_resuelto)**: notebooks con las soluciones completas.
3. **[03_modularizacion](03_modularizacion)**: ejercicio para reorganizar el análisis en archivos `.py`.

El proyecto original de LHCb Open Data está publicado bajo licencia GPL-2.0.

<!-- # Hackaton 2024
El presente repositorio desarrolla los archivos Jupyter del Hackaton 2024, véase el [repositorio original](https://github.com/matchuf/PWF_Gt-Hn-CR_LHCb_open-data).

Los notebooks implementados para google colab (LHCb_Open_Data_Project_PWF_whitout-root.ipynb), alternativa para no usar conda y root, se desarrollaron de manera nativa en Windows. Además se realizó una versión para `conda+root` para sistema operativo `Linux`.

Los archivos root `B2HHH_MagnetUp.root`, `B2HHH_MagnetDown.root` y `PhaseSpaceSimulation.root` pueden encontrarse en la página de datos abiertos del [CENR](https://opendata.cern.ch/record/4900); sin embargo, en el repositorio original está el archivo de root `PhaseSpaceSimulation.root` de manera que solo se tendrían que descargar los archivos `B2HHH_MagnetUp.root` y `B2HHH_MagnetDown.root`.


## Implementación de código en Google Colab
En el caso de querer ejecutar los cuadernos en Google Colab, además de necesitar instalar los paquetes
```shell
pip install uproot awkward awkward-pandas
```
se procederá a descargar el archivo [open-data-project-1.1.tar.gz](https://github.com/matchuf/PWF_Gt-Hn-CR_LHCb_open-data), descomprimir y guardar en el Google Drive. Así mismo, se necesita que los archivos [B2HHH_MagnetUp.root](https://drive.google.com/file/d/10FGQvnZBjzQ4F0d_ZfLzTeoZjKD1IGl0/view?usp=sharing) y [B2HHH_MagnetDown.root](https://drive.google.com/file/d/1meKf8X6jOc176WJOasT21tcITngCRjXh/view?usp=sharing) estén guardados en una carpeta `YPUR_DRIVE_PATH/opendata-project-1.1/Data/`.

Al inicio del cuaderno agregar el siguiente comando
```python
from google.colab import drive
drive.mount('/content/drive')

import os
#os.chdir("drive/My Drive/<your-folder>")
os.chdir("/content/drive/MyDrive/work/opendata-project-1.1")

# Check that you are in the correct path
!ls
``` -->