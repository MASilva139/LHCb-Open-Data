## Diseño de las gráficas
Los gráficos 2D de probabilidades usan especialmente el diseño de su primera gráfica: fondo azul oscuro, texto blanco, mapa de color divergente y una normalización que permite distinguir regiones con pocos eventos.

### Anatomía de la primera gráfica 2D (LHCb_WR_01)

```python
fig, ax = plt.subplots(figsize=(6, 4.5))
```

- `fig` representa el lienzo completo: incluye el área del gráfico, los márgenes y la barra de color.
- `ax` representa el área de ejes donde se dibujan los datos.
- `figsize=(6, 4.5)` fija el ancho y alto en **pulgadas**. Cambiar el primer número ensancha la figura; cambiar el segundo aumenta su altura.

```python
ax.set_facecolor("#0a0a2e")
fig.patch.set_facecolor("#0a0a2e")
```

- `ax.set_facecolor(...)` modifica el fondo **interior** de los ejes.
- `fig.patch.set_facecolor(...)` modifica el fondo **exterior** del lienzo.
- Se aplican ambos porque, si solo se cambia uno, pueden quedar márgenes blancos alrededor del gráfico.

```python
h, xedges, yedges, img = ax.hist2d(
    x,
    y,
    bins=50,
    range=[[0.5, 1.0], [0.0, 0.5]],
    cmap="coolwarm",
    norm=mcolors.PowerNorm(gamma=0.4)
)
```

- `x` y `y` contienen las variables que se ubicarán en los ejes horizontal y vertical.
- `bins=50` divide cada eje en 50 intervalos; por tanto, se construye una cuadrícula de hasta \(50\times50\) celdas. Más bins muestran más detalle, pero también aumentan el ruido estadístico.
- `range=[[xmin, xmax], [ymin, ymax]]` fija los límites usados para construir el histograma. Los eventos fuera de esos límites no se dibujan.
- `cmap="coolwarm"` selecciona la paleta. Los colores representan la cantidad de eventos de cada celda.
- `norm=mcolors.PowerNorm(gamma=0.4)` modifica cómo se traducen los conteos a colores. Con `gamma < 1`, las regiones poco pobladas se vuelven más visibles; con `gamma = 1`, la escala vuelve a ser lineal; con `gamma > 1`, se enfatizan más las regiones de alta densidad.
- `h` contiene la matriz numérica de conteos.
- `xedges` y `yedges` contienen los bordes de los bins.
- `img` es el objeto gráfico que se entrega a la barra de color.

```python
cbar = fig.colorbar(img, ax=ax)
```

- Crea una barra que relaciona los colores con la frecuencia.
- `img` garantiza que la barra use exactamente la misma paleta y normalización.
- `ax=ax` indica a qué ejes pertenece y permite que Matplotlib reserve el espacio adecuado.

```python
cbar.set_label("Frecuencia", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
```

- `set_label` modifica el texto descriptivo de la barra.
- `set_tick_params` cambia el color de las pequeñas marcas.
- `plt.setp(...get_ticklabels())` modifica los números escritos junto a esas marcas.

```python
ax.set_xlabel(...)
ax.set_ylabel(...)
ax.set_title(...)
```

- Modifican la etiqueta horizontal, la vertical y el título.
- `color="white"` evita que el texto desaparezca sobre el fondo oscuro.
- Puede usarse `fontsize=...`, `fontweight="bold"` o `pad=...` para cambiar tamaño, grosor y separación.

```python
ax.tick_params(colors="white")
```

Cambia simultáneamente el color de los números y marcas de los ejes.

```python
for spine in ax.spines.values():
    spine.set_edgecolor("white")
```

Los `spines` son las cuatro líneas que delimitan el rectángulo del gráfico. El ciclo cambia el color de todas.

```python
plt.tight_layout()
```

Recalcula automáticamente los márgenes para evitar que títulos, etiquetas o la barra de color queden recortados.

```python
fig.savefig(
    GRAPHICS_DIR / "nombre.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
```

- `dpi` controla la resolución del archivo. Un valor entre 150 y 300 suele ser suficiente; 1000 produce archivos muy grandes y rara vez mejora un informe común.
- `bbox_inches="tight"` elimina márgenes sobrantes.
- `facecolor=fig.get_facecolor()` conserva el fondo oscuro al guardar; sin este argumento, algunos formatos o configuraciones pueden exportar el fondo blanco.

```python
plt.show()
```

Muestra la figura y marca el final lógico de la construcción del gráfico.

### Otros parámetros usados en el notebook

- `histtype="step"` dibuja solo el contorno de un histograma y facilita comparar distribuciones.
- `linewidth` controla el grosor de líneas.
- `alpha` controla transparencia: `0` es invisible y `1` es completamente opaco.
- `density=False` muestra conteos; `density=True` normaliza el área total a uno.
- `s` en `scatter` controla el área de cada marcador.
- `rasterized=True` convierte solo los puntos en una capa ráster al exportar PDF/SVG, reduciendo el tamaño cuando hay miles de eventos.
- `extent=[xmin, xmax, ymin, ymax]` asigna coordenadas físicas a una matriz dibujada con `imshow` o `matshow`.
- `origin="lower"` coloca el origen en la esquina inferior izquierda; sin él, las matrices suelen aparecer invertidas verticalmente.
- `aspect="auto"` permite que el gráfico ocupe el espacio disponible sin forzar celdas cuadradas.
- `vmin` y `vmax` fijan los extremos de la escala de color. En una asimetría se usa normalmente `[-1, 1]`; en significancia puede usarse `[-5, 5]`.
- `cmap="RdBu_r"` es útil para variables con signo: un extremo representa valores negativos, el centro representa cero y el otro extremo valores positivos.
- `yscale("log")` usa escala logarítmica y permite ver simultáneamente regiones con conteos muy diferentes. No debe usarse si los valores a representar son negativos.
- `label` asigna un nombre a una curva o conjunto de datos; `ax.legend()` construye la leyenda con esos nombres.
- `zorder` controla qué objeto queda encima de otro: valores mayores se dibujan al frente.