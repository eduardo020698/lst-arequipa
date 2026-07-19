# Downscaling de LST para Arequipa con U-Net

Réplica adaptada de *Kurchaba & Meyer (2026), "Spatiotemporal Downscaling and
Nowcasting of Urban Land Surface Temperatures With Deep Neural Networks"*
(IEEE Access) para el Perú:

| Componente | Paper (Europa) | Este proyecto (Arequipa) |
|---|---|---|
| Geoestacionario | SEVIRI/Meteosat, 3 km / 15 min | **GOES-East ABI**, 2 km / 1 h (producto LSTF) |
| Polar (verdad) | MODIS MOD21/MYD21, 1 km | **VIIRS VNP21A1** día+noche, 1 km |
| Canales extra | ángulo cenital solar | ángulo solar + **DEM Copernicus** (clave en los Andes) |
| Zona | ciudades europeas >1M hab | Arequipa (15.2–17.6°S, 73.0–70.6°O) |

La U-Net aprende a mapear el campo GOES de 2 km hacia el campo VIIRS de 1 km.
Una vez entrenada, convierte **cada hora de GOES** en un mapa de 1 km — alta
resolución espacial Y temporal a la vez.

## Requisitos

1. Cuenta gratuita de NASA Earthdata: https://urs.earthdata.nasa.gov (para VIIRS).
   GOES y el DEM son de acceso anónimo — no requieren cuenta.
2. Google Colab con GPU (gratis) o cualquier máquina con GPU.

## Cómo correr (Colab)

1. Abre el notebook directo desde GitHub (un clic):
   https://colab.research.google.com/github/eduardo020698/lst-arequipa/blob/main/notebooks/entrenamiento_colab.ipynb
2. Activa GPU: `Entorno de ejecución → Cambiar tipo → T4 GPU`.
3. Corre las celdas en orden: el código se clona solo y los datos/checkpoints
   se guardan en tu Google Drive (carpeta `lst_arequipa`). Primero la
   **prueba de humo** (5 días, ~10 min); si pasa, lanza el rango completo.
4. El entrenamiento es **reanudable** (`--resume`): si Colab corta la sesión,
   vuelve a correr la celda y continúa desde el último checkpoint.

## Estructura

```
scripts/geo_utils.py       grillas, proyecciones (GOES geoestacionaria,
                           sinusoidal VIIRS), ángulo solar, remapeo KDTree
scripts/download_goes.py   LST GOES 2km desde AWS (anónimo) -> grilla 0.01°
scripts/download_viirs.py  LST VIIRS 1km vía earthaccess -> grilla 0.01°
scripts/build_dataset.py   pares colocalizados + DEM -> data/samples/*.npz
model/unet.py              U-Net (entrada 3 canales, salida residual)
train.py                   entrenamiento con split temporal, RMSE/MBE en °C
notebooks/entrenamiento_colab.ipynb   orquestador para Colab
```

## Expectativas honestas

- El paper logra RMSE 1.92 °C con ~20 años de datos y decenas de ciudades.
  Con 1–2 años y una sola región espera algo mayor (2.5–4 °C) — suficiente
  como proof-of-concept y mejorable agregando años o ciudades andinas.
- VIIRS/GOES solo miden en **cielo despejado**: los samples nublados se
  descartan solos (por eso Arequipa es ideal: ~300 días despejados/año).
- El split de validación/test es **por fecha** (últimos tramos del período)
  para evitar fuga temporal de información, igual que el paper.
- Volumen estimado: ~2 GB por año de pares procesados (los GRIB/HDF crudos
  se descartan tras el remapeo).

## Siguiente paso al terminar

Trae el archivo `checkpoints/best.pt` y lo integro al dashboard: el workflow
bajará GOES cada 3 h, le aplicará la red y publicará la capa LST de 1 km
sobre Arequipa junto a las capas CAMS existentes.
