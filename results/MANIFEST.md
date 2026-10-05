# Procedencia de los resultados (Lab 7)

Qué modelo y configuración produjo cada archivo de `results/`. Todos los resultados de analogías, SimLex y AG News
se calculan sobre el **mismo corpus preprocesado** (`cache/`: prefijo de WikiText-103, 29,552,958 tokens, vocabulario de
109,646 entradas con `min_count = 5`) y el mismo split de AG News (90/10 estratificado, semilla 42).

## Modelos de embeddings

| Nombre en las tablas | Run / archivo | Modelo | Configuración de entrenamiento | Tiempo | Hardware |
|---|---|---|---|---|---|
| **SGNS (d300)** (ganador) | `cache/runs/d300_c100.json`, matriz `d300_c100_W_in.npy` | SGNS propio, PyTorch desde cero (`sgns.py`) | d=300, ventana 5, 5 negativos, subsampling t=1e-4, 3 epochs, SparseAdam lr=3e-3 (decae a 1 %), batch 4096, semilla 42, corpus 100 % | 3,650 s | CPU (modelo de CPU/RAM: **pendiente de que lo reporte A**) |
| **SGNS (d100)** | `cache/runs/d100_c100.json` | igual | igual, d=100 | 1,906 s | igual |
| **gensim (d300, 3 ep)** (referencia del SGNS) | `results/gensim_d300_e3.json`, `cache/gensim_d300_e3.kv` | `Word2Vec` de gensim, skip-gram + negative sampling | d=300, ventana 5, 5 negativos, sample=1e-4, 3 epochs, alpha 0.025→1e-4, `min_count=1` sobre el vocabulario prefiltrado, semilla 42 | 71 s | Apple M5, 10 núcleos, 16 GB, sin GPU (10 hilos) |
| **gensim (d100, 10 ep)** (corrida anterior, conservada aparte) | `results/gensim_d100_e10.json`, `cache/gensim_d100_e10.kv` | igual | igual, d=100, 10 epochs | 147 s | igual |
| **GloVe-100** | `glove-wiki-gigaword-100` (sin entrenamiento propio) | GloVe (Pennington et al., 2014) | ver `results/glove_reportado.json`: Wikipedia 2014 + Gigaword 5 (6 B tokens), ventana 10, AdaGrad lr 0.05, 50 iteraciones | matriz ≈ 85 min (1 hilo) + 14 min/iteración con d=300 y 32 núcleos; **total para d=100 no reportado** | dual Intel Xeon E5-2658 2.1 GHz, 32 núcleos |

El `.kv` del mejor SGNS para entregar (float16, 65 MB; no cambia las métricas) está en `vectors/sgns_d300_c100.kv`.
No se reporta memoria de GPU: ningún entrenamiento usó GPU.

## Archivos de resultados

| Archivo | Lo produce | Contenido | Modelos |
|---|---|---|---|
| `resumen_modelos.csv` / `.json` | `build_resumen.py` | Una fila por modelo con todo lo de la tabla de la sección 7 (corpus, vocabulario, dimensión, configuración, tiempo, hardware, analogías, WordSim, SimLex, paralelismo, OOV y F1 de AG News) y la fuente de cada columna | SGNS d300, SGNS d100, gensim d300, gensim d100 (sin SimLex/AG News), GloVe, TF-IDF, aleatorio |
| `gensim_d300_e3.json`, `gensim_d100_e10.json` | `gensim_glove.ipynb` | Log por epoch (pérdida, analogías, WordSim, vecinos, tiempos), configuración, hardware | gensim, una corrida cada uno |
| `glove_reportado.json` | `gensim_glove.ipynb` | Entrenamiento y hardware de GloVe según sus autores (con fuente y la cota estimada, marcada como estimación propia) | GloVe |
| `aritmetica_vectorial.json` | `aritmetica_vectorial.ipynb` | Analogías con **vocabulario compartido de 30,000 palabras** (3CosAdd y 3CosMul, por categoría, con y sin exclusión de a, b y c), paralelismo de vectores diferencia, pureza t-SNE, analogías propias. Cada modelo trae su `procedencia` | los 5 modelos de arriba |
| `clasificacion_B.json` | `clasificacion_agnews.ipynb` | Baseline TF-IDF, variantes congelada/fine-tuning por inicialización (val, 3 semillas, curvas por epoch), test una vez por mejor variante, matrices de confusión, OOV, SimLex-999 y WordSim-353 | aleatorio, SGNS d300, SGNS d100, gensim d300, GloVe, TF-IDF |
| `baseline_curvas.json` | `baseline_tfidf_curvas.ipynb` | Curvas de pérdida de train/validación del baseline por iteración de L-BFGS (C=10) | TF-IDF + regresión logística |
| `curva_datos.csv` | `curva_datos_agnews.ipynb` | F1/accuracy de test por (modelo, variante, fracción, semilla): 416 corridas. Las filas `gensim (d100, 10 ep)` son de la corrida anterior | aleatorio, SGNS d300, SGNS d100, gensim d300, gensim d100, GloVe, TF-IDF |
| `curva_datos_meta.json` | `curva_datos_agnews.ipynb` | Procedencia de cada curva y protocolo (fracciones, semillas, learning rates, early stopping) | idem |
| `clasificacion_sgns/` | `clasificacion_sgns.ipynb` (A) | Clasificación del SGNS d300 corrida por A con `clasificacion.py`; **reproduce exactamente** el resultado de `clasificacion_B.json` (F1 de test 0.91703, misma matriz de confusión) | SGNS d300 |

## Figuras (`results/figs/`)

| Figura | Notebook | Modelo(s) |
|---|---|---|
| `analogias_por_categoria.png`, `paralelismo.png` | `aritmetica_vectorial.ipynb` | los 5 modelos |
| `tsne_mejor_modelo.png` | `aritmetica_vectorial.ipynb` | SGNS (d300), el mejor SGNS por 3CosAdd con vocabulario compartido |
| `gensim_curvas_por_epoch.png` | `gensim_glove.ipynb` | gensim d300 y d100 |
| `clasificacion_curvas_perdida.png`, `clasificacion_matrices_confusion.png` | `clasificacion_agnews.ipynb` | todas las variantes; matrices de la mejor variante por inicialización |
| `baseline_tfidf_curvas_perdida.png` | `baseline_tfidf_curvas.ipynb` | TF-IDF + regresión logística |
| `f1_vs_fraccion_datos.png` | `curva_datos_agnews.ipynb` | todas las curvas (gráfica 2 de la sección 7); la corrida anterior de gensim no se grafica |

## Cosas a tener en cuenta

- **Dos corridas de gensim, no mezclar:** la referencia del SGNS es `gensim (d300, 3 ep)`; `gensim (d100, 10 ep)` tiene otra dimensión y otro número de epochs.
- **Analogías (accuracy):** vocabulario compartido de las 30,000 palabras más frecuentes presentes en todos los modelos; mismas 11,833 de 19,544 preguntas para todos (cobertura 60.5 %). Los valores del `Evaluator` de `sgns.py` (por epoch, sobre el vocabulario propio) coinciden con estos a ±0.0003.
- **WordSim-353 / SimLex-999** de `gensim (d100, 10 ep)` no se calcularon (SimLex se reservó para la evaluación final de la sección 6, que usa gensim d300).
- **Tiempo de entrenamiento de GloVe:** no está publicado para d=100; ver la cota estimada y su advertencia en `glove_reportado.json`.
- Las matrices `*_W_in.npy` / `*_W_out.npy` no se versionan desde `results/` (`.gitignore`); las de `cache/runs/` sí están en el historial de A.
