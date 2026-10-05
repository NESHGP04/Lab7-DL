"""
Consolida los resultados finales de cada conjunto de embeddings en una tabla por modelo
(results/resumen_modelos.csv y results/resumen_modelos.json), con la procedencia de cada cifra.

Es la fuente para la tabla comparativa de la sección 7. No calcula nada nuevo: solo lee los JSON que escriben
los notebooks (aritmetica_vectorial, clasificacion_agnews, gensim_glove) y los logs de entrenamiento del SGNS.

    python build_resumen.py
"""
import json
from pathlib import Path

import pandas as pd

R = Path("results")
CACHE = Path("cache")

# nombre mostrado -> (run de entrenamiento, tipo)
MODELOS = {
    "SGNS (d300)": ("d300_c100", "sgns"),
    "SGNS (d100)": ("d100_c100", "sgns"),
    "gensim (d300, 3 ep)": ("gensim_d300_e3", "gensim"),
    "gensim (d100, 10 ep)": ("gensim_d100_e10", "gensim"),
    "GloVe-100": ("glove-wiki-gigaword-100", "glove"),
}


def cargar(path):
    return json.load(open(path)) if Path(path).exists() else {}


arit = cargar(R / "aritmetica_vectorial.json")
clas = cargar(R / "clasificacion_B.json")
glove = cargar(R / "glove_reportado.json")
simlex = {d["embeddings"]: d for d in clas.get("simlex", [])}
f1_test_variantes = clas.get("test", {})
mejor = clas.get("mejor_variante", {})

filas = []
for nombre, (run, tipo) in MODELOS.items():
    a = arit.get(nombre, {})
    fila = dict(modelo=nombre, run=run, tipo=tipo, dim=a.get("dim"), vocab=a.get("vocab"))
    if tipo == "sgns":
        log = cargar(CACHE / "runs" / f"{run}.json")
        cfg = log.get("config", {})
        fila.update(tokens_corpus=log.get("n_tokens"), epochs=cfg.get("epochs"), ventana=cfg.get("window"), negativos=cfg.get("neg"),
                    submuestreo_t=cfg.get("subsample_t"), optimizador=f"SparseAdam lr={cfg.get('lr')} (decae a {cfg.get('min_lr_frac')}x), batch {cfg.get('batch_size')}",
                    tiempo_entrenamiento_s=log.get("train_time_s"), hardware=f"dispositivo: {log.get('device')} (modelo de CPU/RAM: pendiente de reportar por A)",
                    memoria_pico_gpu_mb=None, fuente_entrenamiento=f"cache/runs/{run}.json")
    elif tipo == "gensim":
        log = cargar(R / f"{run}.json")
        cfg, meta = log.get("meta", {}).get("config", {}), log.get("meta", {})
        fila.update(tokens_corpus=meta.get("corpus", {}).get("tokens"), epochs=cfg.get("epochs"), ventana=cfg.get("window"), negativos=cfg.get("negative"),
                    submuestreo_t=cfg.get("sample"), optimizador=f"SGD par a par, alpha {cfg.get('alpha')} -> {cfg.get('min_alpha')}",
                    tiempo_entrenamiento_s=log.get("summary", {}).get("total_train_s"),
                    hardware=f"{meta.get('hardware', {}).get('cpu')}, {meta.get('hardware', {}).get('cpu_count')} núcleos, {meta.get('hardware', {}).get('ram_gb')} GB RAM, sin GPU ({meta.get('workers')} hilos)",
                    memoria_pico_gpu_mb=None, fuente_entrenamiento=f"results/{run}.json")
    else:
        t = glove.get("tiempo_reportado", {})
        fila.update(tokens_corpus=glove.get("tokens_corpus"), epochs=f"{glove.get('entrenamiento', {}).get('iteraciones_dim_menor_300')} iteraciones", ventana="10 (simétrica)",
                    negativos="n/a (no usa negative sampling)", submuestreo_t="n/a", optimizador="AdaGrad lr=0.05",
                    tiempo_entrenamiento_s=None,
                    hardware=glove.get("hardware_reportado"),
                    tiempo_reportado=f"matriz de co-ocurrencia ~{t.get('matriz_cooccurrencia_min')} min (1 hilo); {t.get('iteracion_d300_min')} min por iteración con d=300 y {t.get('iteracion_d300_nucleos')} núcleos; total para d=100: {t.get('total_d100')}",
                    memoria_pico_gpu_mb=None, fuente_entrenamiento="results/glove_reportado.json (Pennington et al., 2014)")
    if a:
        fila.update(cobertura_analogias=a["cobertura"], preguntas_evaluadas=a["preguntas_evaluadas"],
                    analogias_sem_3cosadd=a["add"]["semantic"], analogias_sin_3cosadd=a["add"]["syntactic"], analogias_total_3cosadd=a["add"]["total"],
                    analogias_sem_3cosmul=a["mul"]["semantic"], analogias_sin_3cosmul=a["mul"]["syntactic"], analogias_total_3cosmul=a["mul"]["total"],
                    analogias_total_3cosadd_sin_exclusion=a["sin_exclusion"]["3CosAdd sin exclusión"],
                    paralelismo_medio=a["paralelismo_medio"], paralelismo_azar=a["paralelismo_azar"],
                    tsne_pureza_vecinos=a["tsne_pureza"], fuente_analogias="results/aritmetica_vectorial.json (vocabulario compartido de 30,000 palabras)")
    s = simlex.get(nombre)
    if s:
        fila.update(wordsim353=s["WordSim-353 (vocab propio)"], simlex999=s["SimLex-999 (vocab propio)"], simlex999_vocab_compartido=s["SimLex-999 (vocab compartido)"],
                    fuente_similitud="results/clasificacion_B.json")
    oov = clas.get("oov", {}).get(nombre)
    if oov:
        fila["oov_agnews_test_pct"] = oov["OOV test (%)"]
    var = mejor.get(nombre)
    if var:
        t = f1_test_variantes.get(f"{nombre} ({var})")
        if t:
            fila.update(agnews_mejor_variante=var, agnews_f1_test=t["f1"], agnews_acc_test=t["acc"])
    filas.append(fila)

# referencias de clasificación que no son embeddings preentrenados
extra = []
for k in ("TF-IDF + LogReg", "Aleatorio (entrenada)"):
    if k in f1_test_variantes:
        extra.append(dict(modelo=k, agnews_f1_test=f1_test_variantes[k]["f1"], agnews_acc_test=f1_test_variantes[k]["acc"]))

df = pd.DataFrame(filas + extra)
df.to_csv(R / "resumen_modelos.csv", index=False)
json.dump(df.where(df.notna(), None).to_dict(orient="records"), open(R / "resumen_modelos.json", "w"), indent=1, ensure_ascii=False, default=float)
print(f"{len(df)} filas -> {R / 'resumen_modelos.csv'}")
print(df.T.to_string(max_colwidth=60))
