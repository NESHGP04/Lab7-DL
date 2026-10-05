"""
Prepara las figuras y los datos del informe (PDF de máx. 5 páginas) a partir de los resultados del repositorio.
Escribe informe/figs/*.png e informe/data.json; el documento Word se arma con informe/build_informe.js.

    python informe/prep_informe.py        (desde la raíz del repo)
"""
import base64
import glob
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import evaluation as EV
import utils as U

np.seterr(all="ignore")
OUT = Path("informe"); FIGS = OUT / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
R = Path("results"); CACHE = Path("cache")
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3})
D = {}

# ---------- Zipf (figura del notebook de preprocesamiento)
nb = json.load(open("preprocesamiento.ipynb"))
for c in nb["cells"]:
    if "Frecuencia vs. rango" in "".join(c["source"]):
        for o in c.get("outputs", []):
            if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
                (FIGS / "zipf.png").write_bytes(base64.b64decode(o["data"]["image/png"]))
assert (FIGS / "zipf.png").exists(), "no se encontró la figura de Zipf"

# ---------- Pares skip-gram por epoch (antes y después del submuestreo)
meta = json.load(open(CACHE / "vocab.json")); offsets = np.load(CACHE / "train_offsets.npy"); ids = np.load(CACHE / "train_ids.npy")
counts = np.array(meta["counts"], dtype=np.int64)
v = types.SimpleNamespace(counts=counts, in_vocab_total=int(counts[2:].sum()))
rng = np.random.default_rng(42)
pares = [dict(config="sin submuestreo", tokens=len(ids), fija=U.count_skipgram_pairs(offsets, 5), dinamica=U.count_skipgram_pairs(offsets, 5, dynamic=True))]
for t in [1e-3, 1e-4, 1e-5]:
    si, so = U.subsample(ids, offsets, U.keep_probs(v, t=t, formula="gensim"), rng)
    pares.append(dict(config=f"t = {t:g}", tokens=len(si), fija=U.count_skipgram_pairs(so, 5), dinamica=U.count_skipgram_pairs(so, 5, dynamic=True)))
D["pares"] = pares

# ---------- Iteraciones del SGNS (cache/runs)
runs = {}
for f in sorted(glob.glob(str(CACHE / "runs" / "*.json"))):
    h = json.load(open(f)); runs[h["config"]["name"]] = h
orden = ["d50_c100", "d100_c25", "d100_c25_lr1e-2", "d100_c25_neg15", "d100_c25_t1e-3", "d100_c25_w10", "d100_c50", "d100_c100", "d300_c100"]
tabla_runs = []
for n in orden:
    h = runs[n]; c = h["config"]; e = h["epochs"][-1]
    tabla_runs.append(dict(run=n, dim=c["dim"], ventana=c["window"], neg=c["neg"], t=c["subsample_t"], lr=c["lr"], epochs=c["epochs"],
                           tokens=h["n_tokens"], pares=e["pairs"], loss=e["loss"], sem=e["analogy_sem"], syn=e["analogy_syn"], tot=e["analogy_total"],
                           ws=e["wordsim353"], tiempo=h["train_time_s"], loss_ep=[x["loss"] for x in h["epochs"]], tot_ep=[x["analogy_total"] for x in h["epochs"]],
                           ws_ep=[x["wordsim353"] for x in h["epochs"]]))
D["runs"] = tabla_runs
best = runs["d300_c100"]
D["best_epochs"] = [dict(epoch=e["epoch"], loss=e["loss"], sem=e["analogy_sem"], syn=e["analogy_syn"], tot=e["analogy_total"], ws=e["wordsim353"], t=e["time_s"]) for e in best["epochs"]]
D["best_neighbors"] = best["epochs"][-1]["neighbors"]
D["best_neighbors_ep1"] = best["epochs"][0]["neighbors"]

# Curvas por epoch (3 iteraciones: d50, d100, d300)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 1.85))
for n, col in [("d50_c100", "tab:blue"), ("d100_c100", "tab:orange"), ("d300_c100", "tab:red")]:
    e = runs[n]["epochs"]; x = [k["epoch"] for k in e]
    lab = f"d={runs[n]['config']['dim']}"
    ax[0].plot(x, [k["loss"] for k in e], "o-", color=col, label=lab)
    ax[1].plot(x, [k["analogy_total"] for k in e], "o-", color=col, label=lab)
    ax[2].plot(x, [k["wordsim353"] for k in e], "o-", color=col, label=lab)
for a, t in zip(ax, ["Pérdida de entrenamiento", "Accuracy de analogías (total)", "Spearman WordSim-353"]):
    a.set_title(t); a.set_xlabel("epoch"); a.set_xticks([1, 2, 3]); a.grid(alpha=0.3)
ax[0].legend(); plt.tight_layout(pad=0.4); plt.savefig(FIGS / "sgns_curvas.png", dpi=220); plt.close()

# ---------- Gráfica 1: accuracy de analogías vs tokens del corpus (SGNS propio) con GloVe como referencia
glove = json.load(open(R / "aritmetica_vectorial.json"))["GloVe-100"]["add"]["total"]
fig, ax = plt.subplots(figsize=(3.45, 2.3))
d100 = [runs[n] for n in ["d100_c25", "d100_c50", "d100_c100"]]
ax.plot([h["n_tokens"] / 1e6 for h in d100], [h["epochs"][-1]["analogy_total"] for h in d100], "o-", color="tab:orange", label="SGNS d=100 (25/50/100 %)")
for n, mk, lab in [("d50_c100", "s", "SGNS d=50"), ("d300_c100", "D", "SGNS d=300")]:
    h = runs[n]; ax.plot(h["n_tokens"] / 1e6, h["epochs"][-1]["analogy_total"], mk, color="tab:blue" if "50" in n else "tab:red", label=lab)
ax.axhline(glove, color="k", ls="--", lw=0.9, label=f"GloVe-100: {glove:.3f} (6 000 M tokens)")
ax.set_xscale("log"); ax.set_xlabel("tokens del corpus de entrenamiento (millones)"); ax.set_ylabel("accuracy de analogías (3CosAdd)")
ax.set_title("Analogías vs. tamaño del corpus"); ax.grid(alpha=0.3); ax.legend(loc="center right"); ax.set_ylim(0, 0.72)
plt.tight_layout(pad=0.4); plt.savefig(FIGS / "grafica1_analogias_vs_tokens.png", dpi=220); plt.close()

# ---------- Gráfica 2: F1 de test vs fracción de datos (curvas principales)
df = pd.read_csv(R / "curva_datos.csv")
df["curva"] = np.where(df.variante.isin(["entrenada", "C elegido"]), df.modelo, df.modelo + " · " + df.variante)
agg = df.groupby(["curva", "frac"]).test_f1.agg(["mean", "std"]).reset_index()
estilo = {"TF-IDF + LogReg": ("black", ":", "o", "TF-IDF + LogReg"), "Aleatorio": ("tab:gray", "-", "o", "Aleatorio"),
          "SGNS (d300) · fine-tuning": ("tab:red", "-", "o", "SGNS d300 · fine-tuning"), "SGNS (d300) · congelada": ("tab:red", "--", "s", "SGNS d300 · congelada"),
          "GloVe-100 · fine-tuning": ("tab:orange", "-", "o", "GloVe · fine-tuning"), "GloVe-100 · congelada": ("tab:orange", "--", "s", "GloVe · congelada"),
          "gensim (d300, 3 ep) · fine-tuning": ("tab:green", "-", "o", "gensim d300 · fine-tuning")}
fig, ax = plt.subplots(figsize=(3.45, 2.3))
for k, (col, ls, mk, lab) in estilo.items():
    d = agg[agg.curva == k].sort_values("frac")
    ax.plot(d.frac * 100, d["mean"], ls, color=col, marker=mk, label=lab)
    ax.fill_between(d.frac * 100, d["mean"] - d["std"].fillna(0), d["mean"] + d["std"].fillna(0), color=col, alpha=0.1)
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 25, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "100"])
ax.set_xlabel("% de los datos de entrenamiento de AG News"); ax.set_ylabel("F1 macro en test"); ax.set_title("F1 de test vs. fracción de datos (5 semillas)")
ax.grid(alpha=0.3); ax.legend(loc="lower right", ncol=1)
plt.tight_layout(pad=0.4); plt.savefig(FIGS / "grafica2_f1_vs_fraccion.png", dpi=220); plt.close()
D["curva_1pct_100pct"] = {k: [float(agg[(agg.curva == k) & (agg.frac == f)]["mean"].iloc[0]) for f in (0.01, 1.0)] for k in agg.curva.unique()}

# ---------- Clasificación: curvas de pérdida y matrices de confusión
cl = json.load(open(R / "clasificacion_B.json")); bl = json.load(open(R / "baseline_curvas.json"))
sel = [("Aleatorio/entrenada", "Aleatorio"), ("SGNS (d300)/congelada", "SGNS d300 · congelada"), ("SGNS (d300)/fine-tuning", "SGNS d300 · fine-tuning"),
       ("GloVe-100/congelada", "GloVe · congelada"), ("GloVe-100/fine-tuning", "GloVe · fine-tuning")]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.0))
for a, (k, lab) in zip(ax.ravel(), sel):
    h = pd.DataFrame(cl["val"][k]["history"]); be = cl["val"][k]["info"]["best_epoch"]
    a.plot(h.epoch, h.train_loss, "o-", label="train"); a.plot(h.epoch, h.val_loss, "o-", label="validación"); a.axvline(be, color="gray", ls=":", lw=0.8)
    a.set_title(lab); a.set_xlabel("epoch"); a.grid(alpha=0.3)
b = pd.DataFrame(bl["curva_por_iteracion_lbfgs"]); a = ax.ravel()[5]
a.plot(b.iteracion, b.train_loss, "o-"); a.plot(b.iteracion, b.val_loss, "o-"); a.set_xscale("log"); a.set_title("TF-IDF + LogReg (L-BFGS)"); a.set_xlabel("iteración"); a.grid(alpha=0.3)
ax[0, 0].legend(); ax[0, 0].set_ylabel("entropía cruzada"); ax[1, 0].set_ylabel("entropía cruzada")
plt.tight_layout(pad=0.4); plt.savefig(FIGS / "clasif_curvas.png", dpi=220); plt.close()

C4 = ["World", "Sports", "Business", "Sci/Tech"]
paneles = [("TF-IDF + LogReg", "TF-IDF + LogReg"), ("Aleatorio (entrenada)", "Aleatorio"), ("SGNS (d300) (fine-tuning)", "SGNS d300 · fine-tuning"), ("GloVe-100 (fine-tuning)", "GloVe · fine-tuning")]
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.95))
for a, (k, lab) in zip(ax, paneles):
    cm = np.array(cl["test"][k]["confusion"]); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=5.8, color="white" if cmn[i, j] > 0.5 else "black")
    a.set_xticks(range(4)); a.set_xticklabels(C4, rotation=40, ha="right", fontsize=5.5); a.set_yticks(range(4)); a.set_yticklabels(C4, fontsize=5.5)
    a.set_title(f"{lab}\nF1 = {cl['test'][k]['f1']:.3f}", fontsize=6.5)
ax[0].set_ylabel("real", fontsize=6.5)
plt.tight_layout(pad=0.3); plt.savefig(FIGS / "clasif_confusion.png", dpi=220); plt.close()

# ---------- Analogías individuales (3 modelos principales)
sgns = EV.load_sgns("cache", "d300_c100", "SGNS (d300)")
gens = EV.load_kv(CACHE / "gensim_d300_e3.kv", "gensim (d300, 3 ep)")
import gensim.downloader as api
glo = EV.EmbeddingSpace.from_keyed_vectors(api.load("glove-wiki-gigaword-100"), "GloVe-100")
own = [("género", "man", "king", "woman", "queen"), ("género", "man", "actor", "woman", "actress"), ("país–capital", "france", "paris", "italy", "rome"),
       ("país–capital", "japan", "tokyo", "germany", "berlin"), ("país–gentilicio", "spain", "spanish", "france", "french"),
       ("comparativo", "good", "better", "bad", "worse"), ("superlativo", "good", "best", "bad", "worst"), ("tiempo verbal", "go", "went", "see", "saw"),
       ("plural", "car", "cars", "child", "children")]
filas = []
for tipo, a, b, c, d in own:
    f = dict(tipo=tipo, q=f"{a}:{b}::{c}:?", esperada=d)
    for sp in (sgns, gens, glo):
        top, rk = EV.analogy(sp, a, b, c, k=5, return_rank_of=d)
        f[sp.name] = dict(top1=top[0][0], sim=top[0][1], rango=rk, top5=[(w, s) for w, s in top])
        f[sp.name]["sin_excl"] = EV.analogy(sp, a, b, c, k=1, exclude_query=False)[0][0]
    filas.append(f)
D["analogias"] = filas

# ---------- Resultados agregados
ar = json.load(open(R / "aritmetica_vectorial.json"))
cats = [c["category"] for c in ar["SGNS (d300)"]["por_categoria"]]
main = ["SGNS (d300)", "gensim (d300, 3 ep)", "GloVe-100"]
catrows = []
for i, c in enumerate(cats):
    row = dict(cat=c, n_eval=ar["SGNS (d300)"]["por_categoria"][i]["n_eval"], n_total=ar["SGNS (d300)"]["por_categoria"][i]["n_total"])
    for m in main:
        row[m] = dict(add=ar[m]["por_categoria"][i]["acc_add"], mul=ar[m]["por_categoria"][i]["acc_mul"], par=ar[m]["paralelismo_por_categoria"][i]["mean_cos_pairs"])
    catrows.append(row)
D["categorias"] = catrows
D["categorias_resumen"] = {m: dict(add=ar[m]["add"], mul=ar[m]["mul"], par=ar[m]["paralelismo_medio"], azar=ar[m]["paralelismo_azar"], sin_excl=ar[m]["sin_exclusion"]) for m in ar["_meta"]["modelos"]}
D["propias_top1"] = {m: ar[m]["propias_top1"] for m in ar["_meta"]["modelos"]}
D["tsne_pureza"] = {m: ar[m]["tsne_pureza"] for m in ar["_meta"]["modelos"]}
D["spearman_categorias"] = dict(
    sgns_gensim=float(spearmanr([c["acc_add"] for c in ar["SGNS (d300)"]["por_categoria"]], [c["acc_add"] for c in ar["gensim (d300, 3 ep)"]["por_categoria"]]).statistic),
    sgns_glove=float(spearmanr([c["acc_add"] for c in ar["SGNS (d300)"]["por_categoria"]], [c["acc_add"] for c in ar["GloVe-100"]["por_categoria"]]).statistic))
D["resumen"] = json.load(open(R / "resumen_modelos.json"))
D["clasif"] = dict(val={k: dict(info={kk: vv for kk, vv in v["info"].items()}) for k, v in cl["val"].items()}, test=cl["test"], mejor=cl["mejor_variante"],
                   oov=cl["oov"], tfidf=cl["tfidf"], simlex=cl["simlex"])
D["glove"] = json.load(open(R / "glove_reportado.json"))
json.dump(D, open(OUT / "data.json", "w"), indent=1, default=float, ensure_ascii=False)
print("ok", sorted(p.name for p in FIGS.iterdir()))
