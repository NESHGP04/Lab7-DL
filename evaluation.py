"""
Módulo de evaluación de embeddings para el Lab 7 (CC3092).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SPECIAL_TOKENS = ("<pad>", "<unk>")
CONTROL_WORDS = ["king", "france", "computer", "good", "january", "run"]


# Datos de los benchmarks (incluidos en gensim)
def _datapath(name: str) -> str:
    from gensim.test.utils import datapath
    return datapath(name)


def load_analogies(path: str | None = None, lower: bool = True):
    """questions-words.txt -> lista de (categoría, a, b, c, d). 19,544 preguntas, 14 categorías."""
    path = path or _datapath("questions-words.txt")
    out, cat = [], None
    for line in Path(path).read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            cat = line[1:].strip()
            continue
        words = line.lower().split() if lower else line.split()
        out.append((cat, *words))
    return out


def load_word_pairs(name: str):
    """'wordsim353.tsv' o 'simlex999.txt' -> lista de (palabra1, palabra2, puntaje humano)."""
    out = []
    for line in Path(_datapath(name)).read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        w1, w2, score = line.split("\t")[:3]
        out.append((w1.lower(), w2.lower(), float(score)))
    return out


def is_semantic(category: str) -> bool:
    """Las 5 primeras categorías son semánticas; las 9 'gramN-...' son sintácticas."""
    return not category.startswith("gram")


# Espacio de embeddings
class EmbeddingSpace:
    """
    Matriz de embeddings + word2idx. Guarda una copia normalizada (norma L2 = 1) porque
    todas las métricas (analogías, vecinos, Spearman) usan similitud coseno.
    Los tokens especiales (<pad>, <unk>) se excluyen: nunca deben salir como respuesta.
    """

    def __init__(self, E, word2idx: dict, name: str = "", drop=SPECIAL_TOKENS):
        if hasattr(E, "detach"):                       # torch.Tensor / nn.Parameter
            E = E.detach().cpu().numpy()
        E = np.asarray(E, dtype=np.float32)
        itos = [None] * len(E)
        for w, i in word2idx.items():
            if i < len(E):
                itos[i] = w
        keep = [i for i, w in enumerate(itos) if w is not None and w not in drop]
        self.name = name
        self.itos = [itos[i] for i in keep]
        self.word2idx = {w: i for i, w in enumerate(self.itos)}
        self.E = np.ascontiguousarray(E[keep])
        norms = np.linalg.norm(self.E, axis=1, keepdims=True)
        self.unit = self.E / np.maximum(norms, 1e-12)

    @classmethod
    def from_keyed_vectors(cls, kv, name: str = ""):
        return cls(kv.vectors, kv.key_to_index, name=name)

    @classmethod
    def from_itos(cls, E, itos, name: str = ""):
        return cls(E, {w: i for i, w in enumerate(itos)}, name=name)

    def __len__(self):
        return len(self.itos)

    def __contains__(self, w):
        return w in self.word2idx

    @property
    def dim(self):
        return self.E.shape[1]

    def vec(self, w):
        return self.E[self.word2idx[w]]

    def subset(self, words, name: str | None = None):
        """Nuevo espacio con solo `words` (en ese orden); ignora las que no están."""
        words = [w for w in words if w in self.word2idx]
        idx = [self.word2idx[w] for w in words]
        return EmbeddingSpace(self.E[idx], {w: i for i, w in enumerate(words)},
                              name=name or self.name)

    def restrict(self, n: int):
        """Las primeras n palabras (más frecuentes si el vocabulario viene ordenado por frecuencia)."""
        return self.subset(self.itos[:n])

    def to_keyed_vectors(self):
        from gensim.models import KeyedVectors
        kv = KeyedVectors(self.dim)
        kv.add_vectors(self.itos, self.E)
        return kv


def save_kv(space_or_E, itos_or_path, path=None, dtype=None):
    """
    Guarda vectores en formato gensim .kv (lo pide el enunciado). save_kv(space, path) o save_kv(E, itos, path).
    dtype=np.float16 reduce el archivo a la mitad (útil por el límite de 100 MB de GitHub; no cambia las métricas).
    """
    if path is None:
        space, path = space_or_E, itos_or_path
    else:
        space = EmbeddingSpace.from_itos(space_or_E, itos_or_path)
    kv = space.to_keyed_vectors()
    if dtype is not None:
        kv.vectors = kv.vectors.astype(dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kv.save(str(path))


def load_kv(path, name: str = ""):
    from gensim.models import KeyedVectors
    return EmbeddingSpace.from_keyed_vectors(KeyedVectors.load(str(path)), name=name)


def load_sgns(cache="cache", run=None, name: str = "SGNS", verbose: bool = True):
    """
    SGNS propio de A. Busca la matriz W_in guardada por sgns.save_run (<run>_W_in.npy, filas alineadas con
    cache/vocab.json) en cache/runs/, results/ y cache/; si `run` es None usa cache/sgns_best.kv.
    Devuelve None (con aviso) si todavía no existe.
    """
    cache = Path(cache)
    if run is None and (cache / "sgns_best.kv").exists():
        return load_kv(cache / "sgns_best.kv", name=name)
    for d in (cache / "runs", cache.parent / "results", cache):
        f = d / f"{run}_W_in.npy"
        if run and f.exists():
            itos = json.load(open(cache / "vocab.json"))["itos"]
            return EmbeddingSpace.from_itos(np.load(f), itos, name=name)
    if verbose:
        print(f"AVISO: no se encontró el SGNS '{run}' ({name}); se continúa sin él.")
    return None


def hardware_info() -> dict:
    """Máquina donde corre el código (para reportar hardware en la tabla comparativa)."""
    import platform
    import subprocess
    info = dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count())
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info["ram_gb"] = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30)
    except Exception:
        pass
    try:
        import torch
        info["gpu"] = "CUDA" if torch.cuda.is_available() else "ninguna (CPU)"
    except Exception:
        pass
    return info


def shared_vocabulary(spaces, n: int = 30000, order_by=None):
    """
    Las n palabras más frecuentes presentes en TODOS los espacios.
    La frecuencia se toma del orden de `order_by` (por defecto el primer espacio; los tres
    modelos vienen ordenados por frecuencia).
    """
    ref = order_by or spaces[0]
    common = set.intersection(*[set(s.itos) for s in spaces])
    return [w for w in ref.itos if w in common][:n]


# Analogías
def _sims(A, U):
    """A @ U.T. NumPy 2.0 + Accelerate (Apple Silicon) lanza avisos espurios en matmul; los resultados son finitos."""
    with np.errstate(all="ignore"):
        return A @ U.T


def _scores(space: EmbeddingSpace, ia, ib, ic, method: str):
    """Puntaje de todas las palabras candidatas para a:b::c:?  (filas = preguntas)."""
    U = space.unit
    Sa, Sb, Sc = _sims(U[ia], U), _sims(U[ib], U), _sims(U[ic], U)
    if method == "add":                                  # 3CosAdd: cos(x,b) - cos(x,a) + cos(x,c)
        return Sb - Sa + Sc
    if method == "mul":                                  # 3CosMul (Levy & Goldberg, 2014)
        f = lambda s: (s + 1.0) / 2.0
        return f(Sb) * f(Sc) / (f(Sa) + 1e-3)
    raise ValueError(method)


def analogy(space: EmbeddingSpace, a: str, b: str, c: str, k: int = 5,
            exclude_query: bool = True, method: str = "add", return_rank_of: str | None = None):
    """
    «a es a b como c es a ?».  3CosAdd con vectores normalizados:
        q = u_b - u_a + u_c,   resultado = argmax_x cos(x, q)
    Devuelve [(palabra, puntaje)] de las k mejores. Con method="add" el puntaje es la similitud
    coseno con el vector resultante q; con "mul" es el puntaje 3CosMul.
    exclude_query=True excluye a, b y c de los candidatos.
    return_rank_of=d agrega el rango (1 = primero) de la respuesta correcta d.
    """
    for w in (a, b, c):
        if w not in space:
            raise KeyError(f"'{w}' no está en el vocabulario de {space.name or 'el modelo'}")
    ia, ib, ic = (np.array([space.word2idx[w]]) for w in (a, b, c))
    if method == "add":
        q = space.unit[ib[0]] - space.unit[ia[0]] + space.unit[ic[0]]
        s = _sims((q / np.linalg.norm(q))[None], space.unit)[0]   # coseno con el vector resultante
    else:
        s = _scores(space, ia, ib, ic, method)[0]
    s = s.copy()
    if exclude_query:
        s[[ia[0], ib[0], ic[0]]] = -np.inf
    top = np.argpartition(-s, min(k, len(s) - 1))[:k]
    top = top[np.argsort(-s[top])]
    res = [(space.itos[i], float(s[i])) for i in top]
    if return_rank_of is not None:
        if return_rank_of not in space:
            return res, None
        rank = int((s > s[space.word2idx[return_rank_of]]).sum()) + 1
        return res, rank
    return res


def evaluate_analogies(space: EmbeddingSpace, questions=None, methods=("add", "mul"),
                       restrict_vocab: int | None = None, exclude_query: bool = True):
    """
    Accuracy en questions-words.txt. Solo se evalúan las preguntas cuyas 4 palabras están en el
    espacio (reporta cobertura = evaluadas / total). Los candidatos son todo el espacio, o las
    primeras `restrict_vocab` palabras (como gensim). Accuracy = aciertos top-1 / evaluadas.

    Devuelve dict con: coverage, n_eval, n_total, per_category (DataFrame) y, por método,
    {"semantic", "syntactic", "total"}.
    """
    questions = questions if questions is not None else load_analogies()
    if restrict_vocab:
        space = space.restrict(restrict_vocab)
    cats = np.array([q[0] for q in questions])
    w2i = space.word2idx
    idx = np.array([[w2i.get(w, -1) for w in q[1:]] for q in questions])
    ok = (idx >= 0).all(axis=1)

    correct = {m: np.zeros(len(questions), dtype=bool) for m in methods}
    rows = np.where(ok)[0]
    batch = max(32, int(3e7 // max(len(space), 1)))
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        ia, ib, ic, idd = idx[r, 0], idx[r, 1], idx[r, 2], idx[r, 3]
        for m in methods:
            sc = _scores(space, ia, ib, ic, m)
            if exclude_query:
                n = np.arange(len(r))
                sc[n, ia] = sc[n, ib] = sc[n, ic] = -np.inf
            correct[m][r] = sc.argmax(axis=1) == idd

    per_cat = []
    for cat in dict.fromkeys(cats):
        m_cat = cats == cat
        n_eval = int((m_cat & ok).sum())
        row = dict(category=cat, n_total=int(m_cat.sum()), n_eval=n_eval,
                   coverage=n_eval / m_cat.sum())
        for m in methods:
            row[f"acc_{m}"] = correct[m][m_cat & ok].sum() / n_eval if n_eval else np.nan
        per_cat.append(row)
    per_cat = pd.DataFrame(per_cat)

    sem = np.array([is_semantic(c) for c in cats])
    out = dict(coverage=float(ok.mean()), n_eval=int(ok.sum()), n_total=len(questions),
               per_category=per_cat)
    for m in methods:
        acc = lambda mask: float(correct[m][mask & ok].sum() / max((mask & ok).sum(), 1))
        out[m] = dict(semantic=acc(sem), syntactic=acc(~sem), total=acc(np.ones(len(cats), bool)))
    return out


# Similitud de palabras (Spearman)
def word_similarity(space: EmbeddingSpace, pairs):
    """Spearman entre coseno del modelo y puntaje humano; solo pares con ambas palabras en el espacio."""
    s_model, s_human = [], []
    for w1, w2, h in pairs:
        if w1 in space and w2 in space:
            s_model.append(float(space.unit[space.word2idx[w1]] @ space.unit[space.word2idx[w2]]))
            s_human.append(h)
    if len(s_model) < 3:
        return dict(spearman=np.nan, n_used=len(s_model), n_total=len(pairs), coverage=0.0)
    rho = spearmanr(s_model, s_human).statistic
    return dict(spearman=float(rho), n_used=len(s_model), n_total=len(pairs),
                coverage=len(s_model) / len(pairs))


# Vecinos más cercanos
def neighbors(space: EmbeddingSpace, words=CONTROL_WORDS, k: int = 5):
    """{palabra: [(vecino, coseno), ...]} sin incluir la palabra misma; None si no está en el vocabulario."""
    out = {}
    for w in words:
        if w not in space:
            out[w] = None
            continue
        s = _sims(space.unit[space.word2idx[w]][None], space.unit)[0]
        s[space.word2idx[w]] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        out[w] = [(space.itos[i], float(s[i])) for i in top]
    return out


# Paralelismo de vectores diferencia
def parallelism(space: EmbeddingSpace, questions=None, seed: int = 0):
    """
    Coseno promedio entre los vectores diferencia (u_b - u_a) de las parejas de una misma categoría
    (qué tan paralelas son las «flechas» a -> b). Por categoría devuelve:
      - mean_cos_pairs     : promedio sobre todos los pares de parejas distintas de la categoría
      - mean_cos_questions : promedio de cos(u_b - u_a, u_d - u_c) en las preguntas evaluadas
      - random_baseline    : lo mismo con parejas de palabras al azar (referencia de «azar»)
    """
    questions = questions if questions is not None else load_analogies()
    rng = np.random.default_rng(seed)
    U, w2i = space.unit, space.word2idx

    def mean_offdiag(D):
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        n = len(D)
        return float((np.linalg.norm(D.sum(0)) ** 2 - n) / (n * (n - 1))) if n > 1 else np.nan

    rows = []
    for cat in dict.fromkeys(q[0] for q in questions):
        qs = [q for q in questions if q[0] == cat and all(w in w2i for w in q[1:])]
        if len(qs) < 2:
            continue
        pairs = sorted({(q[1], q[2]) for q in qs} | {(q[3], q[4]) for q in qs})
        D = np.stack([U[w2i[b]] - U[w2i[a]] for a, b in pairs])
        Dq1 = np.stack([U[w2i[q[2]]] - U[w2i[q[1]]] for q in qs])
        Dq2 = np.stack([U[w2i[q[4]]] - U[w2i[q[3]]] for q in qs])
        cq = (Dq1 * Dq2).sum(1) / np.maximum(np.linalg.norm(Dq1, axis=1) * np.linalg.norm(Dq2, axis=1), 1e-12)
        r = rng.integers(0, len(U), size=(len(pairs), 2))
        rows.append(dict(category=cat, n_pairs=len(pairs), mean_cos_pairs=mean_offdiag(D),
                         mean_cos_questions=float(cq.mean()),
                         random_baseline=mean_offdiag(U[r[:, 1]] - U[r[:, 0]])))
    return pd.DataFrame(rows)


# Memoria de GPU y registro por epoch
def peak_gpu_mb():
    """Memoria pico de GPU en MB (CUDA: pico real; MPS: memoria asignada actual). None si no hay GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 2**20
    except Exception:
        pass
    return None


def _reset_gpu_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class EpochLogger:
    """
    Registro por epoch (sección 4). Uso en el ciclo de entrenamiento:

        log = EpochLogger(name="sgns_d100")
        for epoch in range(1, n_epochs + 1):
            log.begin_epoch()
            ... entrenar ...
            log.end_epoch(epoch, loss=mean_loss, E=model.in_embed.weight, itos=vocab.itos)
        log.df          # una fila por epoch
        log.summary()   # tiempo total, pico de GPU, última fila
        log.save("results/sgns_d100.json")

    El tiempo por epoch NO incluye el tiempo de evaluación (se reporta aparte en eval_s).
    """

    def __init__(self, name: str = "", questions=None, wordsim=None, control_words=CONTROL_WORDS,
                 restrict_vocab: int | None = 30000, config: dict | None = None, verbose: bool = True,
                 track_gpu: bool = True):
        self.name, self.config, self.verbose, self.track_gpu = name, dict(config or {}), verbose, track_gpu
        self.questions = questions if questions is not None else load_analogies()
        self.wordsim = wordsim if wordsim is not None else load_word_pairs("wordsim353.tsv")
        self.control_words, self.restrict_vocab = list(control_words), restrict_vocab
        self.history, self.neighbors_by_epoch = [], {}
        self._t_start = None
        self._t_epoch = None

    def begin_epoch(self):
        if self._t_start is None:
            self._t_start = time.perf_counter()
        if self.track_gpu:
            _reset_gpu_peak()
        self._t_epoch = time.perf_counter()

    def end_epoch(self, epoch: int, loss: float | None = None, E=None, itos=None, word2idx=None,
                  space: EmbeddingSpace | None = None, **extra):
        train_s = time.perf_counter() - self._t_epoch
        gpu_mb = peak_gpu_mb() if self.track_gpu else None
        t0 = time.perf_counter()
        if space is None:
            space = EmbeddingSpace(E, word2idx or {w: i for i, w in enumerate(itos)}, name=self.name)
        an = evaluate_analogies(space, self.questions, restrict_vocab=self.restrict_vocab)
        ws = word_similarity(space, self.wordsim)
        nb = neighbors(space, self.control_words, k=5)
        eval_s = time.perf_counter() - t0
        row = dict(epoch=epoch, loss=loss,
                   acc_sem=an["add"]["semantic"], acc_syn=an["add"]["syntactic"],
                   acc_total=an["add"]["total"], acc_total_mul=an["mul"]["total"],
                   coverage=an["coverage"], wordsim353=ws["spearman"],
                   epoch_s=train_s, eval_s=eval_s, gpu_peak_mb=gpu_mb, **extra)
        self.history.append(row)
        self.neighbors_by_epoch[epoch] = nb
        if self.verbose:
            print(f"[{self.name}] epoch {epoch:>2} | loss {loss if loss is None else round(loss, 4)} | "
                  f"analogías sem {row['acc_sem']:.3f} sin {row['acc_syn']:.3f} total {row['acc_total']:.3f} | "
                  f"WS353 {row['wordsim353']:.3f} | {train_s:.0f}s (+{eval_s:.1f}s eval)")
        return row

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def summary(self) -> dict:
        df = self.df
        gpu = df["gpu_peak_mb"].dropna()
        return dict(name=self.name, config=self.config, epochs=len(df),
                    total_train_s=float(df["epoch_s"].sum()),
                    gpu_peak_mb=float(gpu.max()) if len(gpu) else None,
                    last=df.iloc[-1].to_dict() if len(df) else None)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(summary=self.summary(), history=self.history,
                       neighbors={str(k): v for k, v in self.neighbors_by_epoch.items()}),
                  open(path, "w"), indent=1, default=float)


# Pérdida SGNS comparable entre implementaciones
class SGNSLossProbe:
    """
    Pérdida SGNS promedio por par (centro, contexto) sobre una muestra FIJA de pares y de negativos:

        L = -log σ(v_ctx · u_centro) - Σ_{j=1..k} log σ(-v_neg_j · u_centro)

    Sirve para dar a gensim una pérdida comparable con la del SGNS propio: el acumulador de pérdida de
    gensim es float32 y pierde precisión conforme crece (su pérdida por epoch deja de ser confiable).
    Los pares salen de los primeros `max_tokens` tokens del corpus codificado (ventana dinámica) y los
    negativos de la distribución unigrama^0.75, igual que en el entrenamiento.

        probe = SGNSLossProbe(ids, offsets, counts, window=5, k=5)
        probe.loss(W_in, W_out)                       # filas alineadas con los ids del corpus
        probe.loss(W_in, W_out, rows=ids_a_filas)     # rows[id_corpus] = fila en la matriz (-1 si no está)
    """

    def __init__(self, ids, offsets, counts, window: int = 5, k: int = 5, n_pairs: int = 200_000,
                 max_tokens: int = 1_000_000, seed: int = 0):
        import utils as U
        rng = np.random.default_rng(seed)
        p1 = int(np.searchsorted(offsets, max_tokens, side="right")) - 1
        sub_ids, sub_off = ids[:offsets[p1]], offsets[:p1 + 1]
        cs, xs = zip(*U.iter_pair_chunks(sub_ids, sub_off, window, dynamic=True,
                                         chunk_tokens=len(sub_ids) + 1, rng=rng))
        cs, xs = np.concatenate(cs), np.concatenate(xs)
        sel = rng.choice(len(cs), size=min(n_pairs, len(cs)), replace=False)
        self.centers, self.contexts, self.k = cs[sel], xs[sel], k
        p = np.asarray(counts, dtype=np.float64) ** 0.75
        p[:2] = 0.0                                               # nunca <pad>/<unk> como negativo
        self.negatives = rng.choice(len(p), size=(len(sel), k), p=p / p.sum()).astype(np.int32)

    def loss(self, W_in, W_out, rows=None, chunk: int = 50_000) -> float:
        W_in = W_in.detach().cpu().numpy() if hasattr(W_in, "detach") else np.asarray(W_in)
        W_out = W_out.detach().cpu().numpy() if hasattr(W_out, "detach") else np.asarray(W_out)
        m = (lambda a: a) if rows is None else (lambda a: rows[a])
        total = 0.0
        with np.errstate(all="ignore"):
            for s in range(0, len(self.centers), chunk):
                u = W_in[m(self.centers[s:s + chunk])]
                pos = (u * W_out[m(self.contexts[s:s + chunk])]).sum(1)
                neg = np.einsum("nd,nkd->nk", u, W_out[m(self.negatives[s:s + chunk])])
                total += np.logaddexp(0, -pos).sum() + np.logaddexp(0, neg).sum()
        return float(total / len(self.centers))


class _NullStream:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


@contextlib.contextmanager
def mute_stderr():
    """
    Silencia stderr (el descriptor de C y sys.stderr) mientras dura el bloque. gensim 4.4 + Cython 3
    imprime «Exception ignored in ...our_dot_float» (sin tipo ni traceback) cientos de veces durante
    train(); no afecta los resultados (las métricas y la pérdida mejoran con normalidad). Una excepción
    real de Python dentro del bloque sigue propagándose y mostrándose al salir.
    """
    py_stderr, saved, devnull = sys.stderr, os.dup(2), os.open(os.devnull, os.O_WRONLY)
    sys.stderr = _NullStream()
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        sys.stderr = py_stderr
        os.close(saved), os.close(devnull)


mute_c_stderr = mute_stderr   # nombre anterior


def _gensim_callback_base():
    from gensim.models.callbacks import CallbackAny2Vec
    return CallbackAny2Vec


class GensimEpochCallback(_gensim_callback_base()):
    """
    Callback para Word2Vec de gensim con el mismo registro que EpochLogger:

        cb = GensimEpochCallback(EpochLogger(name="gensim"), probe=probe, rows=rows)
        w2v = Word2Vec(...)
        with mute_stderr():
            w2v.train(..., callbacks=[cb])

    La pérdida se calcula con SGNSLossProbe sobre (wv.vectors, syn1neg), así es comparable con la del
    SGNS propio. `rows[id_corpus]` = fila del vocabulario de gensim para ese id (-1 si no está).
    """

    def __init__(self, logger: EpochLogger, probe: SGNSLossProbe | None = None, rows=None):
        self.logger, self.probe, self.rows, self.epoch = logger, probe, rows, 0

    def on_epoch_begin(self, model):
        self.logger.begin_epoch()

    def on_epoch_end(self, model):
        self.epoch += 1
        t_train = time.perf_counter() - self.logger._t_epoch
        loss = self.probe.loss(model.wv.vectors, model.syn1neg, self.rows) if self.probe else None
        self.logger._t_epoch = time.perf_counter() - t_train      # el cálculo de la pérdida no cuenta como entrenamiento
        self.logger.end_epoch(self.epoch, loss=loss,
                              space=EmbeddingSpace.from_keyed_vectors(model.wv, name=self.logger.name))
