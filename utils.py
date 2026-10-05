"""
Utilidades de preprocesamiento para el Lab 7 (CC3092).

Se usan en la sección 2 (WikiText-103), en el entrenamiento de SGNS (sección 4)
y en AG News (sección 6): el MISMO normalizador y tokenizador en todas partes.
"""
import html
import json
import re
import unicodedata
from array import array
from collections import Counter

import numpy as np

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# Estructura de WikiText
_ARTICLE_RE = re.compile(r"^= [^=].*[^=] =$")  # solo títulos de nivel 1


def is_article_title(line: str) -> bool:
    """' = Título = ' (nivel 1) marca el inicio de un artículo."""
    return bool(_ARTICLE_RE.match(line.strip()))


def is_heading(line: str) -> bool:
    """Cualquier encabezado (' = = Sección = = ', etc.)."""
    s = line.strip()
    return s.startswith("=") and s.endswith("=") and len(s) > 2

# Normalización y tokenización
_AT_RE = re.compile(r" @([-,.])@ ")  
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

_CLITIC = r"(?:s|t|re|ve|ll|d|m)\b"
_TOKEN_RE = re.compile(
    r"\d+(?::\d+)+"                                    
    r"|[^\W_]+(?:(?:[-.,]|'(?!" + _CLITIC + r"))[^\W_]+)*"  
    r"|'" + _CLITIC +                                  
    r"|[^\w\s]"                                        
)


def normalize(text: str) -> str:
    """Minúsculas + desescape HTML + quita artefactos @-@ de WikiText."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    return _AT_RE.sub(r"\1", text)


def tokenize(text: str, keep_punct: bool = False) -> list[str]:
    toks = _TOKEN_RE.findall(normalize(text))
    if keep_punct:
        return toks
    return [t for t in toks if any(c.isalnum() for c in t)]

# Lectura del corpus
def corpus_stats(rows) -> dict:
    """Filas, filas no vacías, artículos y tokens (split por espacios) crudos."""
    n_rows = n_nonempty = n_articles = n_tokens = 0
    for line in rows:
        n_rows += 1
        s = line.strip()
        if not s:
            continue
        n_nonempty += 1
        n_articles += is_article_title(s)
        n_tokens += len(s.split())
    return dict(filas=n_rows, filas_no_vacias=n_nonempty,
                articulos=n_articles, tokens_crudos=n_tokens)


def iter_paragraphs(rows, max_tokens=None):
    """
    Devuelve listas de tokens, un párrafo por elemento (sin títulos ni vacíos).
    max_tokens: corta en un límite de párrafo al llegar a ese número de tokens
    (prefijo del corpus => el subconjunto es idéntico para todos los modelos
    y los subconjuntos de 25/50/100 % quedan anidados).
    """
    total = 0
    for line in rows:
        if not line.strip() or is_heading(line):
            continue
        toks = tokenize(line)
        if not toks:
            continue
        total += len(toks)
        yield toks
        if max_tokens is not None and total >= max_tokens:
            return


def count_words(paragraphs) -> tuple[Counter, int]:
    c = Counter()
    n = 0
    for toks in paragraphs:
        c.update(toks)
        n += len(toks)
    return c, n

# Vocabulario
class Vocab:
    def __init__(self, counter: Counter, min_count: int = 5, max_size=None):
        items = [(w, c) for w, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size:
            items = items[:max_size]
        self.min_count = min_count
        self.itos = [PAD, UNK] + [w for w, _ in items]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([0, 0] + [c for _, c in items], dtype=np.int64)
        self.total_tokens = int(sum(counter.values()))
        self.in_vocab_total = int(self.counts.sum())
        self.counts[UNK_ID] = self.total_tokens - self.in_vocab_total
        self.unk_pct = 100 * self.counts[UNK_ID] / self.total_tokens

    def __len__(self):
        return len(self.itos)

    def encode(self, toks, drop_unk=False):
        ids = [self.stoi.get(t, UNK_ID) for t in toks]
        return [i for i in ids if i != UNK_ID] if drop_unk else ids


def load_vocab(path) -> "Vocab":
    """Reconstruye el Vocab guardado en cache/vocab.json por el notebook de la sección 2."""
    d = json.load(open(path))
    v = object.__new__(Vocab)
    v.min_count = d["min_count"]
    v.itos = d["itos"]
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    v.counts = np.array(d["counts"], dtype=np.int64)
    v.in_vocab_total = int(v.counts[2:].sum())
    v.total_tokens = v.in_vocab_total + int(v.counts[UNK_ID])
    v.unk_pct = 100 * v.counts[UNK_ID] / v.total_tokens
    return v


def encode_corpus(paragraphs, vocab: Vocab, drop_unk: bool = True):
    """
    Corpus -> (ids int32 planos, offsets int64). Párrafo p = ids[offsets[p]:offsets[p+1]].
    drop_unk=True elimina las palabras fuera de vocabulario (como hace gensim
    con min_count); se descartan párrafos con < 2 tokens (no generan pares).
    """
    flat, lengths = array("i"), array("q")
    for toks in paragraphs:
        ids = vocab.encode(toks, drop_unk=drop_unk)
        if len(ids) >= 2:
            flat.extend(ids)
            lengths.append(len(ids))
    ids = np.frombuffer(flat, dtype=np.int32).copy()
    offsets = np.concatenate([[0], np.cumsum(np.frombuffer(lengths, dtype=np.int64))])
    return ids, offsets

# Estadística del corpus
def coverage(counter: Counter, ks=(10, 1000, 30000)) -> dict:
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=np.float64)
    cum = np.cumsum(freqs) / freqs.sum()
    return {k: 100 * cum[min(k, len(cum)) - 1] for k in ks}


def zipf_fit(counter: Counter, lo: int = 10, hi: int = 10_000):
    """Pendiente e intercepto de log(freq) ~ log(rango) en el rango [lo, hi]."""
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=np.float64)
    hi = min(hi, len(freqs))
    r = np.log(np.arange(lo, hi + 1))
    f = np.log(freqs[lo - 1:hi])
    slope, intercept = np.polyfit(r, f, 1)
    return slope, intercept

# Submuestreo de palabras frecuentes (Word2Vec)
def keep_probs(vocab: Vocab, t: float = 1e-5, formula: str = "gensim") -> np.ndarray:
    """
    Probabilidad de CONSERVAR cada palabra, con f(w) = count(w) / total en vocabulario.
      - "paper":  P(descartar) = 1 - sqrt(t / f)             (Mikolov et al., 2013)
      - "gensim": keep = sqrt(t / f) + t / f                 (código C de word2vec / gensim)
    """
    f = vocab.counts.astype(np.float64) / vocab.in_vocab_total
    with np.errstate(divide="ignore", invalid="ignore"):
        if formula == "paper":
            keep = np.sqrt(t / f)
        elif formula == "gensim":
            keep = np.sqrt(t / f) + t / f
        else:
            raise ValueError(formula)
    keep = np.clip(np.nan_to_num(keep, nan=0.0, posinf=1.0), 0.0, 1.0)
    keep[[PAD_ID, UNK_ID]] = 0.0
    return keep


def subsample(ids, offsets, keep_p, rng=None):
    """Una muestra aleatoria del submuestreo (equivale a un epoch)."""
    rng = rng or np.random.default_rng()
    mask = rng.random(len(ids)) < keep_p[ids]
    cum = np.concatenate([[0], np.cumsum(mask)])
    return ids[mask], cum[offsets]

# Pares skip-gram (palabra central, contexto)
def count_skipgram_pairs(offsets, window: int, dynamic: bool = False) -> float:
    """
    Número de pares sin generarlos. Los pares NO cruzan límites de párrafo.
      fijo:     2 * sum_{d=1..w} sum_p max(n_p - d, 0)
      dinámico: esperanza con ventana efectiva ~ Uniforme{1..w} (como word2vec/gensim)
    """
    n = np.diff(offsets)
    S = np.array([np.maximum(n - d, 0).sum() for d in range(1, window + 1)], dtype=np.float64)
    if not dynamic:
        return 2 * S.sum()
    weights = window - np.arange(window)  # d=1 -> w, ..., d=w -> 1
    return 2 * (S * weights).sum() / window


def _pairs_from_block(ids, offsets, window, dynamic, rng):
    n = len(ids)
    par = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    b = rng.integers(1, window + 1, size=n) if dynamic else np.full(n, window)
    cs, xs = [], []
    for d in range(1, window + 1):
        if d >= n:
            break
        same = par[:-d] == par[d:]
        m1 = same & (b[:-d] >= d)   # centro i, contexto i+d
        cs.append(ids[:-d][m1]); xs.append(ids[d:][m1])
        m2 = same & (b[d:] >= d)    # centro i+d, contexto i
        cs.append(ids[d:][m2]); xs.append(ids[:-d][m2])
    if not cs:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    return np.concatenate(cs), np.concatenate(xs)


def iter_pair_chunks(ids, offsets, window=5, dynamic=True,
                     chunk_tokens=2_000_000, rng=None):
    """
    Generador de (centros, contextos) por bloques de párrafos completos.
    No se materializan todos los pares (serían ~10x el número de tokens).
    """
    rng = rng or np.random.default_rng()
    n_par = len(offsets) - 1
    p0 = 0
    while p0 < n_par:
        p1 = int(np.searchsorted(offsets, offsets[p0] + chunk_tokens, side="right")) - 1
        p1 = min(max(p1, p0 + 1), n_par)
        lo, hi = offsets[p0], offsets[p1]
        yield _pairs_from_block(ids[lo:hi], offsets[p0:p1 + 1] - lo, window, dynamic, rng)
        p0 = p1