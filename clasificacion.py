"""
Clasificación de AG News con embeddings (Lab 7, sección 6): texto -> tokens -> vectores -> modelo -> salida.

Convenciones (acordar con A si usa su propio código):
  - Cada conjunto de embeddings usa SU vocabulario: las palabras fuera de él (OOV) se descartan del documento
    (como hace gensim con min_count). Si un documento queda vacío se usa un único token <unk> (fila 0).
  - La fila 0 de la tabla es <unk>: ceros en la tabla preentrenada, aleatoria en la aleatoria.
  - Embeddings aleatorios: vocabulario de las palabras del train de AG News con >= 2 apariciones.
  - Validación: 10 % del train, estratificado. El test oficial solo se usa en `evaluate_test`, una vez por modelo.
  - Selección: mejor epoch por F1 macro de validación (early stopping, paciencia 3).
  - Se entrena en CPU: la tabla es esparcida (sparse=True) y SparseAdam no corre en MPS.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import utils as U

CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
UNK = 0


# Datos
def load_agnews(val_frac: float = 0.1, seed: int = 42):
    """Devuelve dict con tokens y etiquetas de train / val (10 % estratificado) / test."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    tok = lambda texts: [U.tokenize(t) for t in texts]
    y = np.array(ds["train"]["label"])
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=val_frac, stratify=y, random_state=seed)
    train_tok = tok(ds["train"]["text"])
    return dict(
        train_tok=[train_tok[i] for i in tr_idx], train_y=y[tr_idx],
        val_tok=[train_tok[i] for i in va_idx], val_y=y[va_idx],
        test_tok=tok(ds["test"]["text"]), test_y=np.array(ds["test"]["label"]),
    )


def stratified_fraction(y, frac: float, seed: int = 0):
    """Índices de un subconjunto estratificado con `frac` de los datos (para la curva de F1 vs. fracción)."""
    if frac >= 1.0:
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=seed)
    return np.sort(idx)


def build_vocab(tokens_list, min_count: int = 2):
    """word2idx (fila 0 = <unk>) con las palabras que aparecen >= min_count veces."""
    c = Counter(w for toks in tokens_list for w in toks)
    words = sorted((w for w, n in c.items() if n >= min_count), key=lambda w: (-c[w], w))
    return {w: i + 1 for i, w in enumerate(words)}


def oov_pct(tokens_list, word2idx) -> float:
    """% de tokens (ocurrencias) que no están en el vocabulario."""
    total = out = 0
    for toks in tokens_list:
        total += len(toks)
        out += sum(w not in word2idx for w in toks)
    return 100.0 * out / max(total, 1)


class Encoded:
    """Documentos como ids planos + offsets (formato de nn.EmbeddingBag)."""

    def __init__(self, tokens_list, labels, word2idx):
        docs = []
        for toks in tokens_list:
            ids = [word2idx[w] for w in toks if w in word2idx]
            docs.append(np.array(ids or [UNK], dtype=np.int64))
        self.docs, self.labels = docs, np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.docs)

    def subset(self, idx):
        """Mismo vocabulario, solo los documentos `idx` (para entrenar con una fracción de los datos)."""
        new = object.__new__(Encoded)
        new.docs, new.labels = [self.docs[i] for i in idx], self.labels[np.asarray(idx)]
        return new

    def batch(self, idx):
        parts = [self.docs[i] for i in idx]
        lens = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        return (torch.from_numpy(np.concatenate(parts)), torch.from_numpy(offsets),
                torch.from_numpy(self.labels[idx]))


def embedding_table(space):
    """(matriz con fila 0 = <unk> en ceros, word2idx) a partir de un EmbeddingSpace."""
    E = np.vstack([np.zeros((1, space.dim), dtype=np.float32), space.E])
    return E, {w: i + 1 for i, w in enumerate(space.itos)}


# Modelo
class BagMLP(nn.Module):
    def __init__(self, vocab_size: int, dim: int, hidden: int = 256, n_classes: int = 4, dropout: float = 0.3,
                 pretrained: np.ndarray | None = None, freeze: bool = False):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, dim, mode="mean", sparse=True)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.from_numpy(pretrained))
        self.emb.weight.requires_grad = not freeze
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, ids, offsets):
        return self.mlp(self.emb(ids, offsets))

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        return total, sum(p.numel() for p in self.parameters() if p.requires_grad)


# Métricas y evaluación
def metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return dict(acc=float(accuracy_score(y_true, y_pred)), precision=float(p), recall=float(r), f1=float(f))


@torch.no_grad()
def predict(model, data: Encoded, batch_size: int = 1024):
    model.eval()
    logits, losses = [], 0.0
    for s in range(0, len(data), batch_size):
        ids, off, y = data.batch(np.arange(s, min(s + batch_size, len(data))))
        out = model(ids, off)
        losses += F.cross_entropy(out, y, reduction="sum").item()
        logits.append(out)
    logits = torch.cat(logits)
    return logits.argmax(1).numpy(), losses / len(data)


def evaluate(model, data: Encoded) -> dict:
    pred, loss = predict(model, data)
    return dict(loss=loss, **metrics(data.labels, pred))


def evaluate_test(model, test: Encoded) -> dict:
    """Evaluación final sobre el test oficial (llamar UNA vez por modelo): métricas + matriz de confusión."""
    pred, loss = predict(model, test)
    cm = confusion_matrix(test.labels, pred, labels=range(len(CLASSES)))
    return dict(loss=loss, **metrics(test.labels, pred), confusion=cm.tolist(),
                per_class_f1=precision_recall_fscore_support(test.labels, pred, zero_division=0)[2].tolist())


# Entrenamiento
def train_classifier(train: Encoded, val: Encoded, vocab_size: int, dim: int, pretrained=None, freeze=False,
                     epochs: int = 15, patience: int = 3, batch_size: int = 128, lr: float = 1e-3,
                     emb_lr: float | None = None, seed: int = 42, verbose: bool = True, name: str = ""):
    """
    Entrena BagMLP. Devuelve (modelo con los pesos del mejor epoch, dict con historial y estadísticas).
    El historial guarda por epoch: pérdida de train y validación y métricas de validación.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BagMLP(vocab_size, dim, pretrained=pretrained, freeze=freeze)
    opt_mlp = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    opt_emb = None if freeze else torch.optim.SparseAdam([model.emb.weight], lr=emb_lr or lr)
    total, trainable = model.n_params()

    hist, best, best_state, bad = [], -1.0, None, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(len(train))
        run, n = 0.0, 0
        for s in range(0, len(perm), batch_size):
            ids, off, y = train.batch(perm[s:s + batch_size])
            loss = F.cross_entropy(model(ids, off), y)
            opt_mlp.zero_grad()
            if opt_emb:
                opt_emb.zero_grad()
            loss.backward()
            opt_mlp.step()
            if opt_emb:
                opt_emb.step()
            run += loss.item() * len(y); n += len(y)
        train_s = time.perf_counter() - t0
        v = evaluate(model, val)
        row = dict(epoch=epoch, train_loss=run / n, val_loss=v["loss"], val_acc=v["acc"], val_precision=v["precision"],
                   val_recall=v["recall"], val_f1=v["f1"], epoch_s=train_s)
        hist.append(row)
        if verbose:
            print(f"[{name}] epoch {epoch:>2} | train {row['train_loss']:.4f} val {row['val_loss']:.4f} | "
                  f"val acc {v['acc']:.4f} F1 {v['f1']:.4f} | {train_s:.1f}s")
        if v["f1"] > best:
            best, bad = v["f1"], 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    best_row = max(hist, key=lambda r: r["val_f1"])
    info = dict(name=name, params_total=total, params_trainable=trainable, freeze=freeze,
                train_time_s=time.perf_counter() - t_start, best_epoch=best_row["epoch"], history=hist,
                val=dict(acc=best_row["val_acc"], precision=best_row["val_precision"], recall=best_row["val_recall"],
                         f1=best_row["val_f1"], loss=best_row["val_loss"]),
                config=dict(epochs=epochs, patience=patience, batch_size=batch_size, lr=lr, emb_lr=emb_lr or lr, seed=seed,
                            vocab_size=vocab_size, dim=dim))
    return model, info


def save_result(info: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(path, "w"), indent=1, default=float)
