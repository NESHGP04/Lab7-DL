"""
Carga el corpus ya preprocesado por la sección 2 (cache/vocab.json, train_offsets.npy, train_ids.npy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import utils as U


def load_corpus_cache(cache: Path | str = "cache", verbose: bool = True):
    """Devuelve (itos, counts, ids, offsets, meta). Párrafo p = ids[offsets[p]:offsets[p+1]]."""
    cache = Path(cache)
    meta = json.load(open(cache / "vocab.json"))
    itos, counts = meta["itos"], np.array(meta["counts"], dtype=np.int64)
    offsets = np.load(cache / "train_offsets.npy")
    ids_path = cache / "train_ids.npy"

    if not ids_path.exists():
        from datasets import load_dataset
        if verbose:
            print("cache/train_ids.npy no existe: regenerando con utils.py (≈1-2 min)...")
        rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"]
        counter, _ = U.count_words(U.iter_paragraphs(rows, meta["max_tokens"]))
        vocab = U.Vocab(counter, min_count=meta["min_count"])
        assert vocab.itos == itos, "el vocabulario regenerado no coincide con cache/vocab.json"
        ids, new_offsets = U.encode_corpus(U.iter_paragraphs(rows, meta["max_tokens"]), vocab, drop_unk=True)
        assert np.array_equal(new_offsets, offsets), "los offsets regenerados no coinciden con cache/train_offsets.npy"
        np.save(ids_path, ids)
    ids = np.load(ids_path)
    assert offsets[-1] == len(ids)
    if verbose:
        print(f"Corpus: {len(ids):,} tokens en {len(offsets)-1:,} párrafos | vocabulario: {len(itos):,} (incluye <pad>, <unk>)")
    return itos, counts, ids, offsets, meta
