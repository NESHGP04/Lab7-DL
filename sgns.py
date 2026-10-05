"""
Skip-gram con negative sampling (SGNS) desde cero en PyTorch (Lab 7, sección 4).
No usa gensim.models.Word2Vec. Solo usa gensim para localizar los archivos de
evaluación (questions-words.txt, wordsim353.tsv).
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

import nlp_utils as U

CONTROL_WORDS = ("king", "france", "computer", "good", "january", "run")

# Configuración
@dataclass
class Config:
    name: str = "sgns"
    dim: int = 100
    window: int = 5
    neg: int = 5                 # negativos por par positivo
    subsample_t: float = 1e-4    # None = sin submuestreo
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 3e-3             # SparseAdam; decae linealmente hasta lr * min_lr_frac
    min_lr_frac: float = 0.01
    corpus_frac: float = 1.0     # fracción (prefijo) del subconjunto de trabajo
    chunk_tokens: int = 2_000_000
    seed: int = 42

# Modelo
class SGNS(nn.Module):
    """Dos tablas: W (entrada, vectores de la palabra central) y W' (salida, contexto)."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim, sparse=True)
        self.out_emb = nn.Embedding(vocab_size, dim, sparse=True)
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, pos, neg):
        """
        center, pos: (B,)   neg: (B, K)
        L = -[ log σ(u_o·v_c) + Σ_k log σ(-u_k·v_c) ], promediada sobre el batch.
        """
        v = self.in_emb(center)                              # (B, d)
        u_pos = self.out_emb(pos)                            # (B, d)
        u_neg = self.out_emb(neg)                            # (B, K, d)
        pos_score = (v * u_pos).sum(-1)                      # (B,)
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)  # (B, K)
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1))
        return loss.mean()


def build_neg_table(vocab, table_size: int = 10_000_000, power: float = 0.75):
    """Tabla para muestrear negativos con P(w) ∝ count(w)^0.75 (excluye <pad>/<unk>)."""
    p = vocab.counts.astype(np.float64).copy()
    p[[U.PAD_ID, U.UNK_ID]] = 0.0
    p = p ** power
    p /= p.sum()
    reps = np.round(p * table_size).astype(np.int64)
    return np.repeat(np.arange(len(p), dtype=np.int64), reps)


def take_fraction(ids, offsets, frac: float):
    """Prefijo del corpus con ~frac de los tokens (corta en límite de párrafo)."""
    if frac >= 1.0:
        return ids, offsets
    p = int(np.searchsorted(offsets, frac * offsets[-1], side="right")) - 1
    p = max(p, 1)
    return ids[:offsets[p]], offsets[:p + 1]

# Evaluación (rápida, en GPU) para monitorear cada epoch.
class Evaluator:
    def __init__(self, vocab, analogies_path, wordsim_path, restrict=30000,
                 controls=CONTROL_WORDS, device="cpu"):
        self.vocab, self.restrict, self.device = vocab, restrict, device
        self.sections, self.totals = self._load_analogies(analogies_path)
        self.ws_i, self.ws_j, self.ws_score, self.ws_total = self._load_wordsim(wordsim_path)
        self.controls = [w for w in controls if w in vocab.stoi]

    def _load_analogies(self, path):
        sections, totals, cur = {}, {}, None
        lim = self.restrict + 2  
        for line in open(path, encoding="utf8"):
            line = line.strip().lower()
            if line.startswith(":"):
                cur = line[1:].strip()
                sections[cur], totals[cur] = [], 0
                continue
            w = line.split()
            if len(w) != 4 or cur is None:
                continue
            totals[cur] += 1
            ids = [self.vocab.stoi.get(x) for x in w]
            if all(i is not None and 2 <= i < lim for i in ids):
                sections[cur].append(ids)
        sections = {k: torch.tensor(v, dtype=torch.long, device=self.device)
                    for k, v in sections.items() if len(v)}
        return sections, totals

    def _load_wordsim(self, path):
        a, b, s, total = [], [], [], 0
        for line in open(path, encoding="utf8"):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            total += 1
            i, j = self.vocab.stoi.get(parts[0].lower()), self.vocab.stoi.get(parts[1].lower())
            if i is not None and j is not None and i >= 2 and j >= 2:
                a.append(i); b.append(j); s.append(score)
        return (torch.tensor(a, device=self.device), torch.tensor(b, device=self.device),
                np.array(s), total)

    @torch.no_grad()
    def __call__(self, W):
        W = W.detach().float()
        res = {}
        E = F.normalize(W[2:self.restrict + 2], dim=1)
        per_sec, tot = {}, {"sem": [0, 0], "syn": [0, 0]}
        for name, q in self.sections.items():
            correct = 0
            for s in range(0, len(q), 1024):
                qq = q[s:s + 1024] - 2
                target = E[qq[:, 1]] - E[qq[:, 0]] + E[qq[:, 2]]
                sims = target @ E.T
                sims.scatter_(1, qq[:, :3], float("-inf"))   # excluye a, b y c
                correct += (sims.argmax(1) == qq[:, 3]).sum().item()
            per_sec[name] = correct / len(q)
            kind = "syn" if name.startswith("gram") else "sem"
            tot[kind][0] += correct; tot[kind][1] += len(q)
        n_eval = tot["sem"][1] + tot["syn"][1]
        res["analogy_sem"] = tot["sem"][0] / max(tot["sem"][1], 1)
        res["analogy_syn"] = tot["syn"][0] / max(tot["syn"][1], 1)
        res["analogy_total"] = (tot["sem"][0] + tot["syn"][0]) / max(n_eval, 1)
        res["analogy_coverage"] = n_eval / sum(self.totals.values())
        res["analogy_per_section"] = per_sec
        Wn = F.normalize(W, dim=1)
        cos = (Wn[self.ws_i] * Wn[self.ws_j]).sum(-1).cpu().numpy()
        res["wordsim353"] = float(spearmanr(cos, self.ws_score)[0])
        res["wordsim_coverage"] = len(self.ws_score) / self.ws_total
        Ef = F.normalize(W[2:], dim=1)
        nn_ = {}
        for w in self.controls:
            i = self.vocab.stoi[w] - 2
            top = (Ef @ Ef[i]).topk(6).indices.tolist()
            nn_[w] = [self.vocab.itos[j + 2] for j in top if j != i][:5]
        res["neighbors"] = nn_
        return res

# Entrenamiento
def train_sgns(ids, offsets, vocab, cfg: Config, evaluator=None, device=None,
               log_every: int = 200, verbose: bool = True):
    """
    Devuelve (modelo, historial). `ids/offsets` ya son el corpus (usar take_fraction antes).
    Cada epoch: nuevo submuestreo -> pares (ventana dinámica) -> barajado por bloque -> batches.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = SGNS(len(vocab), cfg.dim).to(device)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=cfg.lr)
    table = torch.from_numpy(build_neg_table(vocab)).to(device)
    keep = (U.keep_probs(vocab, t=cfg.subsample_t, formula="gensim")
            if cfg.subsample_t else np.where(np.arange(len(vocab)) >= 2, 1.0, 0.0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = dict(config=asdict(cfg), n_tokens=int(len(ids)), vocab_size=len(vocab),
                   device=str(device), epochs=[], loss_trace=[])
    total_pairs_est, pairs_seen, step, train_time = None, 0, 0, 0.0

    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        sub_ids, sub_off = U.subsample(ids, offsets, keep, rng)
        if total_pairs_est is None:
            total_pairs_est = U.count_skipgram_pairs(sub_off, cfg.window, dynamic=True) * cfg.epochs
        run = torch.zeros((), device=device)
        run_n, ep_sum, ep_n, ep_pairs = 0, 0.0, 0, 0
        model.train()
        for c_np, x_np in U.iter_pair_chunks(sub_ids, sub_off, cfg.window, dynamic=True,
                                              chunk_tokens=cfg.chunk_tokens, rng=rng):
            if len(c_np) == 0:
                continue
            c = torch.from_numpy(c_np).to(device).long()
            x = torch.from_numpy(x_np).to(device).long()
            perm = torch.randperm(len(c), device=device)
            c, x = c[perm], x[perm]
            for i in range(0, len(c), cfg.batch_size):
                cb, xb = c[i:i + cfg.batch_size], x[i:i + cfg.batch_size]
                neg = table[torch.randint(0, table.numel(), (len(cb), cfg.neg), device=device)]
                lr = max(cfg.lr * cfg.min_lr_frac, cfg.lr * (1 - pairs_seen / total_pairs_est))
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss = model(cb, xb, neg)
                loss.backward()
                opt.step()
                pairs_seen += len(cb); ep_pairs += len(cb); step += 1
                run += loss.detach(); run_n += 1
                if run_n == log_every:
                    v = run.item() / run_n
                    history["loss_trace"].append((step, v))
                    ep_sum += v * run_n; ep_n += run_n
                    run.zero_(); run_n = 0
        if run_n:
            v = run.item() / run_n
            history["loss_trace"].append((step, v))
            ep_sum += v * run_n; ep_n += run_n
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time

        entry = dict(epoch=epoch, loss=ep_sum / max(ep_n, 1), lr_final=lr, pairs=ep_pairs,
                     time_s=ep_time,
                     peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2**20
                                  if device.type == "cuda" else None))
        if evaluator is not None:
            entry.update(evaluator(model.in_emb.weight))
        history["epochs"].append(entry)
        if verbose:
            msg = f"[{cfg.name}] epoch {epoch}/{cfg.epochs}  loss {entry['loss']:.4f}  {ep_time:.0f}s"
            if evaluator is not None:
                msg += (f"  analogías sem/syn/total "
                        f"{entry['analogy_sem']:.3f}/{entry['analogy_syn']:.3f}/{entry['analogy_total']:.3f}"
                        f"  WS353 {entry['wordsim353']:.3f}")
            print(msg)
            if evaluator is not None:
                for w, nbrs in entry["neighbors"].items():
                    print(f"      {w:>9}: {', '.join(nbrs)}")
    history["train_time_s"] = train_time
    return model, history

# Guardado 
def save_run(model, history, out_dir="cache/runs"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    name = history["config"]["name"]
    np.save(out / f"{name}_W_in.npy", model.in_emb.weight.detach().cpu().numpy().astype(np.float32))
    np.save(out / f"{name}_W_out.npy", model.out_emb.weight.detach().cpu().numpy().astype(np.float32))
    json.dump(history, open(out / f"{name}.json", "w"), indent=1)


def load_history(name, out_dir="cache/runs"):
    return json.load(open(Path(out_dir) / f"{name}.json"))