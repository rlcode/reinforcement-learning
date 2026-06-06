"""런 I/O — 하네스가 구동·감시·판정할 수 있도록 학습 스크립트가 쓰는 표준 산출물.

하네스(별도 레포)를 import 하지 않는다. 계약(파일 인터페이스)만 네이티브로 구현:
  run-dir/metrics.jsonl   한 줄 = 한 업데이트 (frames·sps·entropy·nan_flag·game_return_mean_lastK …)
  run-dir/ckpt/latest.pt  주기적 원자적 체크포인트 (model+optimizer+정규화기 상태 — 없으면 resume이 거짓말)
  run-dir/final.json      정상 종료 시 게이트 수치 (last-K per-game 학습리턴 평균)

스크립트는 --run-dir 없이도 그대로 standalone 동작한다(이 모듈은 run_dir=None이면 무동작).
"""
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu/mps/cuda 공통 진입점 (MPS는 비트 결정성 없음 — 통계적 재현)


def atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # 부분 쓰기가 latest를 덮지 않게


class RunWriter:
    """run-dir 산출물 작성기. run_dir=None이면 전부 no-op (standalone 실행 보존)."""

    def __init__(self, run_dir: str | None, ckpt_every: int | None = None):
        self.dir = Path(run_dir) if run_dir else None
        self.ckpt_every = ckpt_every
        self.ckpt_dir = (self.dir / "ckpt") if self.dir else None
        self._last_ckpt = 0
        self._last_milestone = 0
        self._t0 = time.time()
        self._last_frames = 0
        self._mf = None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._mf = open(self.dir / "metrics.jsonl", "a", buffering=1)  # 라인 버퍼

    def log(self, frames: int, scalars: dict) -> None:
        if not self._mf:
            return
        now = time.time()
        dt = now - self._t0
        sps = (frames - self._last_frames) / dt if dt > 0 else 0.0
        row = {"ts": round(now, 1), "frames": frames, "sps": round(sps, 1), **scalars}
        self._mf.write(json.dumps(row, default=float) + "\n")
        self._t0, self._last_frames = now, frames

    def maybe_ckpt(self, frames: int, state_fn, gate_metric: float | None = None) -> None:
        """state_fn() → dict (model·optimizer·정규화기·frames). 비싼 직렬화는 저장 시에만."""
        if not self.ckpt_dir or not self.ckpt_every:
            return
        if frames - self._last_ckpt >= self.ckpt_every:
            atomic_save(state_fn(), self.ckpt_dir / "latest.pt")
            self._last_ckpt = frames
        if frames - self._last_milestone >= 5_000_000:  # 마일스톤은 보존(삭제 안 함)
            atomic_save(state_fn(), self.ckpt_dir / f"step_{frames // 1_000_000}M.pt")
            self._last_milestone = frames

    def save_final_ckpt(self, state_fn) -> None:
        if self.ckpt_dir:
            atomic_save(state_fn(), self.ckpt_dir / "latest.pt")

    def final(self, frames: int, game_returns: list, k: int = 100) -> None:
        if not self.dir:
            return
        tail = [float(r) for r in game_returns[-k:]]
        mean = statistics.fmean(tail) if tail else float("nan")
        std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
        (self.dir / "final.json").write_text(json.dumps({
            "frames_total": frames, "frames_unit": "agent_steps",
            "gate_metric": "game_return_mean_lastK", "K": k,
            "value_mean": mean, "value_std": std, "episodes_counted": len(tail),
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=1))
        if self._mf:
            self._mf.close()


def resolve_resume(resume_arg: str | None, run_dir: str | None):
    """--resume auto → run-dir/ckpt/latest.pt. 경로면 그대로. 없으면 None."""
    if not resume_arg:
        return None
    if resume_arg == "auto":
        if run_dir:
            cand = Path(run_dir) / "ckpt" / "latest.pt"
            return cand if cand.exists() else None
        return None
    p = Path(resume_arg)
    return p if p.exists() else None
