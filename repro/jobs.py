"""Every job behind the paper: its command, its output folder and the jobs it needs first.

    python -m repro.jobs               # list every job in run order, its pool and what it needs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from wm import experiments as ex
from wm.config import DiagCfg

ROOT = Path(__file__).resolve().parent.parent
QMIX_SEEDS = tuple(f"QMIX-s{s}" for s in ex.SEEDS)


@dataclass
class Job:
    name: str
    cmd: list[str]
    out: str  # output directory, relative to the results root
    needs: tuple[str, ...] = ()
    pool: str = "gpu"  # "gpu" | "cpu": which slots it takes in repro.run


def job(name: str, mock: bool = False, device: str = "cuda") -> Job:
    py, flag = [sys.executable, "-m"], ["--mock"] if mock else []
    run = lambda mod: [*py, mod, name, *flag, f"device={'cpu' if mock else device}"]
    if name in QMIX_SEEDS:
        s = name[-1]
        return Job(name, [*py, "repro.train_qmix", s, *flag], f"{ex.PACKAGES['QMIX']}/qmix-s{s}")
    e = ex.REGISTRY[name]
    exp, oracle = e["exp"], e.get("oracle", {})
    out = f"{ex.PACKAGES[exp]}/{name}"
    if exp == "SUITE":
        return Job(name, run("wm.suite"), out)
    if name == "WX-eval":
        trained = ("WX-sector-s0", "WX-sector-s1", "WX-sector-s2", "WX-local-s0")
        return Job(name, [*py, "wm.weather", name, *flag], out, trained, "cpu")
    if exp == "WX":
        return Job(name, [*py, "wm.weather", name, *flag], out, pool="cpu")
    if exp == "DATA":
        return Job(name, run("wm.data"), "data", pool="cpu")
    if name == "RANK":
        return Job(name, [*py, "wm.ranking", name, *flag], out, ("E4-s0", "WX-sector-s0"), "cpu")
    if name == "E4-s0-fly":
        return Job(name, run("wm.eval"), out, ("E4-s0",))
    if exp == "E4":
        return Job(name, run("wm.train"), out, ("DATA-avoid",))
    needs = (Path(oracle["weather_ckpt"]).parent.name,) if oracle.get("weather_ckpt") else ()
    if e.get("policy") == "marl":
        needs += (f"QMIX-s{Path(e['checkpoint']).name[-1]}",)
    return Job(name, run("wm.eval"), out, needs, "cpu" if e.get("policy") in ("direct", "avoid") else "gpu")


def names() -> list[str]:
    """Every job, each after the jobs it needs."""
    todo, order = [*QMIX_SEEDS, *ex.REGISTRY], []
    while todo:
        ready = next((n for n in todo if all(d in order for d in job(n).needs)), None)
        if ready is None:
            raise RuntimeError(f"unmet dependencies: {todo}")
        order.append(ready)
        todo.remove(ready)
    return order


def results_root(mock: bool = False) -> Path:
    return ROOT / DiagCfg().out / ("mock" if mock else "")


if __name__ == "__main__":
    for n in names():
        j = job(n)
        print(f"{n:26s} {j.pool:4s} {j.out:44s} needs {', '.join(j.needs) or '-'}")
