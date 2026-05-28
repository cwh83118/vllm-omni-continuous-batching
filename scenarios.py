"""Concurrency scenarios for the cockpit dual-stream benchmark.

A scenario is a recipe for which arrival generators run, for how long, with
what parameters. cabin_demo.py reads `--scenario <name>` and looks up the
corresponding ScenarioSpec here.

Five presets (matching plan §A.5):

  pure_proactive : every 2.5s proactive ticks for 24 s. No agent.
                   Establishes audio-prefill baseline.

  pure_agent     : no proactive. 5 agent tasks starting at t = 0/3/6/9/12 s.

  mixed_1agent   : proactive every 2.5s for 30 s + 1 agent task at t=2.

  mixed_3agent   : proactive every 2.5s for 30 s + 3 agent tasks at t=2/8/15.
                   This is the demo headline scenario.

  burst24        : 24 text-only single-turn interactive requests arriving at
                   t=1s. Reuses the existing burst_arrivals code path; kept for
                   backward compatibility with results/burst_*.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentLaunch:
    start_time_s: float
    task_idx: int    # which of the 10 interactive task scripts to use


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    duration_s: float
    proactive_enabled: bool
    proactive_interval_s: float = 2.5
    proactive_max_tokens: int = 220
    agent_launches: tuple[AgentLaunch, ...] = field(default_factory=tuple)
    burst_n: int = 0                 # if >0, run text-only single-turn burst (no proactive, no agent)
    burst_max_tokens: int = 180


SCENARIOS: dict[str, ScenarioSpec] = {
    "pure_proactive": ScenarioSpec(
        name="pure_proactive",
        duration_s=24.0,
        proactive_enabled=True,
        proactive_interval_s=2.5,
        proactive_max_tokens=220,
    ),
    "pure_agent": ScenarioSpec(
        name="pure_agent",
        duration_s=25.0,
        proactive_enabled=False,
        agent_launches=tuple(
            AgentLaunch(start_time_s=float(s), task_idx=i)
            for i, s in enumerate([0.5, 3.5, 6.5, 9.5, 12.5])
        ),
    ),
    "mixed_1agent": ScenarioSpec(
        name="mixed_1agent",
        duration_s=30.0,
        proactive_enabled=True,
        proactive_interval_s=2.5,
        proactive_max_tokens=220,
        agent_launches=(AgentLaunch(start_time_s=2.0, task_idx=0),),
    ),
    "mixed_3agent": ScenarioSpec(
        name="mixed_3agent",
        duration_s=30.0,
        proactive_enabled=True,
        proactive_interval_s=2.5,
        proactive_max_tokens=220,
        agent_launches=(
            AgentLaunch(start_time_s=2.0, task_idx=0),
            AgentLaunch(start_time_s=8.0, task_idx=3),
            AgentLaunch(start_time_s=15.0, task_idx=7),
        ),
    ),
    "burst24": ScenarioSpec(
        name="burst24",
        duration_s=30.0,
        proactive_enabled=False,
        burst_n=24,
        burst_max_tokens=180,
    ),
}


def get(name: str) -> ScenarioSpec:
    try:
        return SCENARIOS[name]
    except KeyError as e:
        raise SystemExit(f"unknown --scenario {name!r}; pick from {list(SCENARIOS)}") from e
