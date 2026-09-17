# Journal v1 protocol

This branch intentionally stops treating the SIGIR conference instance generator as normative. The goal is a scientifically defensible journal benchmark for **social recruitment followed by finite-capacity worker-task allocation**.

## Problem

1. Select a global seed set S from candidate recruiters.
2. Recruit additional participants through a global Independent Cascade social diffusion process.
3. Given the realized active set A(S,w), allocate active workers to spatial tasks with capacity constraint `sum_i x_ui <= c_u` (default c_u=1).
4. Task i receives effective sensing capacity `sum_u x_ui * accessibility_ui * quality_ui` and satisfaction `min(capacity_i / demand_i, 1)`.
5. Optimize expected mean task satisfaction over diffusion worlds.

The finite-capacity assignment stage removes the unrealistic relaxation in which every activated worker contributes simultaneously to every task.

## Data construction

- SNAP Gowalla / Brightkite raw check-ins and friendship edges.
- Same geographic rectangle as the conference study for continuity, but the journal protocol is not designed to reproduce conference numbers.
- Per-user 90/10 temporal split. Training history constructs social/task features; future check-ins determine task demand.
- Users must have at least 10 training check-ins and are sampled from a connected eligible social component.
- Candidate worker pool: 300 workers sampled across activity strata.
- Tasks are selected only from training cells with at least 20 check-ins and 5 unique users.
- Main layout: spatially scattered, popularity-aware farthest-point sampling. Clustered and contiguous layouts are supported for sensitivity tests.
- Accessibility: `exp(-minimum historical haversine distance / 10 km)`.
- Quality: `1-exp(-historical visits within 5 km / 3)`.
- Demand: future unique-user density, monotonically mapped to `[3,10]` in the main workload.
- Social influence: asymmetric global campaign diffusion using sender/receiver activity, home-location similarity, and neighbor Jaccard; probabilities clipped to `[0.01,0.20]`.

## Evaluation

Primary metrics:

- mean task satisfaction;
- P10 task satisfaction;
- unsatisfied-task ratio;
- saturated-task ratio;
- online selection time;
- offline preprocessing/training time.

Every generated instance should be rejected or redesigned if trivial heuristics saturate the objective at very small budgets. `journal_sanity.py` explicitly reports this diagnostic together with the unrealistic parallel-service relaxation.

## Current journal-v1 model scaffold

The Rainbow selector and gated dual-input architecture are retained. During formulation validation, the social input is a transparent structural graph embedding and the task input is worker-task service potential normalized by demand. This is intentionally a scaffold rather than a final representation-learning claim. Once the benchmark is validated, the social representation learner and task-side encoder can be upgraded without changing the evaluation protocol.
