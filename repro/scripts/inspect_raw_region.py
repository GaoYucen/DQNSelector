#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

LON_MIN, LON_MAX = -122.50, -118.00
LAT_MIN, LAT_MAX = 33.80, 37.90


def op(path):
    p = Path(path)
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.suffix == ".gz" else p.open("r", encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True)
    p.add_argument("--checkins", required=True)
    a = p.parse_args()
    region_users = set()
    records = 0
    with op(a.checkins) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            x = line.split()
            if len(x) < 4:
                continue
            try:
                user = int(x[0]); lat = float(x[2]); lon = float(x[3])
            except ValueError:
                continue
            if LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX:
                region_users.add(user); records += 1
    edge_nodes = set(); region_edges = 0; all_edges = 0
    with op(a.edges) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            x = line.split()
            if len(x) < 2:
                continue
            u, v = int(x[0]), int(x[1]); all_edges += 1
            edge_nodes.add(u); edge_nodes.add(v)
            if u in region_users and v in region_users:
                region_edges += 1
    eligible = region_users & edge_nodes
    print(f"region_users={len(region_users)}")
    print(f"region_users_with_social_graph={len(eligible)}")
    print(f"region_records={records}")
    print(f"source_edges={all_edges}")
    print(f"region_induced_source_edges={region_edges}")


if __name__ == "__main__":
    main()
