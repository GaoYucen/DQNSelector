Method

- DegGreedy: sort based on degree
- Cov: sort by one-step coverage capability
- FastSelector: heuristic 
- KTVoting: heuristic
- PIANO: https://github.com/lihuixidian/PIANO
- CELF: based on marginal effects

Dataset：

- Edge: source, target, probability

- nodes: id, coverage ability for subareas



To run baselines:

```
cd code
python run.py
```

To run DQNSelector:

```
cd code
python DQNSelector.py
```

