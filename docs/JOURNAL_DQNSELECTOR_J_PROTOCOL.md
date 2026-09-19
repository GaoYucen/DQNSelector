# DQNSelector-J journal protocol

`DQNSelector` remains the frozen v3 reference. `DQNSelector-J` is a separately
named residual-demand, teacher-regularized Rainbow policy; its checkpoints are
not compatible with the conference model.

The scenario grid uses only observable social structure and historical check-in
proxies. The sixteen configurations are selected on development seeds only.
The selected primary scenario must be frozen before test seeds are built, and
all sixteen development results are retained.

The main table reports the repaired five non-CELF baselines, CELF, frozen
DQNSelector, and DQNSelector-J. PIANO checkpoint selection uses held-out
influence spread, never final EC. KTVoting2-feasible is the main released-code
adapter; the older grouped repair is appendix-only.

Success is evaluated independently for Gowalla/Brightkite and 3000/5000 users:
at least four of six budgets must have mean DQNSelector-J relative gain of 5%
or more over the strongest non-CELF baseline, with a positive simultaneous
paired-bootstrap lower bound.
