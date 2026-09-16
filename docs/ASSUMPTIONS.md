# Explicit assumptions and unresolved choices

The clean reproduction separates statements that are specified by the SIGIR paper from choices that are required to make an executable implementation.

## Algorithm choices not fully specified by the paper

The paper names Rainbow components but does not provide all implementation hyperparameters needed for a unique Rainbow implementation. The following are therefore explicit reproduction choices rather than claims about the historical code:

- categorical distributional RL uses a configurable C51 support (`atoms`, `v_min`, `v_max`);
- prioritized replay uses proportional priorities with configurable alpha/beta annealing;
- n-step length is configurable (default 3 in the clean training API, not asserted to be the historical value);
- NoisyNet uses factorized Gaussian noise;
- the variable worker action space is handled by a pair-conditioned Q network `Q(state, candidate)`;
- the dueling advantage is centered over the currently valid candidate worker set;
- the gating layer is trained jointly with the Rainbow selector;
- target-network synchronization and optimizer settings are configurable.

These decisions instantiate the Rainbow mechanisms explicitly named by the paper while avoiding invented claims about omitted historical hyperparameters.

## ECM evaluation choice

For repeated rewards and CELF comparisons, `LiveEdgeECOracle` fixes a collection of Independent-Cascade live-edge Monte-Carlo worlds. This is distributionally equivalent to IC Monte Carlo and makes all methods see the same random worlds. It also prevents method rankings from being dominated by independent Monte-Carlo noise at every marginal-gain call.

## Data reconstruction choices

The paper specifies the sensing rectangle, 5 km grid, target-subarea count, graph sizes, worker-pool size, edge-probability range, and distance formula, but does not uniquely specify all preprocessing details.

Current transparent reconstruction choices are:

- source users/check-ins are first restricted to the paper's stated rectangle;
- users are sampled uniformly from region users that also appear in the social graph;
- released SNAP friendship edges are treated as bidirectional by default because SNAP exposes the friendship networks as undirected; this can be disabled;
- each resulting directed edge gets an independently sampled probability in `[0.1,0.5]`;
- target grid cells and worker pool are sampled from a recorded RNG seed;
- distance is computed literally on `(longitude, latitude)` coordinates as written in the paper's Euclidean-distance equation;
- because the exact numerical map from check-in count to `d_i` is absent, demand transformations are named variants (`raw_checkin_count`, `sqrt_count`, `max_normalized`) and never silently conflated with the historical paper setting.

## Claim discipline

Outputs are labelled at one of three levels:

1. `algorithm verification`: synthetic/unit evidence that equations/components are implemented consistently;
2. `transparent reconstruction`: experiments generated from fully recorded assumptions above;
3. `historical paper reproduction`: reserved until the original preprocessing/data-generation details are recovered or independently established.
