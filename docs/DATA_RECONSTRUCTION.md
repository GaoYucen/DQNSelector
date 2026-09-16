# Data reconstruction status

## What the paper specifies

The experiment section states that the source datasets are Gowalla and Brightkite and that both social relationships and check-in/location information are used. It further specifies:

- a rectangular sensing region with longitude approximately `[-122.50, -118.00]` and latitude `[33.80, 37.90]`;
- a 5 km x 5 km grid, yielding 7200 subareas;
- 100 target sensing subareas sampled from that grid;
- graph sizes of 3000 and 5000 users;
- worker-pool size 300;
- edge influence probabilities sampled in `[0.1, 0.5]`;
- seed budgets in `{50,60,70,80,90,100}`;
- participation probability and sensing quality computed from each user's minimum Euclidean distance to each target subarea, using the normalized formula in Eqs. (23)--(24);
- subarea demand `d_i` is based on the number of check-ins in the subarea, with more check-ins implying higher demand.

## What is missing for exact reconstruction

The paper/repository do not provide enough detail to uniquely recreate the reported random instances:

1. the random seeds and exact user samples for each 3000/5000-user graph;
2. the exact rule used to map check-in count to numerical demand `d_i`;
3. the exact 100 target subareas selected for each instance;
4. the exact worker-pool samples;
5. whether all social edges between sampled users were retained and how direction was assigned when the source relationship is undirected;
6. the complete generated instance collection used to average results;
7. the processed Brightkite/Gowalla split corresponding to the released `dataset/data_1` sample.

Because of these omissions, a script that silently chooses reasonable defaults may produce a useful *new benchmark*, but it cannot honestly be described as an exact reconstruction of the paper's Figure 5.

## Reconstruction plan

The clean reproduction therefore uses two tracks:

### Track A: released processed instance

Load `dataset/data_1/input_node_3000_3.txt` and `input_edge_3000_3.txt` exactly as released, then use it for implementation diagnostics and legacy comparisons. The stored node vectors are treated as opaque processed values because separate `p_v^i`, `q_v^i`, and `d_i` cannot be recovered uniquely.

### Track B: transparent regenerated instances

When raw Gowalla/Brightkite files are available, construct a new reproducible pipeline with every random seed and transformation recorded in a JSON manifest. For the under-specified demand mapping, run explicitly named alternatives (for example linear count normalization, quantile normalization, or a recovered historical mapping if later found) rather than choosing one invisibly.

## Acceptance criteria before claiming paper-level reproduction

A Figure-5-style result should only be labelled `paper reproduction` if either:

- the original generated instances / preprocessing code are recovered; or
- the missing transformations are independently identified with sufficient evidence.

Otherwise results should be labelled `transparent reconstruction` and compared qualitatively, not presented as the exact historical numbers.
