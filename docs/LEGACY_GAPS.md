# Legacy code gaps found during reproduction

The public `main` branch is not treated as an executable specification. The following discrepancies are already visible before any large-scale rerun.

## A. Rainbow is not implemented as described

The paper states that DQNSelector uses Rainbow extensions including DDQN, Dueling DQN, prioritized replay, multi-step learning, distributional RL, and noisy net. The released `code/DQNSelector.py` instead contains a TensorFlow-v1-style ordinary DQN with a basic replay array, an evaluate network and target network, RMSProp, and epsilon/random action behavior. The named Rainbow components are not all present.

## B. Gating layer is absent from the released training path

The paper defines two learned gates before concatenating the social and coverage embeddings. The released `train` / `test` path directly performs `np.concatenate([step_cov_embedding[node], node2vec_embedding[node]])`. Thus the paper's gated DQNSelector and the public code path are not the same model.

## C. Paper hyperparameters do not match the executed script defaults

Paper experiment settings include:

- social influence embedding dimension 128;
- MC simulation times 100;
- graph iteration times 30;
- DQN training 100 episodes.

The released bottom-of-file execution path calls the embedding learner with `MC_times=30`, `MC_iter_times=1`, `train_iter_times=1`, and sets `episode=50`. These are not the paper values.

## D. Coverage embedding generation is commented out but later loaded

The public script comments out the code that computes `step_cov_embedding`, then later loads `../param/step_cov_embedding_*.npy`. The public repository does not contain a complete `param/` state required to make this an end-to-end runnable pipeline.

## E. PIIC-equivalent legacy function adds an undocumented normalization

The released `coverage_step` carries out a layer-wise weighted recurrence similar to PIIC but then divides every coverage dimension by its maximum across nodes. This max-normalization is not stated in the paper's Eq. (10), Eq. (11), or Algorithm 2.

## F. Reward/effective coverage differs from the paper definition

The paper defines `EC(a_i,S)=min(C(a_i,S)/d_i,1)`. The released `effective_cov` sums approximate activation probability times each node's stored vector and returns the mean and standard deviation without an explicit per-subarea demand division/cap in that function. This may mean the stored vectors are already transformed, but that transformation is not documented in the repository, so the equivalence cannot be assumed.

## G. Released data is incomplete relative to the paper experiments

The paper reports Gowalla and Brightkite, 3000 and 5000 users, multiple randomly generated graphs/instances, and averages over two datasets. The public repository currently exposes only `input_node_3000_3.txt` and `input_edge_3000_3.txt` under `dataset/data_1`.

## H. Model filename / experiment-size inconsistencies

The released script defaults to 3000 users while saving the final checkpoint under a `5000_*.ckpt` filename. This is another sign that the public script is a partial experiment artifact rather than a polished reproduction pipeline.

## Policy for the new implementation

When paper and code disagree, the clean `repro/` implementation follows the paper and records any additional assumption explicitly. Legacy behavior is isolated in `dqnselector.legacy` so it can be compared without contaminating the paper-level implementation.
