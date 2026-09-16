# Review-driven extension checklist

This checklist maps the saved SIGIR review summary to concrete reproduction / journal-extension tasks.

- [ ] Clarify DQNSelector vs PIANO using matched indicators and an implementation-level comparison.
- [ ] Expand the PIIC explanation with the matrix recurrence, explicit path example, complexity derivation, and cycle limitation.
- [ ] Expand Section 4.2-equivalent documentation so every Rainbow component is individually defined and ablated.
- [ ] Add at least one non-sensing-task scenario if the journal version continues to claim broader worker-recruitment applicability.
- [ ] Add a symbol table and keep notation consistent between code/configuration and manuscript.
- [ ] Discuss practical applicability: data privacy, social-network access, worker-selection overhead, and operational cost.
- [ ] Make training assumptions explicit, including how instances are generated and whether graph/model parameters transfer across instances.
- [ ] Add a worker-reliability/noisy-quality robustness experiment rather than assuming every activated worker is reliable.
- [ ] Report exact numerical results in tables in addition to plots; improve Figure-5 readability.
- [ ] Explain why SocialRecruiter is or is not included as a baseline and, if excluded, make the model incompatibility explicit.
- [ ] Reassess the practical runtime argument against CELF because the review noted that CELF runtime may still be acceptable for the reported graph sizes.
- [ ] Weaken or re-test the motivation based on rapid social-network change; one crowdsourcing assignment may see little topology change.
- [ ] Add sensing-area geometry experiments: scattered target subareas, contiguous target subareas, and several spatial clusters.

## Reproduction prerequisites before extension experiments

- [ ] Unit-test IC propagation and ECM capping.
- [ ] Unit-test PIIC against explicit path enumeration on DAGs.
- [ ] Verify social embedding fits MC probabilities.
- [ ] Verify gated fusion is actually used in training.
- [ ] Verify all Rainbow components are active.
- [ ] Recover or transparently regenerate complete Gowalla/Brightkite instances.
- [ ] Re-run every baseline from the same objective implementation and data manifest.
