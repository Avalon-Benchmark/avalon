Unit tests it would be nice to have:
- rollouts
  - things that need tested
    x it works properly in the base case (procgen parameters)
    - obs_first and obs_last
    x time limit bootstrapping
    x single and parallel envs
    - partial batches work properly (add a random time-delay wrapper to test?)
    x single and multiprocessing
  x deterministically compare the fast rollout worker to the simple rollout worker, under various configurations
- action head / action distributions
- storage
- algorithms
  - PPO
    - parameters
      - not too many come to mind?
    - things that need tested
      - it (ideally deterministically?) matches a known good implementation 
  - dreamer
  - others?
- evaluation

Regenerating the determinism tests (only do this when you know you've broken determinism and want to reset the test!):
`agent/tests/generate_test_data.py --input --output`
