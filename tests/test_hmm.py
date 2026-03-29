import numpy as np
from tropiq.hmm import HMM

def test_viterbi_basic():
    """
    2-state HMM, 3 observations.

    States: 0 = "low", 1 = "high"
    Observations: 0 or 1

    Setup: each state strongly prefers to emit its own index (0 or 1),
    and strongly prefers to stay in the same state.
    Observations [0, 0, 1] should decode to states [0, 0, 1].
    """
    hmm = HMM(n_states=2)
    hmm.log_init  = np.log([0.9, 0.1])
    hmm.log_trans = np.log([[0.9, 0.1],   # from state 0: stay with 0.9
                             [0.1, 0.9]])  # from state 1: stay with 0.9
    hmm.log_emit  = np.log([[0.9, 0.1],   # state 0: emits 0 with 0.9
                             [0.1, 0.9]]) # state 1: emits 1 with 0.9

    observations = np.array([0, 0, 1])
    states = hmm.viterbi(observations)

    np.testing.assert_array_equal(states, [0, 0, 1])
