import numpy as np


class HMM:
    """Hidden Markov Model with Viterbi decoding."""

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.log_init = None   # (n_states,)            log initial distribution
        self.log_trans = None  # (n_states, n_states)   log transition matrix
        self.log_emit = None   # (n_states, n_obs)      log emission matrix

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """
        Find the most probable hidden state sequence for the given observations.

        Args:
            observations: 1D integer array of observation indices, length T

        Returns:
            1D integer array of state indices, length T
        """
        T = len(observations)
        S = self.n_states

        # --- FORWARD PASS ---
        psi = np.zeros((T, S), dtype=int)
        delta = self.log_init + self.log_emit[:, observations[0]]

        for t in range(1, T):
            scores = delta[:, np.newaxis] + self.log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta = np.max(scores, axis=0) + self.log_emit[:, observations[t]]

        # --- TRACEBACK ---
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta)
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states
