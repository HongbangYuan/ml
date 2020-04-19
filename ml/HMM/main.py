
"""
Info    Hidden Markov Model.
Author  Yiqun Chen
Time    2020-04-19
"""

from ml.utils.metrics import Logger
import numpy as np
import copy

class HiddenMarkovModel(object):
    def __init__(
        self, state_transition_matrix, observation_matrix, initialial_state, \
            hidden_state, observation, method="viterbi",
    ):
        """
        Info    initialize the hidden markov model.
        Args    state_transition_matrix (numpy.ndarray): 
                observation_matrix (numpy.ndarray):
                initial_state (numpy.ndarray): initial state vector.
                hidden_state (dict): 
                    {
                        "state_name": row_of_state_transition_matrix, ...
                    }
                observation (dict):
                    {
                        "observation_name": col_of_transition_matrix, ...
                    }
                method (string): the method to solve the problem.
        """
        super().__init__()
        # self.sequence_length = sequence_length
        self.state_transition_matrix = state_transition_matrix
        self.observation_matrix = observation_matrix
        self.hidden_state = hidden_state
        self.observation = observation
        self.initial_state = initialial_state
        self.method = method

    def forward(self, sequence, t):
        """
        Info    calculate forward probability.
        Args    sequence (list of string): a sequence of string.
                t (int): the index of sequence, start with 0.
        Returns alpha (numpy.ndarray): forward probability.
        """
        alpha = self.initial_state * self.observation_matrix[:, self.observation[sequence[0]]]
        for _t in range(t):
            alpha = (alpha*self.state_transition_matrix.T).T
            alpha = np.sum(alpha, axis=0) * self.observation_matrix[:, self.observation[sequence[_t+1]]]
        return alpha

    def backward(self, sequence, t):
        """
        Info    calculate backward probability.
        Args    sequence (list of string): a sequence of string.
                t (int): the index of sequence, start with 0.
        Returns beta (numpy.ndarray): backward probability.
        """
        beta = np.ones(self.state_transition_matrix.shape[0])
        for _t in range(len(sequence)-t-1):
            beta = beta * self.observation_matrix[:, self.observation[sequence[len(sequence)-_t-1]]] \
                * self.state_transition_matrix
            beta = np.sum(beta, axis=1)
        return beta

    def cal_state_probability(self, sequence, t, state):
        alpha = self.forward(sequence, t)
        beta = self.backward(sequence, t)
        probability = alpha[self.hidden_state[state]] * beta[self.hidden_state[state]] /\
            np.sum(alpha * beta)
        return probability

    def viterbi(self, sequence):
        pass

if __name__ == "__main__":
    sequence = ["red", "white", "red", "red", "white", "red", "white", "white"]
    state_transition_matrix = np.array([
        [0.5, 0.1, 0.4], 
        [0.3, 0.5, 0.2], 
        [0.2, 0.2, 0.6], 
    ])
    observation_matrix = np.array([
        [0.5, 0.5], 
        [0.4, 0.6], 
        [0.7, 0.3], 
    ])
    initial_state = np.array([0.2, 0.3, 0.5])
    observation = {"red": 0, "white": 1}
    hidden_state = {"box_0": 0, "box_1": 1, "box_2": 2}

    model = HiddenMarkovModel(
        state_transition_matrix, observation_matrix, initial_state, hidden_state, observation
    )

    ##  index of sequence will start with 0 not 1.
    probability = model.cal_state_probability(sequence, 3, "box_2")
    print(">>>> probability: {:.3f}".format(probability))