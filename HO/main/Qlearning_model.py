import numpy as np
import pandas as pd

INITIAL_EPSILON = 0.9  # starting value of epsilon
FINAL_EPSILON = 0.001  # final value of epsilon


class QLearningTable:
    def __init__(self, actions, n_states=7,learning_rate=0.01, reward_decay=0.95):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = INITIAL_EPSILON
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # self.q_table = pd.DataFrame(np.zeros((n_states,len(actions))), columns = actions)
                                    

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def action(self, observation, Q):
        if observation not in self.q_table.index:
            action = 10
        # self.check_state_exist(observation)
        else:
            state_action = Q.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            # self.q_table = pd.concat([self.q_table,pd.Series(
            #         [0] * len(self.actions),
            #         index=self.q_table.columns,
            #         name=state)])
                                     
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),index=self.q_table.columns,name=state,))