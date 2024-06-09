import numpy as np
import tensorflow as tf
from HO.main.globalval import GlobalVal as gv
import matplotlib.pyplot as plt
# np.random.seed(6)
# tf.random.set_seed(6)
class DeepQNetwork:

    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.6,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.compat.v1.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.compat.v1.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.compat.v1.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.compat.v1.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections are used later when assign to target net
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections are used later when assign to target net
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1,w2) + b2
        with tf.name_scope('loss'):
                self.loss = tf.reduce_sum(tf.compat.v1.squared_difference(self.q_target,self.q_eval))
        with tf.name_scope('train'):
                self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        #-----------build target_net----------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32,[None,self.n_features], name='s_')
        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params',tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1 = tf.compat.v1.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2 = tf.compat.v1.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_next = tf.matmul(l1,w2) + b2

        '''Read model parameters'''
        # saver = tf.compat.v1.train.Saver()
        # # init = tf.compat.v1.global_variables_initializer()
        # self.sess = tf.compat.v1.Session()
        # self.sess.run(tf.compat.v1.global_variables_initializer())
        # saver.restore(self.sess, "../model/DQN.ckpt")

    def save_net(self):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(self.sess, "../model/DQN.ckpt")
        print("Save to path: ", save_path)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        gv.cost_DQN.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self,episode):
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)  
        plt.plot(np.arange(len(gv.cost_DQN)), gv.cost_DQN)  # gv.cost_his
        print(gv.cost_DQN)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        # plt.savefig('..\\cost\\cost' +str(episode)+'+with prediction.png',format='png',dpi = 300)
        # plt.savefig('..\\cost\\cost' + str(episode) + '+without prediction.png', format='png', dpi=300)
        # plt.show()
