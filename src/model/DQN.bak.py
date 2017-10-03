import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import math
import sys

from collections import deque

from Memory import Memory

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:

    learn_step_counter = 1

    def __init__(
            self,
            n_action,
            n_lookback,
            n_feature,
            n_channel,
            learning_rate=0.0001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            prioritized=True,
            sess=None,
            load_memory=False
    ):

        self.n_action = n_action
        self.n_feature = n_feature
        self.n_channel = n_channel
        self.lr = learning_rate
        self.op_decay = 0.9
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not
        self.prioritized = prioritized
        self.output_graph = output_graph

        self.keep_prob_l1 = 0.8
        self.l1_dim = 32
        self.fc_dim = 16
        self.trace_length = 4
        # self.learn_step_counter = 1
        self.cost = 0
        # self.memory = np.zeros((self.memory_size, n_feature*2+2))

        if load_memory:
            self.load_memory()
        elif self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_feature*2+2))

        self.totalLoss = 0.0
        self.totalQ = 0.0
        self.totalMaxQ = 0.0
        self.r_actions = deque()
        self.cost_his = deque()

        self.graph = tf.Graph()
        self._build_net()

        with self.graph.as_default() as graph:
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        if sess is None:
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init)
        else:
            self.sess = sess

        if self.output_graph:
            self.summary_writer = tf.summary.FileWriter("log/", self.graph)

    def _build_net(self):
        def build_layers(s, c_names, n_l1, n_fc, w_initializer, b_initializer, sample_size):

            s = tf.reshape(s, [sample_size, self.n_feature, self.n_channel])
            n_filter = 32
            # keep_prob = 1

            with tf.variable_scope('conv1') as scope:
                # k1 = tf.get_variable('kernel1', shape=[1, self.n_channel, self.n_feature])
            #     conv1 = tf.nn.conv1d(s, k1, stride=2, padding='SAME', use_cudnn_on_gpu=True)
                conv1 = tf.layers.conv1d(s, n_filter, 1, strides=1, padding='SAME', activation=None)
                # conv1 = tf.layers.max_pooling1d(conv1, 2, 4, padding='SAME')
            #
            with tf.variable_scope('conv2') as scope:
            #     # k2 = tf.get_variable('kernel2', shape=[1, n_l1, n_l1])
            #     # conv2 = tf.nn.conv1d(conv1, k2, stride=2, padding='SAME', use_cudnn_on_gpu=True)
                conv2 = tf.layers.conv1d(s, n_filter, 1, strides=1, padding='SAME', activation=None)
                conv2 = tf.layers.conv1d(conv2, n_filter, 3, strides=1, padding='SAME', activation=None)
                conv2 = tf.layers.conv1d(conv2, n_filter, 3, strides=1, padding='SAME', activation=None)
                # conv2 = tf.layers.max_pooling1d(conv2, 2, 4, padding='SAME')
            #
            with tf.variable_scope('conv3') as scope:
            # #     k3 = tf.get_variable('kernel3', shape=[1, n_l1, n_fc])
            # #     fc = tf.nn.conv1d(conv2, k3, stride=2, padding='SAME', use_cudnn_on_gpu=True)
                conv3 = tf.layers.conv1d(s, n_filter, 1, strides=1, padding='SAME', activation=None)
                conv3 = tf.layers.conv1d(conv3, n_filter, 5, strides=1, padding='SAME', activation=None)
                # conv3 = tf.layers.max_pooling1d(conv2, 2, 4, padding='SAME')

            with tf.variable_scope('conv4') as scrope:
                conv4 = tf.layers.average_pooling1d(s, 3, 1, padding='SAME')
                conv4 = tf.layers.conv1d(conv4, n_filter, 1, strides=1, padding='SAME', activation=None)

            # with tf.variable_scope('conv1') as scope:
            #     conv1_filter = tf.Variable(tf.truncated_normal([1, 1, self.n_channel, n_filter]))
            #     conv1 = tf.nn.conv2d(s, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
            #     conv1 = tf.layers.max_pooling2d(conv1, 2, (1, 4), padding='SAME')
            #
            # with tf.variable_scope('conv2') as scope:
            #     conv2_filter = tf.Variable(tf.truncated_normal([3, 3, self.n_channel, n_filter]))
            #     conv2 = tf.nn.conv2d(s, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
            #     conv2 = tf.layers.max_pooling2d(conv2, 2, (1, 4), padding='SAME')
            #
            # with tf.variable_scope('conv3') as scope:
            #     conv3_filter = tf.Variable(tf.truncated_normal([5, 5, self.n_channel, n_filter]))
            #     conv3 = tf.nn.conv2d(s, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
            #     conv3 = tf.layers.max_pooling2d(conv3, 2, (1, 4), padding='SAME')
            #
            # with tf.variable_scope('conv4') as scope:
            #     conv4 = tf.nn.avg_pool(s, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            #     conv4 = tf.layers.max_pooling2d(conv4, 2, (1, 4), padding='SAME')

            with tf.variable_scope('concat') as scope:
                inception1 = tf.concat([conv1, conv2, conv3, conv4], axis=2)
                bias = tf.get_variable(name='biases', initializer=tf.constant_initializer(), shape=[4 * n_filter])
                inception1 = tf.nn.bias_add(inception1, bias)
                inception1 = tf.nn.relu(inception1)
                fc = tf.layers.average_pooling1d(inception1, 8, 8, padding='SAME')
                fc = tf.reshape(fc, shape=[sample_size, 256])

            # with tf.variable_scope('conv5') as scope:
            #     conv5 = tf.layers.conv1d(inception1, n_filter, 1, strides=1, padding='SAME', activation=None)
            #     bias = tf.Variable(tf.truncated_normal([n_filter]))
            #     conv5 = tf.nn.bias_add(conv5, bias)
            #     conv5 = tf.nn.relu(conv5)
            #     conv5 = tf.layers.max_pooling1d(conv5, 3, 4, padding='SAME')
            #     fc = tf.reshape(conv5, shape=[sample_size, 1408])

            # with tf.variable_scope('dense') as scope:
            #     W = tf.get_variable(name='weights', initializer=tf.contrib.layers.variance_scaling_initializer(), shape=[1078, n_filter])
            #     z = tf.einsum('ijk,kl->ijl', inception1, W)
            #     b = tf.get_variable(name='biases', initializer=tf.constant_initializer(), shape=[n_filter])
            #     fc = z + b

            # with tf.variable_scope('conv1') as scope:
            #     conv1_filter = tf.Variable(tf.truncated_normal([1, 1, self.n_channel, n_filter]))
            #     conv1 = tf.nn.conv2d(s, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
            #     conv1 = tf.layers.max_pooling2d(conv1, 2, (1, 4, 4, 1), padding='SAME')
            #     bias = tf.Variable(tf.truncated_normal([n_filter]))
            #     fc = tf.nn.bias_add(conv1, bias)
            #     fc = tf.nn.relu(fc)
            #     fc = tf.reshape(fc, shape=[sample_size, 9, 704])

            # with tf.variable_scope('rnn') as scope:
            #     cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_l1, state_is_tuple=True)
            #     state_in = cell.zero_state(tf.shape(fc)[0], tf.float32)
            #     # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            #     rnn, state = tf.nn.dynamic_rnn(inputs=fc, cell=cell, dtype=tf.float32, initial_state=state_in)
            #     fc = state[1]



            # with tf.variable_scope('l1') as scope:
            #     w1 = tf.get_variable('w1', shape=[5632, n_l1], initializer=w_initializer, collections=c_names)
            #     b1 = tf.get_variable('b1', shape=[1, n_l1], initializer=b_initializer, collections=c_names)
            #     l1 = tf.nn.relu(tf.matmul(fc, w1) + b1)
            #
            # with tf.variable_scope('l2') as scope:
            #     w2 = tf.get_variable('w2', shape=[n_l1, n_fc], initializer=w_initializer, collections=c_names)
            #     b2 = tf.get_variable('b2', shape=[1, n_fc], initializer=b_initializer, collections=c_names)
            #     fc = tf.nn.relu(tf.matmul(l1, w2) + b2)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w_out = tf.get_variable('w_out', [256, 1], initializer=w_initializer, collections=c_names)
                    b_out = tf.get_variable('b_out', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(fc, w_out) + b_out
                    # self.V = tf.nn.bias_add(tf.matmul(l1, w2), b2)

                with tf.variable_scope('Advantage'):
                    w_out = tf.get_variable('w_out', [256, self.n_action], initializer=w_initializer, collections=c_names)
                    b_out = tf.get_variable('b_out', [1, self.n_action], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(fc, w_out) + b_out
                    # self.A = tf.nn.bias_add(tf.matmul(l1, w2), b2)

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w_out = tf.get_variable('q_w_out', [n_fc, self.n_action], initializer=w_initializer, collections=c_names)
                    b_out = tf.get_variable('q_b_out', [1, self.n_action], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(fc, w_out) + b_out
                    # out = tf.nn.bias_add(tf.matmul(l1, w2), b2)

            return out
            # return w1, w2, w3, out
            # return w1, out

        with self.graph.as_default() as graph:
            # ------------------ build evaluate_net ------------------
            if self.prioritized:
                self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.s = tf.placeholder(tf.float32, shape=(None, self.n_feature), name='s')  # input
            self.q_target = tf.placeholder(tf.float32, shape=(None, self.n_action), name='Q_target')  # for calculating loss
            self.sample_size = tf.Variable(1, dtype=tf.int32, name='sample_size')

            with tf.variable_scope('eval_net'):
                # c_names, n_l1, w_initializer, b_initializer = \
                #     ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                #     tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
                c_names, n_l1, n_fc, w_initializer, b_initializer = \
                    ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.l1_dim, self.fc_dim, \
                    tf.contrib.layers.xavier_initializer(), tf.random_normal_initializer()

                self.q_eval = build_layers(self.s, c_names, n_l1, n_fc, w_initializer, b_initializer, self.sample_size)
                # w1, w2, w3, self.q_eval = build_layers(self.s, c_names, n_l1, n_fc, w_initializer, b_initializer)
                # w1, self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

            with tf.variable_scope('loss'):
                if self.prioritized:
                    self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                    self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
                tf.summary.scalar('loss', self.loss)

            # with tf.variable_scope('l2_loss'):
            #     self.loss = self.loss + tf.nn.l2_loss(w1) * 0.01
            #     self.loss = self.loss + tf.nn.l2_loss(w2) * 0.01
            #     self.loss = self.loss + tf.nn.l2_loss(w3) * 0.01

            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.op_decay).minimize(self.loss)
                # self._train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None, self.n_feature], name='s_')    # input
            with tf.variable_scope('target_net'):
                c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                self.q_next = build_layers(self.s_, c_names, n_l1, n_fc, w_initializer, b_initializer, self.sample_size)
                # w1_target, w2_target, w3_target, self.q_next = build_layers(self.s_, c_names, n_l1, n_fc, w_initializer, b_initializer)
                # w1_target, self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

            ######
            w_c_names = 'eval_net_params_summaries'

            with tf.variable_scope('summary') as scope:
                scalar_summary_tags = ['loss_avg', 'e_balance', \
                                     'q_max', 'q_total', 'epsilon', \
                                     'sharpe_ratio', 'n_trades', \
                                     'win', 'win_buy', 'win_sell', \
                                     'max_profit', 'avg_profit', 'max_loss', 'avg_loss', \
                                     'total_profit', 'total_loss', \
                                     'max_holding_period', 'avg_holding_period', \
                                     'avg_profit_holding_period', 'avg_loss_holding_period', \
                                     'max_floating_profit', 'max_floating_loss', \
                                     'max_total_balance', 'profit_make_good', \
                                     'up_buy', 'down_sell', \
                                     'n_buy', 'n_sell', 'reward', 'diff_sharpe']

                self.summary_placeholders = {}
                self.summary_ops = {}

                for tag in scalar_summary_tags:
                    self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag.replace(' ', '_') + '_0')
                    self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

            # with tf.variable_scope('training_step'):
            #     training_step_mse = tf.summary.scalar('mse', self.loss)
                histogram_summary_tags = ['r_actions']

                for tag in histogram_summary_tags:
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_') + '_0')
                    self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

            # with tf.variable_scope('param'):

            #     histogram_w_tags = ['l1_w', 'l1_b', 'lout_w', 'lout_b']
            #
            #     for tag in histogram_w_tags:
            #         tf.summary.histogram(tag, self.w[tag], collections = [w_c_names])

            with tf.variable_scope('Extra') as scope:

                # boltzmann
                self.temp = tf.placeholder(shape=None, dtype=tf.float32)
                self.q_dist = tf.nn.softmax(self.q_eval/self.temp)
                self.action = tf.squeeze(tf.multinomial(tf.log(self.q_dist), 1))

                # e-greedy
                # self.action = tf.argmax(self.q_eval, axis=1)

            # if self.output_graph:
            # self.merged = tf.summary.merge([training_step_mse])
                # self.writer = tf.summary.FileWriter('data/' + self.dir, self.sess.graph)
            ######

            # self.merged = tf.summary.merge_all()

    def store_transition(self, s, a, r, s_):

        if not s.shape[0] == s_.shape[0]:
            print ('Error observation shapes are not the same, s: {}, s_: {}'.format(s.shape, s_.shape))
            sys.exit(2)

        # transition = np.hstack((list(s.flat), [a, r], list(s_.flat)))
        transition = {'s': s, 'a': a, 'r': r, 's_': s_}

        if self.prioritized:    # prioritized replay
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0

            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        # boltzmann
        # action, q_eval = self.sess.run([self.action, self.q_eval], feed_dict={self.s: observation, self.sample_size: 1, self.temp: 1 - self.epsilon})
        # action = np.random.choice(q_dist[0], p=q_dist[0])
        # action = np.argmax(q_dist[0] == action)

        # e-greedy
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation, self.sample_size: 1})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_action)

        # if not hasattr(self, 'q'):
        #     self.q = []
        #     self.running_q = 0
        # self.running_q = self.running_q * 0.99 + 0.01 * np.max(q_eval)
        # self.q.append(self.running_q)
        #
        # self.r_actions.append(action)

        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            # print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            # sample_index = np.random.choice(self.memory_size - self.trace_length, size=self.batch_size)
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            # sample_memory = self.memory[sample_index, :]
            #
            # batch_memory = []
            # for index in sample_index:
            #     batch_memory.append(self.memory[index: index + self.trace_length, :])
            # print(batch_memory.shape)
            # sys.exit(2)
        # pointer = int(self.n_feature * self.n_channel)

        length = len(batch_memory)
        s = np.array([batch_memory[i]['s'] for i in range(length)])
        # s = batch_memory[:, :pointer]
        # s = np.reshape(s, (-1, self.n_channel, self.n_feature))
        # print('s.shape: {}'.format(s.shape))

        s_ = np.array([batch_memory[i]['s_'] for i in range(length)])
        # s_ = batch_memory[:, -pointer:]
        # s_ = np.reshape(s_, (-1, self.n_channel, self.n_feature))
        # print('s_.shape: {}'.format(s_.shape))

        q_next, q_eval4next,  = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: s_, self.s: s_, self.sample_size: self.batch_size})    # next observation
            # feed_dict={self.s_: batch_memory[:, -self.n_feature:],    # next observation
            #            self.s: batch_memory[:, -self.n_feature:],
            #            self.sample_size: self.batch_size})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: s, self.sample_size: self.batch_size})
        # q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_feature], self.sample_size: self.batch_size})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = np.array([batch_memory[i]['a'] for i in range(length)], dtype=np.int32)
        # eval_act_index = batch_memory[:, pointer].astype(int)
        # eval_act_index = batch_memory[:, self.n_feature].astype(int)
        reward = np.array([batch_memory[i]['r'] for i in range(length)])
        # reward = batch_memory[:, pointer + 1]
        # reward = batch_memory[:, self.n_feature + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]

        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                        feed_dict={self.s: s,
                                        # feed_dict={self.s: batch_memory[:, :pointer],
                                        #  feed_dict={self.s: batch_memory[:, :self.n_feature],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights,
                                                    self.sample_size: self.batch_size})
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_feature],
                                                    self.q_target: q_target,
                                                    self.sample_size: self.batch_size})

        # if self.output_graph:
        #     self.summary_writer.add_summary(merged, self.learn_step_counter)
        # self.cost_his.append(self.cost)

        if self.learn_step_counter % 100 == 0:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # self.epsilon = self.epsilon + 0.1 * math.expm1(self.epsilon_increment * self.learn_step_counter) if self.epsilon < self.epsilon_max else self.epsilon_max
        # print ('self.epsilon: {}, self.epsilon_increment: {}, self.learn_step_counter: {}'.format(self.epsilon, self.epsilon_increment, self.learn_step_counter))
        self.learn_step_counter += 1

        self.totalLoss += self.cost
        self.totalQ += q_eval.mean(axis = 1).mean(axis = 0)
        self.totalMaxQ += np.max(q_eval, axis=1).mean()

    # mode 0: normal save, 1: period save
    # def saveParam(self, dir = 'tmp', mode = 0):
    #     subdir = ''
    #
    #     if mode == 1:
    #         subdir = 'history/%s/' % (dir)
    #
    #     fulldir = 'data/%s/%s' % (self.dir, subdir)
    #
    #     mkdir(fulldir)
    #     self.saver.save(self.sess, '%s%s' % (fulldir, self.ckptFile))

    def inject_summary(self, tag_dict, episode):

        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })

        for summary_str in summary_str_lists:
            self.summary_writer.add_summary(summary_str, episode)

        # self.summary_writer.add_summary(self.param_summary, episode)

    def finish_episode(self, episode, stat):

        if episode > 0:
            injectDict = {
                # scalar
                'loss_avg': self.totalLoss / float(stat['count']),
                'e_balance': stat['total_balance'],
                'sharpe_ratio': stat['sharpe_ratio'],
                'n_trades': stat['n_trades'],
                'win': float(stat['n_win']) / float(stat['n_trades']),
                'win_buy': float(stat['win_buy']) / float(stat['n_buy']) if int(stat['n_buy']) > 0 else 0,
                'win_sell': float(stat['win_sell']) / float(stat['n_sell']) if int(stat['n_sell']) > 0 else 0,
                'n_buy': stat['n_buy'],
                'up_buy': float(stat['n_up_buy']) / float(stat['n_buy']),
                'n_sell': stat['n_sell'],
                'down_sell': float(stat['n_down_sell']) / float(stat['n_sell']),
                'max_profit': stat['max_profit'],
                'avg_profit': float(stat['total_profit']) / float(stat['n_win']),
                'total_profit': stat['total_profit'],
                'max_loss': stat['max_loss'],
                'avg_loss': float(stat['total_loss']) / (float(stat['n_trades']) - float(stat['n_win'])),
                'total_loss': stat['total_loss'],
                'reward': stat['reward'],
                'max_holding_period': stat['max_holding_period'],
                'avg_holding_period': float(stat['total_holding_period']) / float(stat['n_trades']),
                'avg_profit_holding_period': float(stat['total_profit_holding_period']) / float(stat['n_win']),
                'avg_loss_holding_period': float(stat['total_loss_holding_period']) / (float(stat['n_trades']) - float(stat['n_win'])),
                'max_floating_profit': stat['max_floating_profit'],
                'max_floating_loss': stat['max_floating_loss'],
                'max_total_balance': stat['max_total_balance'],
                'profit_make_good': stat['profit_make_good'],
                # 'r_balance': realBalance,
                'epsilon': self.epsilon,
                'q_max': self.totalMaxQ,
                'q_total': self.totalQ,
                'r_actions': self.r_actions,
                'diff_sharpe': stat['diff_sharpe']
            }

            if self.output_graph:
                self.inject_summary(injectDict, episode)

            # self.saveParam(mode = 0)
            # if episode % self.ckptSavePeriod == 0:
            #     self.saveParam(dir = '%d' % (episode), mode = 1)

        self.r_actions = deque()
        self.totalLoss = 0.0
        self.totalQ = 0.0
        self.totalMaxQ = 0.0

    def load(self, step=0):

        print(sys.path)

        # checkpoint_dir = '/Users/cc/Project/Lean/Launcher/bin/Debug/python/oracle/data/'
        checkpoint_dir = './data'

        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            self.learn_step_counter = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        except:
            ckpt = None

        if not (ckpt and ckpt.model_checkpoint_path):
            print('Cannot find any saved sess in checkpoint_dir')
            #sys.exit(2)
        else:
            try:
                # self.saver = tf.train.Saver()
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=step)
                print('Sess restored successfully: {}'.format(ckpt.model_checkpoint_path))
            except Exception as e:
                print('Failed to load sess: {}'.format(str(e)))
                # sys.exit(2)
                self.learn_step_counter = 1

    def save(self, path=None):

        if (path is not None):
            save_path = path
        else:
            save_path = './data/sess.ckpt'

        self.saver.save(self.sess, save_path, global_step=self.learn_step_counter)
        print('Saving sess to {}: {}'.format(save_path, self.learn_step_counter))

    def load_memory(self):

        path = '/Users/cc/Project/Lean/Launcher/bin/Debug/python/oracle/data/memory.pickle'
        print('Loading memory from {}'.format(path))

        with open(path, 'rb') as f:
            self.memory = pickle.load(f)

    def save_memory(self):

        path = './data/memory.pickle'
        print('Saving memory to {}'.format(path))

        with open(path, 'wb') as f:
            pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)