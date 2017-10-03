import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import math
import sys

from collections import deque

from memory import Memory

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:

    def __init__(
            self,
            n_action,
            n_width,
            n_height,
            n_channel,
            learning_rate=0.0001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
            double_q=True,
            dueling=True,
            prioritized=True,
            sess=None,
            load_memory=False
    ):

        self.n_action = n_action
        self.n_width = n_width
        self.n_height = n_height
        self.n_channel = n_channel

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q
        self.dueling = dueling
        self.prioritized = prioritized
        self.output_graph = output_graph

        self.learn_step_counter = 0
        self.cost = 0
        self.totalLoss = 0
        self.totalQ = 0
        self.totalMaxQ = 0

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_width*2+2))

        self.graph = tf.Graph()
        self._build_net()

        with self.graph.as_default() as graph:
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            t_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init)
            # self.sess = tf.Session()
            # self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if self.output_graph:
            self.summary_writer = tf.summary.FileWriter("log/", self.graph)
            # self.summary_writer = tf.summary.FileWriter("log/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):

        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):

            # s = tf.reshape(s, [-1, self.n_width, self.n_channel])
            n_filter = 32

            with tf.variable_scope('conv1') as scope:
                k1 = tf.get_variable('kernel1', shape=[1, 1, self.n_channel, n_filter], collections=c_names)
                conv1 = tf.nn.conv2d(s, k1, strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('conv2') as scope:
                k2_1 = tf.get_variable('kernel2_1', shape=[1, 1, self.n_channel, n_filter], collections=c_names)
                conv2 = tf.nn.conv2d(s, k2_1, strides=[1, 1, 1, 1], padding='SAME')
                k2_2 = tf.get_variable('kernel2_2', shape=[3, 3, n_filter, n_filter], collections=c_names)
                conv2 = tf.nn.conv2d(conv2, k2_2, strides=[1, 1, 1, 1], padding='SAME')
                k2_3 = tf.get_variable('kernel2_3', shape=[3, 3, n_filter, n_filter], collections=c_names)
                conv2 = tf.nn.conv2d(conv2, k2_3, strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('conv3') as scope:
                k3_1 = tf.get_variable('kernel3_1', shape=[1, 1, self.n_channel, n_filter], collections=c_names)
                conv3 = tf.nn.conv2d(s, k3_1, strides=[1, 1, 1, 1], padding='SAME')
                k3_2 = tf.get_variable('kernel3_2', shape=[5, 5, n_filter, n_filter], collections=c_names)
                conv3 = tf.nn.conv2d(conv3, k3_2, strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('conv4') as scrope:
                conv4 = tf.layers.average_pooling2d(s, [1, 3], [1, 1], padding='SAME')
                k4 = tf.get_variable('kernel4', shape=[1, 1, self.n_channel, n_filter], collections=c_names)
                conv4 = tf.nn.conv2d(conv4, k4, strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('concat') as scope:
                inception1 = tf.concat([conv1, conv2, conv3, conv4], axis=3)
                bias = tf.get_variable(name='biases', initializer=tf.constant_initializer(), shape=[4 * n_filter], collections=c_names)
                inception1 = tf.nn.relu(tf.nn.bias_add(inception1, bias))
                fc = tf.layers.average_pooling2d(inception1, [1, 8], [1, 8], padding='SAME')
                # fc = tf.contrib.layers.flatten(fc)
                fc = tf.reshape(fc, (tf.shape(fc)[0], self.n_width, n_filter * 8))

            with tf.variable_scope('rnn') as scope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_filter, state_is_tuple=True)
                state_in = cell.zero_state(tf.shape(fc)[0], tf.float32)
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
                rnn, state = tf.nn.dynamic_rnn(inputs=fc, cell=cell, dtype=tf.float32, initial_state=state_in)
                fc = state[1]
                # fc = tf.contrib.layers.flatten(rnn)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [n_filter, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                fc = tf.nn.relu(tf.matmul(fc, w1) + b1)

            if self.dueling:
                with tf.variable_scope('Value'):
                    w_out = tf.get_variable('w_out', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b_out = tf.get_variable('b_out', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(fc, w_out) + b_out

                with tf.variable_scope('Advantage'):
                    w_out = tf.get_variable('w_out', [n_l1, self.n_action], initializer=w_initializer, collections=c_names)
                    b_out = tf.get_variable('b_out', [1, self.n_action], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(fc, w_out) + b_out

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_action], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_action], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        with self.graph.as_default() as graph:

            if self.prioritized:
                self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.s = tf.placeholder(tf.float32, [None, self.n_width, self.n_height, self.n_channel], name='s')  # input
            self.q_target = tf.placeholder(tf.float32, [None, self.n_action], name='Q_target')  # for calculating loss

            with tf.variable_scope('eval_net'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 32, \
                    tf.contrib.layers.xavier_initializer(), tf.random_normal_initializer()

                self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

            with tf.variable_scope('loss'):
                if self.prioritized:
                    self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                    self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None, self.n_width, self.n_height, self.n_channel], name='s_')    # input
            with tf.variable_scope('target_net'):
                c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

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

                histogram_summary_tags = ['r_actions']

                for tag in histogram_summary_tags:
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_') + '_0')
                    self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

    def store_transition(self, s, a, r, s_):

        transition = {'s': s, 'a': a, 'r': r, 's_': s_}

        if self.prioritized:
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation, random=False):

        if np.random.uniform() > self.epsilon or random is True:  # choosing action
            action = np.random.randint(0, self.n_action)
        else:
            observation = observation[np.newaxis, :]
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        s = np.array([batch_memory[i]['s'] for i in range(self.batch_size)])
        s_ = np.array([batch_memory[i]['s_'] for i in range(self.batch_size)])

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: s_, self.s: s_})

        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = np.array([batch_memory[i]['a'] for i in range(self.batch_size)], dtype=np.int32)
        reward = np.array([batch_memory[i]['r'] for i in range(self.batch_size)])

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                        feed_dict={self.s: s,
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_width],
                                                    self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        self.totalLoss += self.cost
        self.totalQ += q_eval.mean(axis = 1).mean(axis = 0)
        self.totalMaxQ += np.max(q_eval, axis=1).mean()

    def inject_summary(self, tag_dict, episode):

        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })

        for summary_str in summary_str_lists:
            self.summary_writer.add_summary(summary_str, episode)

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
