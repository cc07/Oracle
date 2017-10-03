# from http.server import BaseHTTPRequestHandler, HTTPServer
# from urlparse import parse_qs
import sys, os
import getopt

import math
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

from collections import deque
from sklearn.externals import joblib
from math import log
from random import randint

sys.path.append('/Users/cc/documents/ds/oracle/src/model')
sys.path.append('/Users/cc/documents/ds/oracle/src/util')

from ddpg import DDPG
from portfolio import Portfolio

# sys.path.append('/Users/cc/documents/ds/matryoshka/')
# sys.path.append('/home/alchemist/matryoshka/')
#
# from rhythm import Rhythm

def run(load_sess=False, output_graph=True):

    print ('Loading csv data')

    data_train = joblib.load('./data/data_predicted.pkl')
    price_train = joblib.load('./data/price.pkl')
    X_train = joblib.load('./data/observation.pkl')

    print ('Loading finished')

    n_action = 3
    n_width = 13
    n_height = 1
    n_channel = 1

    LR_A = 0.001
    LR_C = 0.002
    GAMMA = 0.9
    TAU = 0.01
    MEMORY_CAPACITY = 10000
    var = 3

    epoches = 10000
    display_interval = 240
    batch_size = 1000
    learn_interval = 10

    step = 0
    history = []

    initial_balance = 1000
    position_base = 5000
    capacity_factor = 0.2
    leverage_factor = 1

    save_interval = 10000

    print('Initializing Oracle')
    oracle = DDPG(n_action, n_width, memory_capacity=MEMORY_CAPACITY, tau=TAU, \
        reward_decay=GAMMA, learning_rate_actor=LR_A, learning_rate_critic=LR_C)

    n_lookback = 9 #X.shape[1]
    n_feature = 32 #X.shape[2]
    n_channel = 4 #X.shape[3]
    n_output = 7 #y.shape[1]

    print('Initializing Rhythm')
    # rhythm = Rhythm(n_lookback, n_feature, n_channel, n_output)
    rhythm = None

    print('Restoring sess for Rhythm')
    # rhythm.load()

    # if (load_sess):
    # oracle.load(959)

    last_save_step = 0
    # X_train = np.reshape(X_train, (-1, )

    total_batch = int(X_train.shape[0] / batch_size)
    data_batches = np.array_split(data_train, total_batch)
    price_batches = np.array_split(price_train, total_batch)
    X_batches = np.array_split(X_train, total_batch)

    terminated = False

    for epoch in range(epoches):

        print('Epoch: {}'.format(epoch))

        observation = None
        goldkeeper = Portfolio(initial_balance, 0, 0, 0, position_base=position_base, capacity_factor=capacity_factor, leverage_factor=leverage_factor)
        action = None
        state = np.array([])
        state_ = np.array([])
        env = deque([], maxlen=9)
        warm_up = 0

        start = randint(0, total_batch - 50) if terminated == True else 0

        for b in range(start, total_batch):

            if goldkeeper.total_balance < 500:
                print('Balance less than 500, starting another epoch')
                terminated = True
                break;

            if observation is None:
                dataset = data_batches[b][0]
                price = price_batches[b][0][2:4]
                emaFast = price_batches[b][0][4]
                emaSlow = price_batches[b][0][5]
                observation = observe_environment(rhythm, goldkeeper, X_batches[b][0], price, dataset, emaFast, emaSlow)
                # env.append(observation)

            # print ('total_batch: {}, len(X_train): {}'.format(total_batch, len(X_train)))
            # print ('len(X_batches[b]): {}'.format(len(X_batches[b])))

            for i in range(len(X_batches[b])):

                # print('price: {}/{}'.format(price[0], price[1]))
                # print('observation: {}'.format(observation))

                # print('dataset: {}'.format(dataset))
                # print('price: {}'.format(price))
                # print('observation: {}'.format(observation))

                if goldkeeper.total_balance < 500 or (i == len(X_batches[b]) and b == total_batch):
                    break;

                action = oracle.choose_action(observation)
                selected_action = np.clip(np.random.normal(action, var), 0, 1)
                selected_action = np.argmax(selected_action, axis=1)

                # selected_action = np.argmax(action, axis=0)
                # print(selected_action)
                # action = oracle.choose_action(np.expand_dims(state, axis=3), True if warm_up < n_width else False)

                leverage_factor = goldkeeper.total_balance / initial_balance
                position = int(max(0, min(position_base * leverage_factor, (goldkeeper.total_balance / capacity_factor) - abs(goldkeeper.position))))

                if selected_action != 0 and abs(goldkeeper.position) == 0 and position < 5000:
                    position = 5000

                try:
                    dataset_ = data_batches[b][i+1] if (i < len(data_batches[b]) - 1) else data_batches[b+1][0]
                    price_ = price_batches[b][i+1][2:4] if (i < len(price_batches[b]) - 1) else price_batches[b+1][0][2:4]
                    emaFast_ = price_batches[b][i+1][4] if (i < len(price_batches[b]) - 1) else price_batches[b+1][0][4]
                    emaSlow_ = price_batches[b][i+1][5] if (i < len(price_batches[b]) - 1) else price_batches[b+1][0][5]

                    reward = goldkeeper.get_reward(selected_action, price, position, price_, emaFast, emaSlow)
                    goldkeeper.book_record(price, selected_action, position, price_)

                    observation_ = X_batches[b][i+1] if (i < len(X_batches[b]) - 1) else X_batches[b+1][0]
                    observation_ = observe_environment(rhythm, goldkeeper, observation_, price_, dataset_, emaFast_, emaSlow_)
                    # env.append(observation_)

                    # state_ = np.array(list(env))

                    # print('========')
                    # print('o: {}'.format(observation))
                    # print('s: {}'.format(state))
                    # print('o_: {}'.format(observation_))
                    # print('s_: {}'.format(state_))

                    # if warm_up > n_width:
                    #     oracle.store_transition(np.expand_dims(state, axis=3), action, reward, np.expand_dims(state_, axis=3))

                    oracle.store_transition(observation, action, reward, observation_)

                except Exception as e:
                    print(str(e))
                    print('Error b: {}, max_b: {}, i: {}, length: {}, step: {}'.format(b, total_batch, i, len(X_train), step))
                    terminated = False
                    break

                dataset = dataset_
                price = price_
                emaFast = emaFast_
                emaSlow = emaSlow_
                observation = observation_
                state = state_
                warm_up += 1

                if step > MEMORY_CAPACITY and step % learn_interval == 0:
                    var *= .999995
                    oracle.learn()

                if i % display_interval == 0:
                    print ('Epoch: {}, Batch: {}, Balance: {}, Position: {}, Trades: {}, Buy: {}, Sell: {}'.format( \
                        epoch, b, int(goldkeeper.total_balance), \
                        goldkeeper.position, goldkeeper.stat['n_trades'], \
                        goldkeeper.stat['n_buy'], goldkeeper.stat['n_sell']))

                # if step > MEMORY_CAPACITY and oracle.learn_step_counter % save_interval == 0 and last_save_step != oracle.learn_step_counter:
                    # oracle.save()
                    # last_save_step = oracle.learn_step_counter

                step += 1

        # oracle.finish_episode(epoch, goldkeeper.stat)

def observe_environment(rhythm, goldkeeper, base_observation, price, dataset, emaFast, emaSlow):

    # observation = np.hstack((rhythm.predict(np.array([dataset])), goldkeeper.get_environment()))
    mid = (price[0] + price[1]) / 2
    # observation = np.hstack((dataset, goldkeeper.get_environment(), log(emaFast/emaSlow), log(mid/emaFast), log(mid/emaSlow)))
    observation = np.hstack((dataset, goldkeeper.get_environment(), log(emaFast/emaSlow), log(mid/emaFast), log(mid/emaSlow)))
    # observation = np.hstack((observation, base_observation))
    return observation

if __name__ == '__main__':

    load_sess = True
    output_graph = True

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            'cd:hl:ot',
            ['dir=', 'help'],
        )
    except getopt.GetoptError:
        print('python run.py -h')
        sys.exit(2)

    for opt, arg in opts:
        # if opt in ('-c'):
            # FLAG['isTrain'] = False
            # FLAG['episodes'] = 1
            # FLAG['outputClose'] = True
        # elif opt in ('-d', '--dir'):
            # FLAG['dir'] = arg
        if opt in ['-h', '--help']:
            print('python3 train/index.py')
            print('-d <dir>: assign the output directory to data/<dir>')
            print('-h --help: help')
            print('-l <num>: load ckpt file from <dir>/history/<num>')
            print('-o: command line output result')
            print('-t: testing mode, default: training mode')
            sys.exit()
        elif opt in ('-l'):
            load_sess = True
            # FLAG['ckptFile'] = 'history/%s/train.ckpt' % (arg)
            # FLAG['loadHisNum'] = int(arg)
        elif opt in ('-o'):
            output_graph = True
            # FLAG['isTrain'] = False
            # FLAG['episodes'] = 1
            # FLAG['cliOutput'] = True
        # elif opt in ('-t'):
            # FLAG['isTrain'] = False
            # FLAG['episodes'] = 1

    print('load_sess: {}'.format(load_sess))
    print('output_graph: {}'.format(output_graph))

    # run(load_sess, output_graph)
    run(True, output_graph)
