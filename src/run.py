# from http.server import BaseHTTPRequestHandler, HTTPServer
# from urlparse import parse_qs
import sys, os
import getopt

import math
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.externals import joblib

from DQN import DeepQNetwork
from Portfolio import Portfolio

sys.path.append('/Users/cc/documents/ds/matryoshka/')

from rhythm import Rhythm

def run(load_sess=False, output_graph=True):

    print ('Loading csv data')
    # train = pd.read_csv('./data/quote_M15.csv')
    # data_train = joblib.load('./data/data.pkl')
    data_train = joblib.load('./data/data_predicted.pkl')
    # price_train = pd.read_csv('./data/quote.csv', header=None)
    price_train = joblib.load('./data/price.pkl')
    X_train = joblib.load('./data/observation.pkl')
    # featureM1 = pd.read_csv('./data/quoteM1.csv')
    # featureM5 = pd.read_csv('./data/quoteM5.csv')
    # featureM15 = pd.read_csv('./data/quoteM15.csv')
    # featureH1 = pd.read_csv('./data/quoteH1.csv')

    # data = [None] * len(featureM1)
    # for i in range(len(featureM1)):
    #     data[i] = np.array([
    #                 featureM1.loc[i].as_matrix(),
    #                 featureM5.loc[i].as_matrix(),
    #                 featureM15.loc[i].as_matrix(),
    #                 featureH1.loc[i].as_matrix()], np.float64)

    # X_train = pickle.load(open('./data/data.pkl', 'rb'))
    # price_train = price_train.as_matrix()
    print ('Loading finished')

    # X_train = train[1:100001]
    # X_train = train[1:40000]
    # X_train = X_train.drop(['Timestamp', 'Date', 'Time'], axis=1)
    n_action = 3
    n_feature = X_train.shape[1] + 10
    n_lookback = 9
    n_channel = 1

    MEMORY_SIZE = 10000
    e_greedy_increment = 0.0001
    reward_decay = 0.995
    learning_rate = 0.0001
    replace_target_iter = 20000
    dueling = True
    prioritized = True

    epoches = 1000
    display_interval = 240
    batch_size = 1000
    learn_interval = 10

    step = 0
    history = []

    initial_balance = 1000
    position_base = 500
    capacity_factor = 0.1
    leverage_factor = 1

    save_interval = 10000

    print('Initializing Oracle')
    oracle = DeepQNetwork(n_action=n_action, n_lookback=n_lookback, n_feature=n_feature, n_channel=n_channel, memory_size=MEMORY_SIZE,
        reward_decay=reward_decay, learning_rate=learning_rate, replace_target_iter=replace_target_iter,
        e_greedy_increment=e_greedy_increment, dueling=dueling, output_graph=output_graph, prioritized=prioritized)

    n_lookback = 9 #X.shape[1]
    n_feature = 32 #X.shape[2]
    n_channel = 1 #X.shape[3]
    n_output = 7 #y.shape[1]

    print('Initializing Rhythm')
    # rhythm = Rhythm(n_lookback, n_feature, n_channel, n_output)
    rhythm = None

    print('Restoring sess for Rhythm')
    # rhythm.load()

    # if (load_sess):
    #     oracle.load()

    last_save_step = 0
    # X_train = np.reshape(X_train, (-1, )

    total_batch = int(X_train.shape[0] / batch_size)
    data_batches = np.array_split(data_train, total_batch)
    price_batches = np.array_split(price_train, total_batch)
    X_batches = np.array_split(X_train, total_batch)

    for epoch in range(epoches):

        print('Epoch: {}'.format(epoch))

        goldkeeper = Portfolio(initial_balance, 0, 0, 0, position_base=position_base, capacity_factor=capacity_factor, leverage_factor=leverage_factor)
        action = None

        for b in range(total_batch):

            if goldkeeper.balance < 200:
                print('Balance less than 200, starting another epoch')
                break;

            # dataset = X_batches[b]
            # price = dataset[0][0:2]
            # observation = dataset[0][2:]
            dataset = data_batches[b][0]
            price = price_batches[b][0][2:4]
            ema = price_batches[b][0][4]
            observation = observe_environment(rhythm, goldkeeper, X_batches[b][0], price, dataset)

            # print ('total_batch: {}, len(X_train): {}'.format(total_batch, len(X_train)))
            # print ('len(X_batches[b]): {}'.format(len(X_batches[b])))

            for i in range(len(X_batches[b])):

                # print('price: {}/{}'.format(price[0], price[1]))
                # print('observation: {}'.format(observation))

                # print('dataset: {}'.format(dataset))
                # print('price: {}'.format(price))
                # print('observation: {}'.format(observation))

                if goldkeeper.balance < 200 or (i == len(X_batches[b]) and b == total_batch):
                    break;

                action = oracle.choose_action(observation)

                leverage_factor = goldkeeper.total_balance / initial_balance
                position = int(max(0, min(position_base * leverage_factor, (goldkeeper.total_balance / capacity_factor) - abs(goldkeeper.position))))

                try:
                    dataset_ = data_batches[b][i+1] if (i < len(data_batches[b]) - 1) else data_batches[b+1][0]
                    price_ = price_batches[b][i+1][2:4] if (i < len(price_batches[b]) - 1) else price_batches[b+1][0][2:4]
                    ema_ = price_batches[b][i+1][4] if (i < len(price_batches[b]) - 1) else price_batches[b+1][0][4]

                    reward = goldkeeper.get_reward(action, price, position, price_, ema)

                    # print('reward: {}, price: {}, price_: {}'.format(reward, price, price_))
                    goldkeeper.book_record(price, action, position, price_)

                    observation_ = X_batches[b][i+1] if (i < len(X_batches[b]) - 1) else X_batches[b+1][0]
                    observation_ = observe_environment(rhythm, goldkeeper, observation_, price_, dataset_)

                except Exception as e:
                    print(str(e))
                    print('Error b: {}, max_b: {}, i: {}, length: {}, step: {}'.format(b, total_batch, i, len(X_train), step))
                    break


                # for idx in range(len(reward)):
                #     oracle.store_transition(observation, idx, reward[idx], observation_)

                oracle.store_transition(observation, action, reward, observation_)

                dataset = dataset_
                price = price_
                ema = ema_
                observation = observation_

                if step > MEMORY_SIZE and step % learn_interval == 0:
                    oracle.learn()

                if i % display_interval == 0:
                    print ('Epoch: {}, Batch: {}, Balance: {}, Position: {}, Trades: {}, Buy: {}, Sell: {}, Cost: {}'.format( \
                        epoch, b, int(goldkeeper.total_balance), \
                        goldkeeper.position, goldkeeper.stat['n_trades'], \
                        goldkeeper.stat['n_buy'], goldkeeper.stat['n_sell'], oracle.cost))

                if step > MEMORY_SIZE and oracle.learn_step_counter % save_interval == 0 and last_save_step != oracle.learn_step_counter:
                    oracle.save()
                    last_save_step = oracle.learn_step_counter

                step += 1

        oracle.finish_episode(epoch, goldkeeper.stat)

def observe_environment(rhythm, goldkeeper, base_observation, price, dataset):

    # observation = np.hstack((rhythm.predict(np.array([dataset])), goldkeeper.get_environment()))
    observation = np.hstack((dataset, goldkeeper.get_environment()))
    observation = np.hstack((observation, base_observation))
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
