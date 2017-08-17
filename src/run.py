# from http.server import BaseHTTPRequestHandler, HTTPServer
# from urlparse import parse_qs
import sys
import getopt

import math
import pandas as pd
import tensorflow as tf
import numpy as np

from DQN import DeepQNetwork
from Portfolio import Portfolio

# class HTTPServer_RequestHandler(BaseHTTPRequestHandler):
#
#     oracle = None
#
#     # def __init__(self, *args):
#         # tf.reset_default_graph()
#
#     def do_GET(self):
#
#         self.send_response(200)
#         self.send_header('Content-type','text/html')
#         self.end_headers()
#
#         # self.wfile.write(bytes(self.path, 'utf8'))
#         msg = '{}'.format(self.oracle.test())
#         self.wfile.write(bytes(msg, 'utf8'))
#         # message = "Hello world!"
#         # self.wfile.write(bytes(message, "utf8"))
#         return

# def getReward(price, price_prev, action, balance, position):
#
#     size = min(10000, balance/0.1)
#     reward = 0
#     #     balance = account['balance']
#     #     position = account['position']
#     #     print ('price: {}, price_next: {}, position: {}'.format(price, price_next, position))
#
#     if action == 1:
#         if (position >= 0):
#             #             print ('Open Buy {}@{}'.format(size, price))
#             position += size
#         elif (position < 0):
#             #             print ('Settle Buy {}@{}, Balance: {}'.format(size, position, balance + reward))
#             position = 0
#     elif action == 2:
#         if (position <= 0):
#             #             print ('Open Sell {}@{}'.format(size, price))
#             position += size * -1
#         elif (position > 0):
#             #             print ('Settle Sell {}@{}, Balance: {}'.format(size, position, balance + reward))
#             position = 0
#
#     if (position != 0 and price_prev > 0):
#         reward = (price - price_prev) * position
#     else:
#         reward = 0
#
#     balance = balance + reward
#
#     #     return reward, balance, position
#     return reward, {'balance': balance, 'position': position}

def run(load_sess=False, output_graph=True):

    print ('Loading csv data')
    train = pd.read_csv('./data/quote_M15.csv')
    print ('Loading finished')

    # X_train = train[1:100001]
    X_train = train[1:40000]
    # X_train = X_train.drop(['Timestamp', 'Date', 'Time'], axis=1)
    n_actions = 7
    n_features = X_train.shape[1]
    MEMORY_SIZE = 100000
    # e_greedy_increment = 0.000001
    e_greedy_increment = 0.00002
    reward_decay = 0.95
    learning_rate = 0.00005
    replace_target_iter = 10000
    dueling = True
    prioritized = True

    epoches = 1000
    display_interval = 240
    batch_size = 1000
    learn_interval = 50

    step = 0
    history = []

    initial_balance = 1000
    position_base = 500
    capacity_factor = 0.1
    leverage_factor = 1

    save_interval = 10000

    oracle = DeepQNetwork(n_actions=n_actions, n_features=n_features, memory_size=MEMORY_SIZE,
        reward_decay=reward_decay, learning_rate=learning_rate, replace_target_iter=replace_target_iter,
        e_greedy_increment=e_greedy_increment, dueling=dueling, output_graph=output_graph, prioritized=prioritized)

    if (load_sess):
        oracle.load()

    saver = tf.train.Saver()
    last_save_step = 0

    for epoch in range(epoches):

        print('Epoch: {}'.format(epoch))

        goldkeeper = Portfolio(initial_balance, 0, 0, 0)
        action = None

        total_batch = int(X_train.shape[0] / batch_size)
        X_batches = np.array_split(X_train.values, total_batch)

        for b in range(total_batch):

            if goldkeeper.balance < 200:
                print('Balance less than 200, starting another epoch')
                break;

            dataset = X_batches[b]
            price = dataset[0][0:2]
            observation = dataset[0][2:]

            if goldkeeper.position == 0:
                holding_status = 0
            elif goldkeeper.position > 0:
                holding_status = 1
            elif goldkeeper.position < 0:
                holding_status = 2

            observation = np.hstack((observation, holding_status))

            holding_capacity = abs(goldkeeper.position) / (goldkeeper.total_balance / capacity_factor)
            observation = np.hstack((observation, holding_capacity))

            for i in range(len(dataset)):

                if goldkeeper.balance < 200 or (i == len(dataset) and b == total_batch):
                    break;

                action = oracle.choose_action(observation)
                direction = 1

                if action == 3 or action == 4 or action == 5:
                    direction = -1

                leverage_factor = goldkeeper.total_balance / initial_balance
                position = int(max(0, min(position_base * leverage_factor, (goldkeeper.total_balance / capacity_factor) - abs(goldkeeper.position)))) * direction

                reward = goldkeeper.get_reward(price, action, position)

                try:
                    price = dataset[i+1][:2] if (i < len(dataset) - 1) else X_batches[b+1][0][:2]
                    observation_ = dataset[i+1][2:] if (i < len(dataset) - 1) else X_batches[b+1][0][2:]

                    if goldkeeper.position == 0:
                        holding_status = 0
                    elif goldkeeper.position > 0:
                        holding_status = 1
                    elif goldkeeper.position < 0:
                        holding_status = 2

                    observation_ = np.hstack((observation_, holding_status))

                    holding_capacity = abs(goldkeeper.position) / (goldkeeper.total_balance / capacity_factor)
                    observation_ = np.hstack((observation_, holding_capacity))

                    oracle.store_transition(observation, action, reward, observation_)
                except Exception as e:
                    print(str(e))
                    print('Error b: {}, max_b: {}, i: {}, length: {}'.format(b, total_batch, i, len(dataset)))
                    break

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

    # host = '127.0.0.1'
    # port = 8000
    # print('starting server...{}:{}'.format(host, port))
    #
    # HTTPServer_RequestHandler.oracle = oracle
    # server_address = (host, port)
    # httpd = HTTPServer(server_address, HTTPServer_RequestHandler)
    #
    # print('running server...')
    # httpd.serve_forever()

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
