import numpy as np
import math
import sys

from collections import deque
from math import log

class Portfolio:

    # order = []
    history = []
    hist_price = 0
    hist_sharpe_ratio = 0

    # def __init__(self, sess, balance, position, order_price):
    def __init__(self, balance, floating_pl, position, order_price, hurdle_rate=0.1):

        # self.sess = sess
        self.hist_balance = balance
        self.balance = balance
        self.floating_pl = 0
        # self.profit_loss = 0
        self.total_balance = balance + floating_pl
        self.position = position
        self.order_price = order_price
        self.contract_size = 100000
        self.hurdle_rate = hurdle_rate
        self.counter = 0

        self.n_order = 0

        self.stat = {
            'win': 0,
            'win_buy': 0,
            'win_sell': 0,
            'sharpe_ratio': 0,
            'total_balance': 0,
            'n_trades': 0,
            'n_buy': 0,
            'n_sell': 0,
            'count': 0,
            'max_win': 0,
            'max_lose': 0,
            'reward': 0
        }

        self.order = deque()
        self.hist_profit_loss = deque()

        print ('Initial balance: {}, Position: {}, Order Price: {}'.format(self.balance, self.position, self.order_price))

    def get_reward(self, price, action, position):

        # print('action: {}, price_prev: {}, price: {}, self.positon: {}'.format(action, price_prev, price, self.position))

        self.stat['count'] += 1
        reward = 0

        # if action != 0 and position != 0:
        #     self.stat['n_trades'] += 1

        # if (action == 1 and self.position >= 0):
        #     self.stat['n_buy'] += 1
        #     self.stat['n_trades'] += 1
        #     self.open_buy(position, price[1])
        # elif (action == 3 and self.position < 0):
        #     # self.stat['n_buy'] += 1
        #     return self.settle_buy(price[1])
        # elif (action == 2 and self.position <= 0):
        #     self.stat['n_sell'] += 1
        #     self.stat['n_trades'] += 1
        #     self.open_sell(position, price[0])
        # elif (action == 3 and self.position > 0):
        #     # self.stat['n_sell'] += 1
        #     return self.settle_sell(price[0])

        if (action == 1 and self.position == 0):
            self.stat['n_buy'] += 1
            self.stat['n_trades'] += 1
            self.open_buy(position, price[1])
        if (action == 2 and self.position > 0):
            self.stat['n_buy'] += 1
            self.stat['n_trades'] += 1
            self.open_buy(position, price[1])
        elif (action == 3 and self.position == 0):
            self.stat['n_sell'] += 1
            self.stat['n_trades'] += 1
            self.open_sell(position, price[0])
        elif (action == 4 and self.position < 0):
            self.stat['n_sell'] += 1
            self.stat['n_trades'] += 1
            self.open_sell(position, price[0])
        elif (action == 5 and self.position > 0):
            # self.stat['n_sell'] += 1
            reward = self.settle_sell(price[0])
        elif (action == 6 and self.position < 0):
            # self.stat['n_buy'] += 1
            reward = self.settle_buy(price[1])

        if self.total_balance < 250 and self.position != 0:
            reward = self.settle_buy(price[1]) if self.position < 0 else self.settle_sell(price[0])
        # elif (action == 7 and self.position > 0 and self.floating_pl < 0):
        #     # self.stat['n_sell'] += 1
        #     return self.settle_sell(price[0])
        # elif (action == 7 and self.position < 0 and self.floating_pl < 0):
        #     # self.stat['n_buy'] += 1
        #     return self.settle_buy(price[1])
        # reward = self.update_stat(price)
        self.update_stat(price)
        self.stat['reward'] += reward
        # print('Reward for action[{}]: {}'.format(action, reward))

        return reward

    def open_buy(self, position, price):

        # position = self.cal_position_size(1)
        self.add_order(position, price)

    def open_sell(self, position, price):

        # position = self.cal_position_size(-1)
        self.add_order(position, price)

    def settle_buy(self, price):
        return self.close_order(price)

    def settle_sell(self, price):
        return self.close_order(price)

    def add_order(self, position, price):

        if (self.position != 0):
            self.order_price = price * position/(self.position + position) + self.order_price * self.position/(self.position + position)
        else:
            self.order_price = price

        # self.book_order(position, price)
        self.position += position
        self.n_order += 1

        # direction = 'Buy' if position > 0 else 'Sell'
        # print('Open {} {}@{}, holdings: {}'.format(direction, position, price, self.position))

    def close_order(self, price):

        # if self.position > 0:
        #     pips = (price - self.order_price)
        # elif self.position < 0:
        #     pips = (self.order_price - price)
        pips = price - self.order_price
        profit_loss = pips * self.position
        self.balance += profit_loss
        # self.profit_loss += profit_loss
        # self.history.append({
        #     'open_price': self.order_price,
        #     'close_price': price,
        #     'position': self.position,
        #     'pips': pips * self.contract_size,
        #     'profit_loss': profit_loss
        # })
        # print('history: {}'.format(self.history))
        if profit_loss > 0:
            self.stat['win'] += self.n_order
            self.stat['max_win'] = max(self.stat['max_win'], profit_loss)
            if self.position > 0:
                self.stat['win_buy'] += self.n_order
            else:
                self.stat['win_sell'] += self.n_order
        else:
            self.stat['max_lose'] = min(self.stat['max_lose'], profit_loss)
        # self.book_order(self.position, price)

        # direction = 'Sell' if self.position > 0 else 'Buy'
        # print('Settle {} {}@{}, P/L: {}'.format(direction, self.position * -1, price, profit_loss))

        self.hist_profit_loss.append(profit_loss)
        self.mean_return = np.mean(self.hist_profit_loss)
        self.std = np.std(self.hist_profit_loss)
        self.sharpe_ratio = self.mean_return / self.std if self.std > 0 else 0
        diff_sharpe_ratio = self.sharpe_ratio - self.hist_sharpe_ratio

        self.n_order = 0
        self.position = 0
        self.order_price = 0
        self.hist_sharpe_ratio = self.sharpe_ratio
        self.stat['sharpe_ratio'] = self.sharpe_ratio
        # return profit_loss
        return diff_sharpe_ratio * 10000
    # def book_order(self, position, price):
    #
    #     try:
    #         if price <= 0:
    #             raise ValueError('Invalid position size or price')

            # self.order.append({
            #     'position': position,
            #     'price': price
            # })
        # except ValueError:
        #     print('position: {}, price: {}'.format(position, price))

    # def cal_position_size(self, direction):
        # print('self.balance/0.1 - abs(self.position) = {}'.format(self.balance/0.1 - abs(self.position)))
        # print('max(0, self.balance/0.1 - abs(self.position)) = {}'.format(max(0, self.balance/0.1 - abs(self.position))))
        # print('min(2000, max(0, self.balance/0.1 - abs(self.position))= {})'.format(min(2000, max(0, self.balance/0.1 - abs(self.position)))))
        # return int(min(float(self.balance)/0.2, max(0, float(self.balance)/0.2 - float(abs(self.position))))) * direction;

    def update_stat(self, price):

        if (self.order_price > 0 and self.position != 0):
            quote = price[1] if self.position < 0 else price[0]
            self.floating_pl = (quote - self.order_price) * self.position
        else:
            self.floating_pl = 0

        self.total_balance = self.balance + self.floating_pl
        episode_return = (self.total_balance - self.hist_balance) / self.hist_balance

        # log_episode_return = log(self.total_balance/self.hist_balance)

        # if len(self.hist_profit_loss) > 500:
        #     self.hist_profit_loss.rotate(np.random.randint(0, len(self.hist_profit_loss)))
        #     self.hist_profit_loss.popleft()
        #
        # self.hist_profit_loss.append(episode_return)
        # #
        # self.mean_return = np.mean(self.hist_profit_loss) if self.counter > 0 else 0
        # self.abs_mean_return = np.mean(np.absolute(self.hist_profit_loss)) if self.counter > 0 else 0
        # self.std_dev = np.std(self.hist_profit_loss)
        # self.sharpe_ratio = 0
        #
        # if (self.std_dev != 0 and not math.isnan(self.std_dev)):
        #     self.sharpe_ratio = self.mean_return / self.std_dev

        self.stat['total_balance'] = self.total_balance
        # diff_sharpe_ratio = self.sharpe_ratio
        # # print('self.mean_return: {}, self.std_dev: {}'.format(self.mean_return, self.std_dev))
        # # print('sharpe_ratio: {}'.format(self.sharpe_ratio))
        # if (len(self.hist_sharpe_ratio) > 1):
        #     diff_sharpe_ratio = float(diff_sharpe_ratio) - float(np.mean(self.hist_sharpe_ratio))
        #
        # if (len(self.hist_sharpe_ratio) > 500):
        #     self.hist_sharpe_ratio.rotate(np.random.randint(0, len(self.hist_sharpe_ratio)))
        #     self.hist_sharpe_ratio.popleft()
        #
        # self.hist_sharpe_ratio.append(self.sharpe_ratio)

        self.counter += 1
        # self.hist_price = price
        # self.hist_balance = self.total_balance
        #
        # if (len(self.hist_sharpe_ratio) > 1):
        #     self.mean_sharpe_ratio = np.mean(self.hist_sharpe_ratio)

        # self.stat['sharpe_ratio'] = self.sharpe_ratio
        # self.stat['sharpe_ratio'] = self.mean_sharpe_ratio
        # print('counter: {}, balance: {}, position: {}, mean_return: {}, std_dev: {}, \n\
        #     sharpe_ratio: {}, diff_sharp_ratio: {}'.format(self.counter, round(self.balance, 2), \
        #     self.position, round(mean_return, 2), round(std_dev, 2), \
        #     round(sharpe_ratio, 2), round(diff_sharpe_ratio, 2)))

        # reward = diff_sharpe_ratio * 100
        # reward = log(self.total_balance/self.hist_balance)
        # reward = self.mean_return / self.abs_mean_return if self.abs_mean_return > 0 else 0

        # if math.isnan(reward):
        #     print('reward is nan, setting its value to 0')
        #     reward = 0
        #
        # return reward

if __name__ == '__main__':

    portfolio = Portfolio(1000, 0, 0)

    # portfolio.get_reward(1.3000, 0, 1, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3000, 1.3001, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3029, 1.3000, 1, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3004, 1.3029, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3004, 1.3004, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3004, 1.3004, 2, portfolio.cal_position_size(2))
    # portfolio.get_reward(1.3004, 1.3004, 2, portfolio.cal_position_size(2))
    # portfolio.get_reward(1.3007, 1.3004, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3001, 1.3007, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3009, 1.3001, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3004, 1.3009, 0, portfolio.cal_position_size(1))
    # portfolio.get_reward(1.3000, 1.3004, 1, portfolio.cal_position_size(1))
    #
    # # print(portfolio.hist_profit_loss)
