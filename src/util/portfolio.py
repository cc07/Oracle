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

    hist_diff_sharpe_top = 0
    hist_diff_sharpe_bottom = 0

    # def __init__(self, sess, balance, position, order_price):
    def __init__(self, balance, floating_pl, position, order_price, hurdle_rate=0.1, position_base=500, capacity_factor=0.1, leverage_factor=1):

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
        self.position_base = position_base
        self.capacity_factor = capacity_factor
        self.leverage_factor = leverage_factor

        self.counter = 1
        self.n_order = 0

        self.sum_top = 0
        self.sum_bottom = 0

        self.holding_period = 1
        self.trend = None

        self.stat = {
            'n_win': 0,
            'win_buy': 0,
            'win_sell': 0,
            'sharpe_ratio': 0,
            'total_balance': 0,
            'n_trades': 0,
            'n_buy': 0,
            'n_up_buy': 0,
            'n_down_sell': 0,
            'n_sell': 0,
            'count': 0,
            'max_profit': 0,
            'total_profit': 0,
            'max_loss': 0,
            'total_loss': 0,
            'max_holding_period': 0,
            'total_holding_period': 0,
            'total_profit_holding_period': 0,
            'total_loss_holding_period': 0,
            'reward': 0,
            'diff_sharpe': 0,
            'max_floating_profit': 0,
            'max_floating_loss': 0,
            'max_total_balance': 0,
            'profit_make_good': 0
        }

        self.order = deque()
        self.hist_profit_loss = deque()
        self.ep_profit_loss = deque()
        self.price_prev = None

        print ('Initial balance: {}, Position: {}, Order Price: {}'.format(self.balance, self.position, self.order_price))

    def get_reward(self, action, price, position, price_, emaFast, emaSlow):

        profit_loss = 0
        mid = (price[0] + price[1]) / 2
        trend_incentive = 0.00001 * (self.holding_period ** 0.5)
        counter_trend_penalty = (price[0] - price[1]) * (abs(self.position) + position)
        additional_order_penalty = (price[0] - price[1]) * (abs(self.position) + position)
        holding_period_factor = (1 + (self.holding_period ** 0.5) / 10)

        self.trend = 1 if mid > emaSlow else 0
        self.price_prev = price if self.price_prev is None else self.price_prev

        if action == 0 and self.position > 0:
            profit_loss = (price_[0] - price[0]) * abs(self.position)
        elif action == 0 and self.position < 0:
            profit_loss = (price[1] - price_[1]) * abs(self.position)
        elif action == 1 and self.position >= 0: # open buy
            profit_loss = (price[0] - price[1]) * position + (price_[0] - price[0]) * abs(self.position)
        elif action == 1 and self.position < 0: # settle buy
            profit_loss = (price[1] - price_[1]) * abs(self.position)
        elif action == 2 and self.position <= 0: # open sell
            profit_loss = (price[0] - price[1]) * position + (price[1] - price_[1]) * abs(self.position)
        elif action == 2 and self.position > 0: # settle sell
            profit_loss = (price_[0] - price[0]) * abs(self.position)
        #
        # if profit_loss < 0:
        #     profit_loss = profit_loss * 1.5
        #
        # if profit_loss < 0 and self.floating_pl < 0 and self.position > 0 and mid < emaFast:
        #     profit_loss = profit_loss * 1.5
        # elif profit_loss < 0 and self.floating_pl < 0 and self.position < 0 and mid > emaFast:
        #     profit_loss = profit_loss * 1.5

        # if action == 1 and self.position < 0: # settle buy
        #     profit_loss += (self.order_price - price[1]) * abs(self.position)
        # elif action == 2 and self.position > 0:
        #     profit_loss += (price[0] - self.order_price) * abs(self.position)

        # trend_adjustment_factor = 1
        # loss_penalty_factor = 1.5
        # stop_loss_factor = 1.25
        #
        # if action == 0 and self.position > 0 and emaFast > emaSlow and self.floating_pl > 0:
        #     trend_adjustment_factor = 1.05
        # elif action == 0 and self.position < 0 and emaFast < emaSlow and self.floating_pl > 0:
        #     trend_adjustment_factor = 1.05
        # elif action == 1 and self.position >= 0 and emaFast > emaSlow and self.floating_pl > 0:
        #     trend_adjustment_factor = 1.05
        # elif action == 1 and self.position >= 0 and emaFast < emaSlow and self.floating_pl < 0:
        #     trend_adjustment_factor = 1.05
        # elif action == 2 and self.position <= 0 and emaFast < emaSlow and self.floating_pl > 0:
        #     trend_adjustment_factor = 1.05
        # elif action == 2 and self.position <= 0 and emaFast > emaSlow and self.floating_pl < 0:
        #     trend_adjustment_factor = 1.05
        #
        # if trend_adjustment_factor > 1:
        #     profit_loss += abs(self.position) * 0.00005 * trend_adjustment_factor
        #
        # if self.floating_pl < abs(self.position) * -0.0100:
        #     profit_loss += abs(self.position) * -0.0001
        # elif self.floating_pl < abs(self.position) * -0.0050:
        #     profit_loss += abs(self.position) * -0.00005
        #
        # if self.floating_pl < 0 and action == 1 and self.position < 0:
        #     profit_loss += abs(self.position) * -0.0005
        # elif self.floating_pl < 0 and action == 2 and self.position > 0:
        #     profit_loss += abs(self.position) * -0.0005

        if self.total_balance + profit_loss > 0:
            log_return = log(self.total_balance + profit_loss) - log(self.total_balance)
        else:
            log_return = log(self.total_balance) * -1

        reward = 0
        decay = 0.9
        profit_make_good = 0
        diff_sharpe = 0
        sum_top = 0
        sum_bottom = 0
        diff_sharpe_top = 0
        diff_sharpe_bottom = 0

        ################
        # profit_make_good_ = 0
        ################

        if profit_loss != 0:

            diff_sharpe_top = self.hist_diff_sharpe_top + decay * (log_return - self.hist_diff_sharpe_top)
            diff_sharpe_bottom = self.hist_diff_sharpe_bottom + decay * (log_return ** 2 - self.hist_diff_sharpe_bottom)
            diff_sharpe = diff_sharpe_top / diff_sharpe_bottom if diff_sharpe_bottom > 0 else 0

            # self.hist_diff_sharpe_top = diff_sharpe_top
            # self.hist_diff_sharpe_bottom = diff_sharpe_bottom
            #
            # self.stat['diff_sharpe'] = diff_sharpe

            ################
            # hist_return = list(self.ep_profit_loss)
            # hist_return.append(log_return)
            #
            # sum_top_ = np.sum(hist_return)
            # sum_bottom_ = np.sum(np.absolute(hist_return))
            #
            # profit_make_good_ = sum_top_ / sum_bottom_ if sum_bottom_ > 0 else 0

            # print('=========')
            # print('hist_return: {}'.format(hist_return))
            # print('sum_top_: {}'.format(sum_top_))
            # print('sum_bottom_: {}'.format(sum_bottom_))
            # print('profit_make_good_: {}'.format(profit_make_good_))
            # print('=========')
            ################

            sum_top = self.sum_top * ((float(self.counter) - 1) / float(self.counter)) + log_return * (1 / float(self.counter))
            sum_bottom = self.sum_bottom * ((float(self.counter) - 1) / float(self.counter)) + abs(log_return) * (1 / float(self.counter))

            profit_make_good = sum_top / sum_bottom if sum_bottom > 0 else 0

            # if action == 1 and self.position < 0:
            #     order_settle_reward = 1 if self.floating_pl > 0 else -1
            # elif action == 2 and self.position > 0:
            #     order_settle_reward = 1 if self.floating_pl > 0 else -1

            # reward = profit_make_good - self.stat['profit_make_good']
            # reward = profit_make_good
            # reward = diff_sharpe - self.stat['diff_sharpe']
            # reward = diff_sharpe
            # reward += order_settle_reward

            #####
            # self.stat['profit_make_good'] = profit_make_good
            # self.ep_profit_loss.append(log_return)

            # reward = diff_sharpe
            # if self.floating_pl < 0 and ((action == 1 and self.position < 0) or (action == 2 and self.position > 0)):
            #     log_return = log_return - 1
            #
            reward = log_return
            # reward = profit_make_good
            self.stat['profit_make_good'] = profit_make_good
            self.stat['diff_sharpe'] = diff_sharpe
            self.hist_diff_sharpe_top = diff_sharpe_top
            self.hist_diff_sharpe_bottom = diff_sharpe_bottom

            self.sum_top = self.sum_top * ((float(self.counter) - 1) / float(self.counter)) + log_return * (1 / float(self.counter))
            self.sum_bottom = self.sum_bottom * ((float(self.counter) - 1) / float(self.counter)) + abs(log_return) * (1 / float(self.counter))

            self.counter += 1

        self.stat['reward'] += reward
        self.price_prev = price

        ################
        # print('----------')
        # print('self.counter: {}'.format(self.counter))
        # print('self.position: {}'.format(self.position))
        # print('self.order_price: {}'.format(self.order_price))
        # print('log_return: {}'.format(log_return))
        # print('profit_loss: {}'.format(profit_loss))
        # print('----------')
        # print('price: {}'.format(price))
        # print('self.price_prev: {}'.format(self.price_prev))
        # print('----------')
        # print('reward: {}'.format(reward))
        # print('profit_make_good: {}'.format(profit_make_good))
        # print('sum_top: {:.10f}'.format(sum_top))
        # print('sum_bottom: {:.10f}'.format(sum_bottom))
        # print('self.sum_top: {:.10f}'.format(self.sum_top))
        # print('self.sum_bottom: {:.10f}'.format(self.sum_bottom))
        # print('----------')
        # print('profit_make_good_: {}').format(profit_make_good_)
        ################

        # return reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom
        #####
        return reward

    def book_ep_stat(self, action, price, reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom):
        if not (self.position == 0 and action == 0):
            # self.ep_profit_loss.append(log_return)
            self.stat['profit_make_good'] = profit_make_good
            self.stat['diff_sharpe'] = diff_sharpe
            self.hist_diff_sharpe_top = diff_sharpe_top
            self.hist_diff_sharpe_bottom = diff_sharpe_bottom

            self.sum_top = self.sum_top * ((float(self.counter) - 1) / float(self.counter)) + log_return * (1 / float(self.counter))
            self.sum_bottom = self.sum_bottom * ((float(self.counter) - 1) / float(self.counter)) + abs(log_return) * (1 / float(self.counter))

            self.counter += 1

        self.price_prev = price
        self.stat['reward'] += reward


    def book_record(self, price, action, position, price_):

        # print('action: {}, price_prev: {}, price: {}, self.positon: {}'.format(action, price_prev, price, self.position))

        self.stat['count'] += 1
        reward = 0

        if (abs(self.position) > 0):
            self.holding_period += 1

        if (action == 1 and self.position >= 0 and position > 0):
            self.stat['n_buy'] += 1
            self.stat['n_up_buy'] += 1 if self.trend == 1 else 0
            self.stat['n_trades'] += 1
            self.open_buy(position, price[1])
        elif (action == 1 and self.position < 0):
            self.settle_buy(price[1])
        elif (action == 2 and self.position <= 0 and position > 0):
            self.stat['n_sell'] += 1
            self.stat['n_down_sell'] += 1 if self.trend == 0 else 0
            self.stat['n_trades'] += 1
            self.open_sell(position, price[0])
        elif (action == 2 and self.position > 0):
            self.settle_sell(price[0])

        self.update_stat(price)

        # self.stat['reward'] += reward
        # print('Reward for action[{}]: {}'.format(action, reward))

        # return reward

    def open_buy(self, position, price):

        # position = self.cal_position_size(1)
        self.add_order(position, price)

    def open_sell(self, position, price):

        # position = self.cal_position_size(-1)
        self.add_order(position * -1, price)

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

        profit_loss = (price - self.order_price) * self.position
        # hurdle_return = 0.001 * abs(self.position)
        # print('hurdle: {}'.format(hurdle_return))
        log_return = log(self.balance + profit_loss) - log(self.balance)
        self.balance += profit_loss

        # print('profit_loss: {}'.format(profit_loss))
        # self.profit_loss += profit_loss
        # self.history.append({
        #     'open_price': self.order_price,
        #     'close_price': price,
        #     'position': self.position,
        #     'pips': pips * self.contract_size,
        #     'profit_loss': profit_loss
        # })
        # print('history: {}'.format(self.history))

        pips = profit_loss / abs(self.position) * 10000

        if profit_loss > 0:
            self.stat['n_win'] += self.n_order
            self.stat['max_profit'] = max(self.stat['max_profit'], pips)
            self.stat['total_profit'] += pips * self.n_order
            self.stat['total_profit_holding_period'] += self.holding_period * self.n_order
            if self.position > 0:
                self.stat['win_buy'] += self.n_order
            else:
                self.stat['win_sell'] += self.n_order
        else:
            self.stat['max_loss'] = min(self.stat['max_loss'], pips)
            self.stat['total_loss'] += pips * self.n_order
            self.stat['total_loss_holding_period'] += self.holding_period * self.n_order

        self.stat['max_holding_period'] = max(self.stat['max_holding_period'], self.holding_period)
        self.stat['total_holding_period'] += self.holding_period * self.n_order
        # self.book_order(self.position, price)

        # direction = 'Sell' if self.position > 0 else 'Buy'
        # print('Settle {} {}@{}, P/L: {}'.format(direction, self.position * -1, price, profit_loss))

        self.hist_profit_loss.append(log_return)

        self.mean_return = np.mean(self.hist_profit_loss)
        self.std = np.std(self.hist_profit_loss)
        # self.sharpe_ratio = (self.mean_return - 0.0010 * abs(self.position))
        self.sharpe_ratio = self.mean_return
        self.sharpe_ratio = self.sharpe_ratio / self.std if self.std > 0.0001 else self.sharpe_ratio / self.sharpe_ratio
        # diff_sharpe_ratio = self.sharpe_ratio - self.hist_sharpe_ratio

        # print('mean: {}, std: {}'.format(self.mean_return, self.std))
        # print('diff_sharpe_ratio: {}, sharpe_ratio: {}'.format(diff_sharpe_ratio, self.sharpe_ratio))
        self.n_order = 0
        self.position = 0
        self.order_price = 0
        self.holding_period = 1
        self.hist_sharpe_ratio = self.sharpe_ratio
        self.stat['sharpe_ratio'] = self.sharpe_ratio
        # return profit_loss

        # decay = 0.9
        # diff_sharpe_top = self.hist_diff_sharpe_top + decay * (log_return - self.hist_diff_sharpe_top)
        # diff_sharpe_bottom = self.hist_diff_sharpe_bottom + decay * (log_return ** 2 - self.hist_diff_sharpe_bottom)
        # diff_sharpe = diff_sharpe_top / diff_sharpe_bottom / 1000 if diff_sharpe_bottom > 0 else 0
        #
        # self.hist_diff_sharpe_top = diff_sharpe_top
        # self.hist_diff_sharpe_bottom = diff_sharpe_bottom
        # self.stat['diff_sharpe'] = diff_sharpe
        #
        # return diff_sharpe
        return log_return

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

        pips = self.floating_pl / abs(self.position) * 10000 if abs(self.position) > 0 else 0

        if self.floating_pl > 0:
            self.stat['max_floating_profit'] = max(self.stat['max_floating_profit'], pips)
        elif self.floating_pl < 0:
            self.stat['max_floating_loss'] = min(self.stat['max_floating_loss'], pips)

        self.total_balance = self.balance + self.floating_pl

        self.stat['max_total_balance'] = max(self.stat['max_total_balance'], self.total_balance)
        # episode_return = (self.total_balance - self.hist_balance) / self.hist_balance

        # hurdle_return = 0.001 * abs(self.position)
        # print('hurdle: {}'.format(hurdle_return))
        # log_return = log(self.total_balance - hurdle_return) - log(self.hist_balance)

        # decay = 0.9
        # diff_sharpe_top = self.hist_diff_sharpe_top + decay * (log_return - self.hist_diff_sharpe_top)
        # diff_sharpe_bottom = self.hist_diff_sharpe_bottom + decay * (log_return ** 2 - self.hist_diff_sharpe_bottom)
        # diff_sharpe = diff_sharpe_top / diff_sharpe_bottom / 1000 if diff_sharpe_bottom > 0 else 0
        #
        # self.hist_diff_sharpe_top = diff_sharpe_top
        # self.hist_diff_sharpe_bottom = diff_sharpe_bottom
        # self.stat['diff_sharpe'] = diff_sharpe

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

        # self.counter += 1
        # self.hist_price = price
        self.hist_balance = self.total_balance
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
        # return diff_sharpe

    def get_holding_status(self):
        if self.position == 0:
            holding_status = 0
        elif self.position > 0:
            holding_status = 1
        elif self.position < 0:
            holding_status = -1

        return holding_status

    def get_holding_capacity(self):
        return abs(self.position) / (self.total_balance / self.capacity_factor)

    def get_position_status(self):
        return 1 if self.floating_pl >= 0 else 0

    def get_environment(self):
        return np.hstack((self.get_holding_status(), self.get_holding_capacity(), self.get_position_status()))

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
