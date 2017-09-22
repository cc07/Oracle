import unittest
from Portfolio import Portfolio

class PortfolioTest(unittest.TestCase):

    def setUp(self):
        self.portfolio = Portfolio(1000, 0, 0, 0)
        # self.portfolio.get_reward([1.3001, 1.3003], 1, 10000)
        # self.portfolio.get_reward([1.3005, 1.3006], 0, 10000)

    # def test_balance(self):
    #     self.assertAlmostEqual(self.portfolio.total_balance, 1002.0, 7,
    #         'Incorrect total_balance: {}, Expected: 1002'.format(self.portfolio.total_balance))
    #
    #     self.portfolio.get_reward([1.3009, 1.3011], 0, 10000)
    #
    #     self.assertAlmostEqual(self.portfolio.total_balance, 1006.0, 7,
    #         'Incorrect total_balance: {}, Expected: 1006'.format(self.portfolio.total_balance))
    #
    #     self.portfolio.get_reward([1.2995, 1.3011], 0, 10000)
    #
    #     self.assertAlmostEqual(self.portfolio.total_balance, 992.0, 7,
    #         'Incorrect total_balance: {}, Expected: 992'.format(self.portfolio.total_balance))
    #
    #     self.portfolio.get_reward([1.2995, 1.3011], 5, -10000)
    #
    #     self.assertAlmostEqual(self.portfolio.total_balance, 992.0, 7,
    #         'Incorrect total_balance: {}, Expected: 992'.format(self.portfolio.total_balance))

    # def test_floating_pl(self):
    #     self.assertAlmostEqual(self.portfolio.floating_pl, 2.0, 7,
    #         'Incorrect floating_pl: {}, Expected: 2'.format(self.portfolio.floating_pl))
    #
    #     self.portfolio.get_reward([1.3009, 1.3011], 0, 10000)
    #
    #     self.assertAlmostEqual(self.portfolio.floating_pl, 6, 7,
    #         'Incorrect floating_pl: {}, Expected: 6'.format(self.portfolio.floating_pl))
    #
    #     self.portfolio.get_reward([1.2995, 1.3011], 0, 10000)
    #
    #     self.assertAlmostEqual(self.portfolio.floating_pl, -8, 7,
    #         'Incorrect floating_pl: {}, Expected: -8'.format(self.portfolio.floating_pl))
    #
    #     self.portfolio.get_reward([1.2995, 1.3011], 2, -10000)
    #     self.portfolio.get_reward([1.2995, 1.3011], 2, -10000)
    #
    #     self.assertAlmostEqual(self.portfolio.floating_pl, -16, 7,
    #         'Incorrect floating_pl: {}, Expected: -6'.format(self.portfolio.floating_pl))
    #
    # def test_closed_pl(self):
    #     reward = self.portfolio.get_reward([1.2995, 1.3011], 2, -10000)
    #     self.assertAlmostEqual(self.portfolio.profit_loss, -8, 7,
    #         'Incorrect profit_loss: {}, Expected: -6'.format(self.portfolio.profit_loss))
    #
    #     self.portfolio.get_reward([1.2995, 1.3011], 2, -10000)
    #     self.portfolio.get_reward([1.2990, 1.2991], 2, -10000)
    #
    #     self.portfolio.get_reward([1.2980, 1.2981], 1, 20000)
    #
    #     self.assertAlmostEqual(self.portfolio.profit_loss, 15, 7,
    #         'Incorrect profit_loss: {}, Expected: -6'.format(self.portfolio.profit_loss))
    #
    # def test_reward(self):
    #
    #     # 0 - Hold, 1 - Open Buy, 2 - Open Buy, 3 - Open Sell, 4 - Open Sell, 5 - Settle Sell, 6 - Settle Buy
    #
    #     self.portfolio = Portfolio(1000, 0, 0, 0)
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3011], 1, 10000)
    #     self.assertAlmostEqual(reward, 0, 0,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3013], 2, 10000)
    #     self.assertAlmostEqual(reward, 0, 7,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.3032, 1.3011], 5, 0)
    #     self.assertAlmostEqual(reward, 1.9802627296179764, 7,
    #         'Incorrect reward: {}, Expected: 1'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3011], 1, 20000)
    #     self.assertAlmostEqual(reward, 0, 0,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3013], 2, 20000)
    #     self.assertAlmostEqual(reward, 0, 7,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.3027, 1.3011], 5, 0)
    #     self.assertAlmostEqual(reward, 1.904819497069621, 7,
    #         'Incorrect reward: {}, Expected: 2'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3011], 1, 10000)
    #     self.assertAlmostEqual(reward, 0, 0,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3013], 2, 10000)
    #     self.assertAlmostEqual(reward, 0, 7,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.3042, 1.3011], 5, 0)
    #     self.assertAlmostEqual(reward, 3.571808260208087, 7,
    #         'Incorrect reward: {}, Expected: 1'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2995, 1.3011], 1, 10000)
    #     self.assertAlmostEqual(reward, 0, 0,
    #         'Incorrect reward: {}, Expected: 0'.format(reward))
    #
    #     reward = self.portfolio.get_reward([1.2991, 1.3011], 5, 0)
    #     self.assertAlmostEqual(reward, -2.620237, 7,
    #         'Incorrect reward: {}, Expected: 1'.format(reward))

    def test_pmg(self):

        action = 0
        price = [1.2000, 1.2001]
        position = 5000
        price_ = [1.2000, 1.2001]
        emaFast = 1.2000
        emaSlow = 1.2000

        reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom = self.portfolio.get_reward(action, price, position, price_, emaFast, emaSlow)
        self.assertAlmostEqual(profit_make_good, 0, 7,
            'Incorrect pmg: {}, Expected: 0'.format(profit_make_good))

        action = 1
        price = [1.2000, 1.2001]
        reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom = self.portfolio.get_reward(action, price, position, price_, emaFast, emaSlow)
        self.assertAlmostEqual(profit_make_good, -1, 7,
            'Incorrect pmg: {}, Expected: -1'.format(profit_make_good))
        self.portfolio.book_ep_stat(action, price, reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom)
        self.portfolio.book_record(price, action, position, price_)

        action = 1
        price = [1.2002, 1.2003]
        reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom = self.portfolio.get_reward(action, price, position, price_, emaFast, emaSlow)
        self.assertAlmostEqual(profit_make_good, 0.333222231476, 7,
            'Incorrect pmg: {}, Expected: 0.333222231476'.format(profit_make_good))
        self.portfolio.book_ep_stat(action, price, reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom)
        self.portfolio.book_record(price, action, position, price_)

        action = 1
        price = [1.2000, 1.2001]
        reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom = self.portfolio.get_reward(action, price, position, price_, emaFast, emaSlow)
        self.assertAlmostEqual(profit_make_good, -0.667236806598, 7,
            'Incorrect pmg: {}, Expected: -0.667236806598'.format(profit_make_good))
        self.portfolio.book_ep_stat(action, price, reward, log_return, profit_make_good, diff_sharpe, diff_sharpe_top, diff_sharpe_bottom)
        self.portfolio.book_record(price, action, position, price_)


if __name__ == '__main__':
    unittest.main()
