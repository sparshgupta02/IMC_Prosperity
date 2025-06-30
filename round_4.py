from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import random
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        
        self.max_positions = {
            'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
            'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
            'JAMS': 350, 'DJEMBES': 60, 'CROISSANTS': 250,
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200,
        }
        self.spreads = {
            'RAINFOREST_RESIN': 3, 'KELP': 1, 'SQUID_INK': 1,
            'PICNIC_BASKET1': 2, 'PICNIC_BASKET2': 2, 
            'JAMS': 1, 'DJEMBES': 1.5, 'CROISSANTS': 1.5,
            'VOLCANIC_ROCK': 0,
            'VOLCANIC_ROCK_VOUCHER_9500': 0,
            'VOLCANIC_ROCK_VOUCHER_9750': 0,
            'VOLCANIC_ROCK_VOUCHER_10000': 0,
            'VOLCANIC_ROCK_VOUCHER_10250': 0,
            'VOLCANIC_ROCK_VOUCHER_10500': 0,
        }
        self.strikes = {'VOLCANIC_ROCK_VOUCHER_9500': 9500,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500,}
        self.strikes = {'VOLCANIC_ROCK_VOUCHER_9500': 9500,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500,}
        self.base_iv_coeffs = {
            'VOLCANIC_ROCK_VOUCHER_9500': [2.06765829, 0.06773109, 0.02454177],
            'VOLCANIC_ROCK_VOUCHER_9750': [3.06261511, 0.1818794 , 0.02685499], 
            'VOLCANIC_ROCK_VOUCHER_10000': [2.17907067, 0.04759995, 0.02276154], 
            'VOLCANIC_ROCK_VOUCHER_10250': [2.05410965, -0.00219456, 0.02205753], 
            'VOLCANIC_ROCK_VOUCHER_10500': [2.80809273, -0.07365772, 0.0227483],
            }
        self.sdevs = {
            'VOLCANIC_ROCK_VOUCHER_9500': 0.004106866154383309,
            'VOLCANIC_ROCK_VOUCHER_9750': 0.0026443017627357864,
            'VOLCANIC_ROCK_VOUCHER_10000': 0.001551847639648702,
            'VOLCANIC_ROCK_VOUCHER_10250': 0.0011120441811421126,
            'VOLCANIC_ROCK_VOUCHER_10500': 0.0031968603359160083
        }
        self.tols = {
            'VOLCANIC_ROCK_VOUCHER_9500': 0.5,
            'VOLCANIC_ROCK_VOUCHER_9750': 0.5,
            'VOLCANIC_ROCK_VOUCHER_10000': 0.5,
            'VOLCANIC_ROCK_VOUCHER_10250': 0.5,
            'VOLCANIC_ROCK_VOUCHER_10500': 0.1
        }

        self.bcontents = {'PICNIC_BASKET1':{'CROISSANTS': 6, 'DJEMBES':1, 'JAMS':3}, 'PICNIC_BASKET2':{'CROISSANTS': 4, 'JAMS':2}}
        self.history = {product: [] for product in self.max_positions}
        self.ema = {product: [] for product in self.max_positions}
        self.max_so_far = {product: float('-inf') for product in self.history}
        self.min_so_far = {product: float('inf') for product in self.history}
        self.traderData = "MarketMakingV1"
        self.realized_pnl = {product: 0 for product in self.history}
        self.losses = {product: 0 for product in self.history}
        self.learnt_spreads = {product: 0 for product in self.history}
        self.realized_pnl = {product: 0 for product in self.history}
        self.pnl_prev_tot = {product: 0 for product in self.history}
        self.fair_price = {product: 0 for product in self.history}
        self.gains = {product: 0 for product in self.history}
        self.losses = {product: 0 for product in self.history}
        self.resin_mean = 10000
        self.resin_count = 1
        self.imbalance = {product: 0.5 for product in self.history}
        self.calc_spread = lambda x, y: x*(1.01**(1-y))
        self.diff = {product: 0 for product in self.history}
        self.kelp_window = []
        self.kelp_window_size = 10
        self.resin_window = []
        self.resin_window_size = 10

########################## Normal functions ############################
    def update_spread(self, product):
        learnt_spread = self.learnt_spreads[product]
        history = self.history[product]
        ema = self.ema[product]
        curr_slope = (ema[-1]-ema[-11])/10
        sudden_change = (history[-1] - history[-11])/10
        learnt_spread = (learnt_spread*(len(history)-1) + self.calc_spread(abs(ema[-1]-history[-1]), abs(sudden_change/curr_slope)))/len(history) if curr_slope!=0 else learnt_spread
        self.learnt_spreads[product] = learnt_spread
        
    def update_diff(self, product, new_diff, alpha=0.8):
        max_poss_profit = (self.max_so_far[product]-self.min_so_far[product])*self.max_positions[product]
        self.calc_spread(new_diff, self.pnl_prev_tot[product]/max_poss_profit) if max_poss_profit>0 else new_diff
        self.diff[product] = alpha*new_diff + (1-alpha)*self.diff[product] if self.diff[product] else new_diff
    
    def update_ema(self, product, price, alpha=0.25):
        prev_ema = self.ema[product][-1] if self.ema[product] else price
        new_ema = alpha * price + (1 - alpha) * prev_ema
        self.ema[product].append(new_ema)
        self.history[product].append(price)

    def update_pnl(self, product, state, fair_price):
        position = state.position.get(product, 0)
        unrealized_pnl = position * fair_price
        realized_pnl = 0
        if product in state.own_trades:
            for i in range(len(state.own_trades[product])):
                trade = state.own_trades[product][i]
                if trade.timestamp != state.timestamp - 100:
                    continue
                if trade.seller == "SUBMISSION":
                    # You sold: money in
                    realized_pnl += trade.quantity * trade.price
                else:
                    # You bought: money out
                    realized_pnl -= trade.quantity * trade.price
            self.realized_pnl[product] += realized_pnl
            total_pnl = self.realized_pnl[product] + unrealized_pnl
            if self.pnl_prev_tot[product] < total_pnl:
                self.losses[product] = 0
            else:
                self.losses[product] += 1
            self.pnl_prev_tot[product] = (total_pnl) + 0.5*int(total_pnl%1 != 0)
    
    def product_logic_factory(self, product):
        logic_map = {
            'RAINFOREST_RESIN': self.resin_logic,
            'KELP': self.kelp_logic,
            'SQUID_INK': self.squink_logic,
            'PICNIC_BASKET1': self.baskets_logic, 
            'PICNIC_BASKET2': self.baskets_logic, 
            'JAMS': self.common_logic,
            'DJEMBES': self.common_logic, 
            'CROISSANTS': self.common_logic,
            'VOLCANIC_ROCK': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_9500': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_9750': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_10000': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_10250': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_10500': self.common_logic,
        }
        return logic_map.get(product)
    
########################## BASKETS functions ############################
    def calculate_rsi(self, product, period=14):
        prices = self.ema[product]
        if len(prices) < period + 2:
            return None  # Not enough data
        delta = prices[-1]-prices[-2]
        rm_delta = prices[-1-period] - prices[-period-2]

        gain = max(delta, 0)
        loss = -min(delta, 0)

        rm_gain = max(rm_delta, 0)
        rm_loss = -min(rm_delta, 0)

        self.gains[product] += gain - rm_gain
        self.losses[product] += loss - rm_loss

        avg_gain = self.gains[product] / period
        avg_loss = self.losses[product] / period
        if avg_loss == 0:
            return 100  # No losses, RSI is maxed

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def baskets_logic(self, state, product):
        fair_price = self.fair_price[product]
        max_position = self.max_positions[product]
        position = state.position.get(product, 0) 
        imbalance = self.imbalance[product]
        buy_volume = int((max_position - position) * 0.8)
        sell_volume = int((max_position + position) * 0.8)
        rsi = Trader.calculate_rsi(self, product, period=40)
        if rsi:
            if rsi > 70:
                # Overbought — prepare to SELL or reduce exposure
                buy_volume = 0
                sell_volume = int((max_position + position) * 1)
            elif rsi < 30:
                # Oversold — prepare to BUY
                buy_volume = int((max_position - position) * 1)
                sell_volume = 0
            elif rsi > 50:
                # Bullish momentum
                buy_volume = int((max_position - position) * 0.3)
            elif rsi < 50:
                # Bearish momentum
                sell_volume = int((max_position + position) * 0.3)
     
        if self.pnl_prev_tot[product]<50:
            buy_volume = int((max_position - position))
            sell_volume = int((max_position + position))
        
        if self.pnl_prev_tot[product]<-100:
                buy_volume = int(buy_volume*(0.7**self.losses[product]))
                sell_volume = int(sell_volume*(0.7**self.losses[product]))
                # spread = self.losses[product]
        if self.pnl_prev_tot[product]<-500:
                buy_volume = int(buy_volume*(0.5**self.losses[product]))
                sell_volume = int(sell_volume*(0.5**self.losses[product]))

        spread = self.spreads[product]
        buy_price = int(fair_price - spread*2*imbalance)
        sell_price = int(fair_price + spread*2*(1-imbalance))
        if len(self.history[product]) == 102:
            self.history[product].pop(0)
            self.ema[product].pop(0)
            prev_slope = (self.ema[product][-1] - self.ema[product][-101])/100
            curr_slope = (self.ema[product][-1] - self.ema[product][-11])/10
            sudden_change = (self.history[product][-1] - self.history[product][-11])/10
            spread = (self.learnt_spreads[product] if self.learnt_spreads[product]!=0 else self.spreads[product])
            buy_price = int(fair_price - spread*2*imbalance)
            sell_price = int(fair_price + spread*2*(1-imbalance))
            prev_slope = max(10**-9, prev_slope)
            if fair_price < int(max(self.ema[product])-5*spread) and abs(curr_slope/prev_slope)<spread/10 or (fair_price <= min(self.ema[product])+2*spread and abs(curr_slope/prev_slope)>10*spread):
                buy_price = int(fair_price - spread*abs(sudden_change/prev_slope) + (curr_slope - self.diff[product]/2)*2*imbalance)

            elif fair_price > int(min(self.ema[product])+5*spread) and abs(curr_slope/prev_slope)<spread/10 or (fair_price >= max(self.ema[product])-2*spread and abs(curr_slope/prev_slope)>10*spread):
                sell_price = int(fair_price + spread*abs(sudden_change/prev_slope) + (curr_slope + self.diff[product]/2)*2*(1-imbalance))

        return buy_price, sell_price, buy_volume, sell_volume
        
    def baskets_decision(self, fair_price, basket):
        difference = (0.05*self.fair_price[basket] + 0.95*self.ema[basket][-1]) if self.ema[basket] else self.fair_price[basket]
        for item in self.bcontents[basket]:
            qty = self.bcontents[basket][item]
            price = (0.05*self.fair_price[item] + 0.95*self.ema[item][-1]) if self.ema[item] else self.fair_price[item]
            difference -= qty*price
        decision = difference>0
        return decision
    
########################## RESIN functions ############################    
    def resin_logic(self, state: TradingState, fair_price, product, imbalance):
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)                   
            # Estimate fair value using best bid/ask
            orders: List[Order] = []
            true_value = 10000

            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depth.sell_orders.items())
            if(len(buy_orders)==0 or len(sell_orders)==0): return orders

            to_buy = self.max_positions[product] - position
            to_sell = self.max_positions[product] + position

            self.resin_window.append(abs(position) == self.max_positions[product])
            if len(self.resin_window) > self.resin_window_size:
                self.resin_window.pop(0)

            soft_liquidate = len(self.resin_window) == self.resin_window_size and sum(self.resin_window) >= self.resin_window_size / 2 and self.resin_window[-1]
            hard_liquidate = len(self.resin_window) == self.resin_window_size and all(self.resin_window)

            max_buy_price = true_value - 1 if position > self.max_positions[product] * 0.5 else true_value
            min_sell_price = true_value + 1 if position < self.max_positions[product] * -0.5 else true_value

            for price, volume in sell_orders:
                if to_buy > 0 and price <= max_buy_price:
                    quantity = min(to_buy, -volume)
                    orders.append(Order('RAINFOREST_RESIN',int(price), quantity))
                    to_buy -= quantity

            if to_buy > 0 and hard_liquidate:
                quantity = to_buy // 2
                orders.append(Order('RAINFOREST_RESIN',int(true_value), quantity))
                to_buy -= quantity

            if to_buy > 0 and soft_liquidate:
                quantity = to_buy // 2
                orders.append(Order('RAINFOREST_RESIN',int(true_value - 2), quantity))
                to_buy -= quantity

            if to_buy > 0:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
                orders.append(Order('RAINFOREST_RESIN',int(price), to_buy))

            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    quantity = min(to_sell, volume)
                    orders.append(Order('RAINFOREST_RESIN',int(price), -quantity))
                    to_sell -= quantity

            if to_sell > 0 and hard_liquidate:
                quantity = to_sell // 2
                orders.append(Order('RAINFOREST_RESIN',int(true_value), -quantity))
                to_sell -= quantity

            if to_sell > 0 and soft_liquidate:
                quantity = to_sell // 2
                orders.append(Order('RAINFOREST_RESIN',int(true_value + 2), -quantity))
                to_sell -= quantity

            if to_sell > 0:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                orders.append(Order('RAINFOREST_RESIN',int(price), -to_sell))
            
            return orders
    
########################## KELP functions ############################    
    def kelp_logic(self, state: TradingState, fair_price, product, imbalance):
        order_depth: OrderDepth = state.order_depths[product]
        position = state.position.get(product, 0)                   
        # Estimate fair value using best bid/ask
        orders: List[Order] = []
        true_value = fair_price

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if(len(buy_orders)==0 or len(sell_orders)==0): return orders

        to_buy = self.max_positions[product] - position
        to_sell = self.max_positions[product] + position

        self.kelp_window.append(abs(position) == self.max_positions[product])
        if len(self.kelp_window) > self.kelp_window_size:
            self.kelp_window.pop(0)

        soft_liquidate = len(self.kelp_window) == self.kelp_window_size and sum(self.kelp_window) >= self.kelp_window_size / 2 and self.kelp_window[-1]
        hard_liquidate = len(self.kelp_window) == self.kelp_window_size and all(self.kelp_window)

        max_buy_price = true_value - 1 if position > self.max_positions[product] * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.max_positions[product] * -0.5 else true_value

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            orders.append(Order(product,int(true_value), quantity))
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            orders.append(Order(product,int(true_value - 2), quantity))
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price+1)
            orders.append(Order(product,int(price), to_buy))

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            orders.append(Order(product,int(true_value), -quantity))
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            orders.append(Order(product,int(true_value + 2), -quantity))
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price-1)
            orders.append(Order(product,int(price), -to_sell))        
        return orders

########################## SQUID INK functions ############################    
    def squink_logic(self, state, product):
        rsi = Trader.calculate_rsi(self, product, period=40)
        fair_price = self.fair_price[product]
        imbalance = self.imbalance[product]
        spread = self.spreads[product]
        max_position = self.max_positions[product]
        # Calculate buy/sell prices and volumes
        buy_price = int(fair_price - spread*2*imbalance)
        sell_price = int(fair_price + spread*2*(1-imbalance))
        position = state.position.get(product, 0)                   
        buy_volume = int((max_position - position)*0.8)
        sell_volume = int((max_position + position)*0.8)
        # Simple momentum logic
        if rsi:
            if rsi > 70:
                # Overbought — prepare to SELL or reduce exposure
                buy_volume = 0
                sell_volume = int((max_position + position) * 1)
            elif rsi < 30:
                # Oversold — prepare to BUY
                buy_volume = int((max_position - position) * 1)
                sell_volume = 0
            elif rsi > 50:
                # Bullish momentum
                buy_volume = int((max_position - position) * 0.3)
            elif rsi < 50:
                # Bearish momentum
                sell_volume = int((max_position + position) * 0.3)

        if len(self.history[product]) >= 102:
            self.history[product].pop(0)
            self.ema[product].pop(0)
            prev_slope = (self.ema[product][-1] - self.ema[product][-101])/100
            curr_slope = (self.ema[product][-1] - self.ema[product][-11])/10
            sudden_change = (self.history[product][-1] - self.history[product][-11])/10
            spread = (self.learnt_spreads[product] if self.learnt_spreads[product]!=0 else self.spreads[product])
            buy_price = int(fair_price - spread*2*imbalance)
            sell_price = int(fair_price + spread*2*(1-imbalance))
            prev_slope = max(10**-9, prev_slope)
            if fair_price < int(max(self.ema[product])-3*spread*abs(curr_slope)) and abs(fair_price-self.ema[product][-1])<spread*abs(curr_slope)/2 or (fair_price <= min(self.ema[product])+2*spread*abs(prev_slope) and abs(fair_price-self.ema[product][-1])>2*spread*abs(prev_slope)):
            # if (fair_price <= self.min_so_far[product]+2*spread and abs(curr_slope/prev_slope)>1.8):
                buy_price = int(self.ema[product][-1] - spread*2*imbalance + curr_slope)
                sell_volume = 0
                return buy_price, sell_price, buy_volume, sell_volume

            elif fair_price > int(min(self.ema[product])+3*spread*abs(curr_slope)) and abs(fair_price-self.ema[product][-1])<spread*abs(curr_slope)/2 or (fair_price >= max(self.ema[product])-2*spread*abs(prev_slope) and abs(fair_price-self.ema[product][-1])>2*spread*abs(prev_slope)):
            # elif (fair_price >= self.max_so_far[product]-2*spread and abs(curr_slope/prev_slope)>1.8):
                sell_price = int(self.ema[product][-1] + spread*2*(1-imbalance) + curr_slope)
                buy_volume = 0
                return buy_price, sell_price, buy_volume, sell_volume
                                                        
            if self.pnl_prev_tot[product]<-100:
                buy_volume = int((max_position - position)*(0.7**self.losses[product]))
                sell_volume = int((max_position + position)*(0.7**self.losses[product]))
                # spread = self.losses[product]
            if self.pnl_prev_tot[product]<-500:
                buy_volume = int((max_position - position)*(0.5**self.losses[product]))
                sell_volume = int((max_position + position)*(0.5**self.losses[product]))
            return buy_price, sell_price, buy_volume, sell_volume
    
        if (fair_price <= self.min_so_far[product]+2*spread) and abs(fair_price-self.ema[product][-1])>3*spread:
            buy_price = int(self.ema[product][-1] + spread*2*imbalance)
            return buy_price, sell_price, buy_volume, sell_volume

        elif (fair_price >= self.max_so_far[product]-2*spread) and abs(fair_price-self.ema[product][-1])>3*spread:
            sell_price = int(self.ema[product][-1] + spread*2*(1-imbalance))
            return buy_price, sell_price, buy_volume, sell_volume
                                                
        if self.pnl_prev_tot[product]<-100:
            buy_volume = int((max_position - position)*(0.9**self.losses[product]))
            sell_volume = int((max_position + position)*(0.9**self.losses[product]))
        if self.pnl_prev_tot[product]<-500:
            buy_volume = int((max_position - position)*(0.8**self.losses[product]))
            sell_volume = int((max_position + position)*(0.8**self.losses[product]))
        return buy_price, sell_price, buy_volume, sell_volume
    
########################## JAMS function ############################    
    def jams_logic(self, state, product):
        fair_price = self.fair_price[product]
        imbalance = self.imbalance[product]
        spread = self.spreads[product]
        max_position = self.max_positions[product]
        buy_price = int(fair_price - spread*2*imbalance)
        sell_price = int(fair_price + spread*2*(1-imbalance))
        position = state.position.get(product, 0)                   
        buy_volume = int(max_position - position)
        sell_volume = int(max_position + position)
        if fair_price < int(max(self.history[product])-5*spread) and abs(self.ema[product][-1] - fair_price)<spread/10 or (fair_price <= min(self.history[product])+2*spread and abs(self.ema[product][-1] - fair_price)>10*spread):
            buy_price = int(self.ema[product][-1] - spread*2*imbalance)
            sell_volume = 0
            return buy_price, sell_price, buy_volume, sell_volume

        elif fair_price > int(min(self.history[product])+5*spread) and abs(self.ema[product][-1] - fair_price)<spread/10 or (fair_price >= max(self.history[product])-2*spread and abs(self.ema[product][-1] - fair_price)>10*spread):
            sell_price = int(self.ema[product][-1] + spread*2*(1-imbalance))
            buy_volume = 0
            return buy_price, sell_price, buy_volume, sell_volume
                                                    
        if self.pnl_prev_tot[product]<-100:
            buy_volume = int((max_position - position)*(0.99**self.losses[product]))
            sell_volume = int((max_position + position)*(0.99**self.losses[product]))
        if self.pnl_prev_tot[product]<-500:
            buy_volume = int((max_position - position)*(0.99**self.losses[product]))
            sell_volume = int((max_position + position)*(0.99**self.losses[product]))
        return buy_price, sell_price, buy_volume, sell_volume

########################## COMMON LOGIC ############################    
    def common_logic(self, state, product):
        imbalance = self.imbalance[product]
        fair_price = self.fair_price[product]
        rsi = Trader.calculate_rsi(self, product, period=40)
        spread = self.spreads[product]
        max_position = self.max_positions[product]
        # Calculate buy/sell prices and volumes
        buy_price = int(fair_price - spread*2*imbalance)
        sell_price = int(fair_price + spread*2*(1-imbalance))
        position = state.position.get(product, 0)                   
        buy_volume = int((max_position - position)*0.8)
        sell_volume = int((max_position + position)*0.8)
        # Simple momentum logic
        if rsi:
            if rsi > 70:
                # Overbought — prepare to SELL or reduce exposure
                buy_volume = 0
                sell_volume = int((max_position + position) * 1)
            elif rsi < 30:
                # Oversold — prepare to BUY
                buy_volume = int((max_position - position) * 1)
                sell_volume = 0
            elif rsi > 50:
                # Bullish momentum
                buy_volume = int((max_position - position) * 0.3)
            elif rsi < 50:
                # Bearish momentum
                sell_volume = int((max_position + position) * 0.3)

        if len(self.history[product]) >= 102:
            self.history[product].pop(0)
            self.ema[product].pop(0)
            prev_slope = (self.ema[product][-1] - self.ema[product][-101])/100
            curr_slope = (self.ema[product][-1] - self.ema[product][-11])/10
            sudden_change = (self.history[product][-1] - self.history[product][-11])/10
            spread = (self.learnt_spreads[product] if self.learnt_spreads[product]!=0 else self.spreads[product])
            buy_price = int(fair_price - spread*2*imbalance)
            sell_price = int(fair_price + spread*2*(1-imbalance))
            prev_slope = max(10**-9, prev_slope)
            if fair_price < int(max(self.ema[product])-3*spread*abs(curr_slope)) and abs(fair_price-self.ema[product][-1])<spread*abs(curr_slope)/2 or (fair_price <= min(self.ema[product])+2*spread*abs(prev_slope) and abs(fair_price-self.ema[product][-1])>2*spread*abs(prev_slope)):
                buy_price = int(self.ema[product][-1] - spread*2*imbalance + curr_slope)
                sell_volume = 0
                return buy_price, sell_price, buy_volume, sell_volume

            elif fair_price > int(min(self.ema[product])+3*spread*abs(curr_slope)) and abs(fair_price-self.ema[product][-1])<spread*abs(curr_slope)/2 or (fair_price >= max(self.ema[product])-2*spread*abs(prev_slope) and abs(fair_price-self.ema[product][-1])>2*spread*abs(prev_slope)):
                sell_price = int(self.ema[product][-1] + spread*2*(1-imbalance) + curr_slope)
                buy_volume = 0
                return buy_price, sell_price, buy_volume, sell_volume
                                                        
            if self.pnl_prev_tot[product]<-100:
                buy_volume = int((max_position - position)*(0.7**self.losses[product]))
                sell_volume = int((max_position + position)*(0.7**self.losses[product]))

            if self.pnl_prev_tot[product]<-500:
                buy_volume = int((max_position - position)*(0.5**self.losses[product]))
                sell_volume = int((max_position + position)*(0.5**self.losses[product]))
            return buy_price, sell_price, buy_volume, sell_volume
    
        # if fair_price < int(max(self.ema[product])-5*spread) or (fair_price <= min(self.ema[product])+2*spread)>10*spread:
        if (fair_price <= self.min_so_far[product]+2*spread) and abs(fair_price-self.ema[product][-1])>3*spread:
            buy_price = int(self.ema[product][-1] + spread*2*imbalance)
            # sell_volume = 0
            return buy_price, sell_price, buy_volume, sell_volume

        # elif fair_price > int(min(self.ema[product])+5*spread) or (fair_price >= max(self.ema[product])-2*spread)>10*spread:
        elif (fair_price >= self.max_so_far[product]-2*spread) and abs(fair_price-self.ema[product][-1])>3*spread:
            sell_price = int(self.ema[product][-1] + spread*2*(1-imbalance))
            # buy_volume = 0
            return buy_price, sell_price, buy_volume, sell_volume
        return buy_price, sell_price, buy_volume, sell_volume
    
    def run(self, state: TradingState):
        result = {}
        conversions = 1
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            # Estimate fair value using best bid/ask
            if product not in self.history.keys():
                continue
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                self.fair_price[product] = (best_ask + best_bid) / 2
                ask_vol = order_depth.sell_orders[best_ask]
                bid_vol = order_depth.buy_orders[best_bid]
                spread = best_ask - best_ask
                self.imbalance[product] = abs(ask_vol)/(abs(ask_vol)+abs(bid_vol)) if abs(ask_vol)+abs(bid_vol)>0 else 0.5
            else:
                self.fair_price[product] = state.own_trades[product][-1].price  # fallback or historical average
            fair_price = self.fair_price[product]
            Trader.update_pnl(self, product, state, fair_price)
            Trader.update_ema(self, product, fair_price, alpha=0.25)
            Trader.update_diff(self, product, spread) 
            if len(self.history[product])>=11:
                Trader.update_spread(self, product) 
            if fair_price > self.max_so_far[product]:
                self.max_so_far[product] = fair_price

            if fair_price < self.min_so_far[product]:
                self.min_so_far[product] = fair_price
    
        for product in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
            if product not in state.order_depths:
                continue
            decision = Trader.baskets_decision(self, self.fair_price[product], product)
            orders: List[Order] = []
            buy_price, sell_price, buy_volume, sell_volume = Trader.baskets_logic(self, state, product)
            sell_volume *= int(decision)
            buy_volume *= 1-int(decision)
            if buy_volume > 0:
                print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))
            if sell_volume > 0:
                print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))  
            result[product] = orders

        for product in ['KELP']:
            order_depth = state.order_depths[product]
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depth.sell_orders.items())
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            popular_fair_price = round((popular_buy_price + popular_sell_price)/2)

            orders: List[Order] = []
            orders = self.kelp_logic(state, popular_fair_price, product, 0)
            result[product] = orders

        for product in ['RAINFOREST_RESIN']:
            orders: List[Order] = []
            orders = self.resin_logic(state, 10000, product, 0)
            result[product] = orders

        for product in ['SQUID_INK']:
            orders: List[Order] = []
            buy_price, sell_price, buy_volume, sell_volume = Trader.squink_logic(self, state, product)
            if buy_volume > 0:
                print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))
            if sell_volume > 0:
                print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))  
            result[product] = orders

        for product in ['JAMS', 'DJEMBES', 'CROISSANTS',
            'VOLCANIC_ROCK',
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500']:
            orders: List[Order] = []
            buy_price, sell_price, buy_volume, sell_volume = Trader.common_logic(self, state, product)
            if buy_volume > 0:
                print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))

            if sell_volume > 0:
                print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))  
            result[product] = orders
        logger.print(self.learnt_spreads)
        logger.flush(state, result, conversions, self.traderData)
        return result, conversions, self.traderData