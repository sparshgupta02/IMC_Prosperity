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
            'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100, 'JAMS': 350,
            'DJEMBES': 60, 'CROISSANTS': 250
        }
        self.spreads = {
            'RAINFOREST_RESIN': 3, 'KELP': 1, 'SQUID_INK': 1,
            'PICNIC_BASKET1': 2, 'PICNIC_BASKET2': 2, 'JAMS': 1,
            'DJEMBES': 1.5, 'CROISSANTS': 1.5
        }
        self.bcontents = {'PICNIC_BASKET1':{'CROISSANTS': 6, 'DJEMBES':1, 'JAMS':3}, 'PICNIC_BASKET2':{'CROISSANTS': 4, 'JAMS':2}}
        self.history = {product: [] for product in self.max_positions}
        self.max_so_far = {product: float('-inf') for product in self.history}
        self.min_so_far = {product: float('inf') for product in self.history}
        self.traderData = "MarketMakingV1"
        self.realized_pnl = {product: 0 for product in self.history}
        self.losses = {product: 0 for product in self.history}
        self.realized_pnl = {product: 0 for product in self.history}
        self.pnl_prev_tot = {product: 0 for product in self.history}
        self.fair_price = {product: 0 for product in self.history}
        self.resin_mean = 10000
        self.resin_count = 1
        self.imbalance = {product: 0.5 for product in self.history}

    def update_ema(self, product, price, alpha=0.3):
        prev_ema = self.history[product][-1] if self.history[product] else price
        new_ema = alpha * price + (1 - alpha) * prev_ema
        self.history[product].append(new_ema)

    def update_pnl(self, product, state, fair_price):
        position = state.position.get(product, 0)
        
        # Use fair price for unrealized PnL instead of last trade price
        unrealized_pnl = position * fair_price
        
        # Calculate realized PnL from new trades since last time
        realized_pnl = 0
        if product in state.own_trades:
            for i in range(len(state.own_trades[product])):
                trade = state.own_trades[product][i]
                if trade.timestamp != state.timestamp - 100:
                    continue
                if trade.seller == "SUBMISSION":
                    # You sold: money in
                    realized_pnl += trade.quantity * trade.price
                    # position -= trade.quantity
                else:
                    # You bought: money out
                    realized_pnl -= trade.quantity * trade.price
                        # position += trade.quantity
            self.realized_pnl[product] += realized_pnl
            total_pnl = self.realized_pnl[product] + unrealized_pnl
            if self.pnl_prev_tot[product] < total_pnl:
                self.losses[product] = 0
            else:
                self.losses[product] += 1
            self.pnl_prev_tot[product] = (total_pnl) + 0.5*int(total_pnl%1 != 0)

    def resin_logic(self, state, fair_price, product, imbalance):
        order_depth: OrderDepth = state.order_depths[product]
        position = state.position.get(product, 0)                   
        # Estimate fair value using best bid/ask
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
        else:
            best_ask = best_bid = self.resin_mean
        buy_price = best_ask
        sell_price = best_bid
        self.resin_count+=1
        self.resin_mean=(self.resin_mean*(self.resin_count-1)+fair_price)/self.resin_count
        if(best_bid>self.resin_mean):
            sell_volume = int(self.max_positions[product] + position)
        else:
            sell_volume = 0
        if(best_ask<self.resin_mean):
            buy_volume =  int(self.max_positions[product] - position)
        else:
            buy_volume = 0
        return buy_price, sell_price, buy_volume, sell_volume
    
    def product_logic_factory(self, product):
        logic_map = {
            'RAINFOREST_RESIN': self.resin_logic,
            'KELP': self.common_logic,
            'SQUID_INK': self.common_logic,
            'PICNIC_BASKET1': self.common_logic, 
            'PICNIC_BASKET2': self.common_logic, 
            'JAMS': self.common_logic,
            'DJEMBES': self.common_logic, 
            'CROISSANTS': self.common_logic
        }
        return logic_map.get(product)
    
    def common_logic(self, state, fair_price, product, imbalance):
        spread = self.spreads[product]
        max_position = self.max_positions[product]
        # Calculate buy/sell prices and volumes
        buy_price = int(fair_price - spread*2*imbalance)
        sell_price = int(fair_price + spread*2*(1-imbalance))
        position = state.position.get(product, 0)                   
        buy_volume = int(max_position - position)
        sell_volume = int(max_position + position)

        #self.history[product].append(fair_price)
        if len(self.history[product]) == 1001:
            self.history[product].pop(0) 
            prev_slope = (self.history[product][900] - self.history[product][0])/900
            curr_slope = (self.history[product][-1] - self.history[product][-101])/100
            sudden_change = (self.history[product][-1] - self.history[product][-11])/10        
            volatility = abs(sudden_change)
            # spread = max(spread, int(volatility*3))
            
            # if prev_slope != 0:
            if (fair_price < self.max_so_far[product]-5 and abs(curr_slope)<0.03):
                spread = 1
                # buy_price = int(fair_price - spread)
                buy_price = int(self.history[product][-1] - spread)
                # buy_volume = max_position - position
                sell_volume = 0

            elif (fair_price > self.min_so_far[product]+5 and abs(curr_slope)<0.03):
                spread = 1
                # sell_price = int(fair_price + spread)
                sell_price = int(self.history[product][-1] + spread)
                # sell_volume = max_position + position
                buy_volume = 0

            if volatility*100>0.1:
                buy_volume = int((max_position - position)*0.8)
                sell_volume = int((max_position + position)*0.8)
            
            if self.pnl_prev_tot[product]<-50:
                buy_volume = int((max_position - position)*0.3)
                sell_volume = int((max_position + position)*0.3)
            
            if self.pnl_prev_tot[product]>50:
                buy_volume = int((max_position - position))
                sell_volume = int((max_position + position))
        return buy_price, sell_price, buy_volume, sell_volume
    
    def baskets_decision(self, fair_price, basket):
        difference = (0.05*self.fair_price[basket] + 0.95*self.history[basket][-1]) if self.history[basket] else self.fair_price[basket]
        # difference = self.fair_price[basket]
        for item in self.bcontents[basket]:
            qty = self.bcontents[basket][item]
            price = (0.05*self.fair_price[item] + 0.95*self.history[item][-1]) if self.history[item] else self.fair_price[item]
            # price = self.fair_price[item]
            difference -= qty*price
        decision = difference>0
        return decision
    
    def run(self, state: TradingState):
        result = {}
        conversions = 1
        
        for product in state.order_depths:   
            order_depth: OrderDepth = state.order_depths[product]
            # Estimate fair value using best bid/ask
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                self.fair_price[product] = (best_ask + best_bid) / 2
                ask_vol = order_depth.sell_orders[best_ask]
                bid_vol = order_depth.buy_orders[best_bid]
                self.imbalance[product] = abs(ask_vol)/(abs(ask_vol)+abs(bid_vol)) if abs(ask_vol)+abs(bid_vol)>0 else 0.5
            else:
                self.fair_price[product] = state.own_trades[product][-1].price  # fallback or historical average
            fair_price = self.fair_price[product]
            Trader.update_pnl(self, product, state, fair_price)
            # self.pnl_total += self.pnl[product] + unrealized_pnl
            Trader.update_ema(self, product, fair_price)

            if fair_price > self.max_so_far[product]:
                self.max_so_far[product] = fair_price

            if fair_price < self.min_so_far[product]:
                self.min_so_far[product] = fair_price

            # Define your spread and position limits
        for product in ('RAINFOREST_RESIN', 'SQUID_INK', 'KELP'):
            orders: List[Order] = []
            logic_function = self.product_logic_factory(product)
            buy_price, sell_price, buy_volume, sell_volume = logic_function(state, self.fair_price[product], product, self.imbalance[product])

            if buy_volume > 0:
                print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))

            if sell_volume > 0:
                print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))  
            result[product] = orders
    
        for basket in ('PICNIC_BASKET1', 'PICNIC_BASKET2'):
            decision = Trader.baskets_decision(self, self.fair_price[basket], basket)
            orders: List[Order] = []
            product = basket
            logic_function = self.product_logic_factory(product)
            buy_price, sell_price, buy_volume, sell_volume = logic_function(state, self.fair_price[product], product, self.imbalance[product])
            sell_volume *= int(decision)
            buy_volume *= 1-int(decision)
            if buy_volume > 0:
                print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))
            if sell_volume > 0:
                print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))  
            result[product] = orders
            for product in self.bcontents[basket]:
                orders: List[Order] = []
                logic_function = self.product_logic_factory(product)
                buy_price, sell_price, buy_volume, sell_volume = logic_function(state, self.fair_price[product], product, self.imbalance[product])

                # sell_volume *= (1-int(decision))
                # buy_volume *= int(decision)
                if buy_volume > 0:
                    print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                    orders.append(Order(product, buy_price, buy_volume))
                if sell_volume > 0:
                    print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                    orders.append(Order(product, sell_price, -sell_volume))  
                result[product] = orders

        logger.print(sum(self.pnl_prev_tot[k] for k in ('PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANTS', 'DJEMBES', 'JAMS')))
        logger.flush(state, result, conversions, self.traderData)
        return result, conversions, self.traderData