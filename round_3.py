from statistics import NormalDist
from typing import List
import numpy as np
import json
from typing import Any, Dict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


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
        self.coupon_objects = {
            'VOLCANIC_ROCK_VOUCHER_9500': {
                "hedged" : False,
                "prev_coupon_price": 0
            },
            'VOLCANIC_ROCK_VOUCHER_9750': {
                "hedged" : False,
                "prev_coupon_price": 0
            },
            'VOLCANIC_ROCK_VOUCHER_10000': {
                "hedged" : False,
                "prev_coupon_price": 0
            },
            'VOLCANIC_ROCK_VOUCHER_10250': {
                "hedged" : False,
                "prev_coupon_price": 0
            },
            'VOLCANIC_ROCK_VOUCHER_10500': {
                "hedged" : False,
                "prev_coupon_price": 0
            }
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
        self.resin_mean = 10000
        self.resin_count = 1
        self.imbalance = {product: 0.5 for product in self.history}
        self.calc_spread = lambda x, y: x*(1.01**(1-0.8*y))

    def update_spread(self, product):
        learnt_spread = self.learnt_spreads[product]
        history = self.history[product]
        ema = self.ema[product]
        curr_slope = (ema[-1]-ema[-11])/10
        sudden_change = (history[-1] - history[-11])/10
        learnt_spread = (learnt_spread*(len(history)-1) + self.calc_spread(abs(ema[-1]-history[-1]), abs(sudden_change/curr_slope)))/len(history) if curr_slope!=0 else learnt_spread
        self.learnt_spreads[product] = learnt_spread
    
    def update_ema(self, product, price, alpha=0.25):
        prev_ema = self.ema[product][-1] if self.ema[product] else price
        new_ema = alpha * price + (1 - alpha) * prev_ema
        self.ema[product].append(new_ema)
        self.history[product].append(price)

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
            'CROISSANTS': self.common_logic,
            'VOLCANIC_ROCK': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_9500': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_9750': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_10000': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_10250': self.common_logic,
            'VOLCANIC_ROCK_VOUCHER_10500': self.common_logic,
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
        if len(self.history[product]) == 102:
            self.history[product].pop(0)
            self.ema[product].pop(0)
            prev_slope = (self.ema[product][-1] - self.ema[product][-101])/100
            curr_slope = (self.ema[product][-1] - self.ema[product][-11])/10
            sudden_change = (self.history[product][-1] - self.history[product][-11])/10
            spread = self.learnt_spreads[product]
            buy_price = int(fair_price - spread*2*imbalance)
            sell_price = int(fair_price + spread*2*(1-imbalance))
            
            # if (fair_price < self.max_so_far[product]-int(2*spread) and abs(curr_slope*100)<spread) or (fair_price <= self.min_so_far[product]+2*spread and abs(curr_slope/prev_slope)>1.8):
            if (fair_price <= self.min_so_far[product]+2*spread and abs(curr_slope/prev_slope)>1.8):
                buy_price = int(self.ema[product][-1] - spread*2*imbalance + prev_slope)
                sell_volume = 0
                return buy_price, sell_price, buy_volume, sell_volume

            # elif (fair_price > self.min_so_far[product]+int(2*spread) and abs(curr_slope*100)<spread) or (fair_price >= self.max_so_far[product]-2*spread and abs(curr_slope/prev_slope)>1.8):
            elif (fair_price >= self.max_so_far[product]-2*spread and abs(curr_slope/prev_slope)>1.8):
                sell_price = int(self.ema[product][-1] + spread*2*(1-imbalance) + prev_slope)
                buy_volume = 0
                return buy_price, sell_price, buy_volume, sell_volume
                                                        
            if self.pnl_prev_tot[product]<-100:
                buy_volume = int((max_position - position)*(0.9**self.losses[product]))
                sell_volume = int((max_position + position)*(0.9**self.losses[product]))
                # spread = self.losses[product]
            if self.pnl_prev_tot[product]<-500:
                buy_volume = int((max_position - position)*(0.99**self.losses[product]))
                sell_volume = int((max_position + position)*(0.99**self.losses[product]))
        return buy_price, sell_price, buy_volume, sell_volume
    
    def baskets_decision(self, fair_price, basket):
        difference = (0.05*self.fair_price[basket] + 0.95*self.ema[basket][-1]) if self.ema[basket] else self.fair_price[basket]
        # difference = self.fair_price[basket]
        for item in self.bcontents[basket]:
            qty = self.bcontents[basket][item]
            price = (0.05*self.fair_price[item] + 0.95*self.ema[item][-1]) if self.ema[item] else self.fair_price[item]
            # price = self.fair_price[item]
            difference -= qty*price
        decision = difference>0
        return decision
    
    def black_scholes_call(self, spot, strike, time_to_expiry, volatility):
        d1 = (np.log(spot) - np.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * np.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price
    
    def delta(self, spot, strike, time_to_expiry, volatility):
        d1 = (
            np.log(spot) - np.log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    def implied_volatility(self,
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = self.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    def get_coupon_mid_price(
            self,
            coupon_order_depth: OrderDepth,
            traderData: Dict[str, Any]
    ):
        if len(coupon_order_depth.buy_orders) > 0 and len(coupon_order_depth.sell_orders) > 0:
            best_bid = max(coupon_order_depth.buy_orders.keys())
            best_ask = min(coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]

    def delta_hedge_rock_position(
        self,
        rock_order_depth: OrderDepth,
        coupon_position: int,
        rock_position: int,
        rock_buy_orders: int,
        rock_sell_orders: int,
        delta: float,
        traderData: Dict[str, Any]
    ) -> List[Order]:
        """
        Delta hedge the overall position in COCONUT_COUPON by creating orders in COCONUT.

        Args:
            rock_order_depth (OrderDepth): The order depth for the COCONUT product.
            coupon_position (int): The current position in COCONUT_COUPON.
            rock_position (int): The current position in COCONUT.
            rock_buy_orders (int): The total quantity of buy orders for COCONUT in the current iteration.
            rock_sell_orders (int): The total quantity of sell orders for COCONUT in the current iteration.
            delta (float): The current value of delta for the COCONUT_COUPON product.
            traderData (Dict[str, Any]): The trader data for the COCONUT_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the COCONUT_COUPON position.
        """
        if traderData["hedged"] == True:
            return None

        target_rock_position = -int(delta * coupon_position)
        hedge_quantity = target_rock_position - (rock_position + rock_buy_orders - rock_sell_orders)

        orders: List[Order] = []
        if hedge_quantity > 0:
            # Buy COCONUT
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(hedge_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT["VOLCANIC_ROCK"] - (rock_position + rock_buy_orders))
            if quantity > 0:
                orders.append(Order("VOLCANIC_ROCK", best_ask, quantity))
        elif hedge_quantity < 0:
            # Sell COCONUT
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(hedge_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT["VOLCANIC_ROCK"] + (rock_position - rock_sell_orders))
            if quantity > 0:
                orders.append(Order("VOLCANIC_ROCK", best_bid, -quantity))

        return orders

    def delta_hedge_coupon_orders(
            self,
            rock_order_depth: OrderDepth,
            coconut_coupon_orders: List[Order],
            rock_position: int,
            rock_buy_orders: int,
            rock_sell_orders: int,
            delta: float,
            traderData: Dict[str, Any]
        ) -> List[Order]:
        """
        Delta hedge the new orders for COCONUT_COUPON by creating orders in COCONUT.

        Args:
            rock_order_depth (OrderDepth): The order depth for the COCONUT product.
            coconut_coupon_orders (List[Order]): The new orders for COCONUT_COUPON.
            rock_position (int): The current position in COCONUT.
            rock_buy_orders (int): The total quantity of buy orders for COCONUT in the current iteration.
            rock_sell_orders (int): The total quantity of sell orders for COCONUT in the current iteration.
            delta (float): The current value of delta for the COCONUT_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the new COCONUT_COUPON orders.
        """
        if traderData["hedged"] == True:
            return None

        if len(coconut_coupon_orders) == 0:
            return None

        net_coconut_coupon_quantity = sum(order.quantity for order in coconut_coupon_orders)
        target_coconut_quantity = -int(delta * net_coconut_coupon_quantity)

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(target_coconut_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT["VOLCANIC_ROCK"] - (rock_position + rock_buy_orders))
            if quantity > 0:
                orders.append(Order("VOLCANIC_ROCK", best_ask, quantity))
        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(target_coconut_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT["VOLCANIC_ROCK"] + (rock_position - rock_sell_orders))
            if quantity > 0:
                orders.append(Order("VOLCANIC_ROCK", best_bid, -quantity))

        return orders

    def coupon_orders(
            self, 
            coupon,
            coupon_order_depth: OrderDepth,
            coupon_position: int,
            traderData: Dict[str, Any],
            volatility: float
        ) -> List[Order]:
            ## TODO:
            mean=0.15959
            thresh=0.00163
            if volatility > mean + thresh:
                if coupon_position != -self.LIMIT[coupon]:
                   target_coupon_position = -self.LIMIT[coupon]
                   if len(coupon_order_depth.buy_orders) > 0:
                       best_bid = max(coupon_order_depth.buy_orders.keys())
                       quantity = min(abs(target_coupon_position - coupon_position), abs(coupon_order_depth.buy_orders[best_bid]))
                       traderData["hedged"] = False
                       return [Order(coupon, best_bid, -quantity)]

            elif volatility < mean + thresh:
                if coupon_position != self.LIMIT[coupon]:
                    target_coupon_position = self.LIMIT[coupon]
                    if len(coupon_order_depth.sell_orders) > 0:
                        best_ask = min(coupon_order_depth.sell_orders.keys())
                        quantity = min(abs(target_coupon_position - coupon_position), abs(coupon_order_depth.sell_orders[best_ask]))
                        traderData["hedged"] = False
                        return [Order(coupon, best_ask, quantity)]

            else:
                return None

    def run(self, state: TradingState):
        result = {}
        conversions = 1
        
        for product in state.order_depths:
            if(product not in ['VOLCANIC_ROCK',
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500']):
                continue
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
                pass  # fallback or historical average
            fair_price = self.fair_price[product]
            Trader.update_pnl(self, product, state, fair_price)
            Trader.update_ema(self, product, fair_price)
            if len(self.history[product])>=11:
                Trader.update_spread(self, product) 
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
                orders.append(Order(product, buy_price, buy_volume))

            if sell_volume > 0:
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
                orders.append(Order(product, buy_price, buy_volume))
            if sell_volume > 0:
                orders.append(Order(product, sell_price, -sell_volume))  
            result[product] = orders
            for product in self.bcontents[basket]:
                orders: List[Order] = []
                logic_function = self.product_logic_factory(product)
                buy_price, sell_price, buy_volume, sell_volume = logic_function(state, self.fair_price[product], product, self.imbalance[product])
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))  
                result[product] = orders

        for coupon in ('VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'):

            if coupon in state.order_depths:
                coupon_position = (
                    state.position[coupon]
                    if coupon in state.position
                    else 0
                )

                rock_position = (
                    state.position["VOLCANIC_ROCK"]
                    if "VOLCANIC_ROCK" in state.position
                    else 0
                )

                rock_order_depth = state.order_depths["VOLCANIC_ROCK"]
                coupon_order_depth = state.order_depths[coupon]
                rock_mid_price = (min(rock_order_depth.buy_orders.keys()) + max(rock_order_depth.sell_orders.keys())) / 2
                coupon_mid_price = self.get_coconut_coupon_mid_price(
                coupon_order_depth,
                self.coupon_objects[coupon]
                )
                #####
                tte = (5/7) - (state.timestamp) / 1000000 / 7
                ######
                volatility = self.implied_volatility(coupon_mid_price, rock_mid_price, self.strikes[coupon], tte)
                delta = self.delta(rock_mid_price, self.strikes[coupon], tte, volatility)

                target_rock_quantity = -int(delta * coupon_position)
                if abs(rock_position) == self.max_positions["VOLCANIC_ROCK"] or target_rock_quantity == rock_position:
                    self.coupon_objects[coupon]["hedged"] = True

                coupon_orders = self.coupon_orders(
                    coupon,
                    state.order_depths[coupon],
                    coupon_position,
                    self.coupon_objects[coupon],
                    volatility
                )
                if coupon_orders != None:
                    result[coupon] = coupon_orders

                    rock_orders = self.delta_hedge_coupon_orders(
                        state.order_depths["VOLCANIC_ROCK"],
                        coupon_orders,
                        rock_position,
                        0,
                        0,
                        delta,
                        self.coupon_objects[coupon]
                    )
                    if rock_orders != None:
                        result["VOLCANIC_ROCK"] = rock_orders
                    
                else:
                    rock_orders = self.delta_hedge_rock_position(
                        state.order_depths["VOLCANIC_ROCK"],
                        coupon_position,
                        rock_position,
                        0,
                        0,
                        delta,
                        self.coupon_objects[coupon]
                    )
                    if rock_orders != None:
                        result["VOLCANIC_ROCK"] = rock_orders


        logger.print(self.learnt_spreads)
        logger.flush(state, result, conversions, self.traderData)
        return result, conversions, self.traderData

