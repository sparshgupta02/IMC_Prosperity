from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

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
        self.history = []
        self.max_so_far = float('-inf')
        self.min_so_far = float('inf')
        self.max_position = 50
        self.traderData = "MarketMakingV1"

    def run(self, state: TradingState):
        result = {}
        conversions = 1

        print("\n=== NEW TICK ===")
        print("Positions:", state.position)
        print("Own trades:", state.own_trades)
        print("Market trades:", state.market_trades)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Estimate fair value using best bid/ask
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                fair_price = (best_ask + best_bid) / 2
            else:
                fair_price = state.own_trades[product][-1].price  # fallback or historical average

            if fair_price > self.max_so_far:
                self.max_so_far = fair_price

            if fair_price < self.min_so_far:
                self.min_so_far = fair_price

            # Define your spread and position limits
            spread = 1
            if(product == "RAINFOREST_RESIN"):
                spread = 2

            # Calculate buy/sell prices and volumes
            buy_price = int(fair_price - spread)
            sell_price = int(fair_price + spread)

            self.history.append(fair_price)
            if len(self.history) == 1001:
                self.history.pop(0)
                prev_slope = (self.history[-1]-self.history[0])/1000
                curr_slope = (self.history[-1] - self.history[-100])/100
                if prev_slope != 0:
                    if fair_price < self.max_so_far-100 and abs(curr_slope/prev_slope)<0.1:
                        buy_price = int(fair_price - spread + 2**(1-abs(curr_slope/prev_slope)))
                        buy_volume = 50

                    elif fair_price > self.min_so_far+100 and abs(curr_slope/prev_slope)<0.1:
                        sell_price = int(fair_price + spread - 2**(1-abs(curr_slope/prev_slope)))
                        sell_volume = 50
                    
                buy_price = int(fair_price - spread**(0.5*abs(curr_slope)))
                sell_price = int(fair_price + spread**(0.5*abs(curr_slope)))

            max_position = 50
            position = state.position.get(product, 0)

            buy_volume = max_position - position
            sell_volume = max_position + position

            if buy_volume > 0:
                print(f"Placing BUY order: {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))

            if sell_volume > 0:
                print(f"Placing SELL order: {sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders

        logger.flush(state, result, conversions, self.traderData)
        return result, conversions, self.traderData