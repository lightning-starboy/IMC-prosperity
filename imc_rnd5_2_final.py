# --- IMPORTS ---
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import collections
import copy
from datamodel import Order, OrderDepth, TradingState, Trade, Listing, Observation, ProsperityEncoder, ConversionObservation, UserId, Symbol, Product

# --- Numpy Fallback ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Voucher Price vs m_t fitting strategy disabled.")
    class dummy_np:
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0.0
        @staticmethod
        def std(x):
            n = len(x); mean = sum(x) / n if n > 0 else 0.0
            if n < 2: return 0.0
            var = sum([(val - mean)**2 for val in x]) / (n - 1) if n > 1 else 0
            return math.sqrt(var) if var > 0 else 0.0
        @staticmethod
        def polyfit(x, y, deg): print("Warning: numpy.polyfit unavailable."); return None
        @staticmethod
        def polyval(p, x):
             print("Warning: numpy.polyval unavailable.")
             if p is not None and len(p) == 3:
                  try: return [p[2]] * len(x)
                  except TypeError: return p[2]
             return x
    np = dummy_np()

# --- py_vollib Section (Remains disabled) ---
PYVOLLIB_AVAILABLE = False

# --- CONSTANTS --- (Adjusted based on LATEST analysis)
class Constants:
    POSITION_LIMIT = {'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50, 'CROISSANTS': 250, 'JAMS': 350, 'DJEMBES': 60, 'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100, 'VOLCANIC_ROCK': 400, 'VOLCANIC_ROCK_VOUCHER_9500': 200, 'VOLCANIC_ROCK_VOUCHER_9750': 200, 'VOLCANIC_ROCK_VOUCHER_10000': 200, 'VOLCANIC_ROCK_VOUCHER_10250': 200, 'VOLCANIC_ROCK_VOUCHER_10500': 200, 'MAGNIFICENT_MACARONS': 75}
    EPSILON = 1e-6

    # Basket Params (Increase threshold, revert B2 aggression)
    BASKET1_COMPONENTS = {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1}; BASKET2_COMPONENTS = {'CROISSANTS': 4, 'JAMS': 2}
    BASKET_BASE_VALUE_THRESHOLD = 7.0 # Increased from 6.0
    BASKET_VOL_WINDOW = 15; BASKET_VOL_THRESHOLD_SCALAR = 0.5
    BASKET1_PASSIVE_QTY = 4; BASKET1_NEUTRALITY_SCALE = 0.7
    BASKET2_PASSIVE_QTY = 4 # Reverted from 6
    BASKET2_NEUTRALITY_SCALE = 0.7 # Reverted from 0.6

    # Component MM Params (Tighter spread)
    COMPONENT_MM_QTY = 18; COMPONENT_SPREAD = 1.0; COMPONENT_NEUTRALITY_SCALE = 0.6 # Reduced spread from 1.5

    # Resin Params (Increase buffer)
    RESIN_EMA_ALPHA = 0.05; RESIN_STD_WINDOW = 20; RESIN_PASSIVE_QTY = 6; RESIN_NEUTRALITY_SCALE = 0.7
    RESIN_SPREAD_BUFFER = 5.0 # Increased from 4.0

    # Squid Ink Params (Disable entry logic, keep tight stop/neutrality)
    SQUID_WINDOW = 12; SQUID_MIN_STD = 1.5
    SQUID_SR_STD_FACTOR = 2.0 # Entry logic disabled below, factor less relevant now
    SQUID_MIN_SR_SPREAD = 6
    SQUID_BASE_AGGRESSION_QTY = 12 # Entry logic disabled below
    SQUID_AGGRESSION_POS_SCALE = 0.8 # Entry logic disabled below
    SQUID_EXIT_STD_FACTOR = 0.5
    SQUID_STOP_LOSS_STD_FACTOR = 0.8 # Keep tight stop loss
    SQUID_NEUTRAL_QTY = 4; SQUID_NEUTRAL_POS_THRESH = 6 # Keep neutrality

    # Kelp Params (Keep as before)
    KELP_WINDOW = 15; KELP_MIN_STD = 0.7; KELP_AGGRESSION_STD_FACTOR = 2.0; KELP_BASE_AGGRESSION_QTY = 22; KELP_AGGRESSION_POS_SCALE = 0.7; KELP_PASSIVE_QTY = 12; KELP_IMBALANCE_THRESH = 0.25; KELP_POS_ADJUST_THRESH = 0.5; KELP_POS_ADJUST_FACTOR = 1.6; KELP_MIN_PASSIVE_SPREAD = 2

    # Voucher Params (Lower threshold significantly)
    VOUCHER_PRODUCTS = [p for p in POSITION_LIMIT if 'VOUCHER' in p]
    VOUCHER_INFO = {'VOLCANIC_ROCK_VOUCHER_9500': {'strike': 9500},'VOLCANIC_ROCK_VOUCHER_9750': {'strike': 9750},'VOLCANIC_ROCK_VOUCHER_10000': {'strike': 10000},'VOLCANIC_ROCK_VOUCHER_10250': {'strike': 10250},'VOLCANIC_ROCK_VOUCHER_10500': {'strike': 10500}}
    UNDERLYING_PRODUCT = 'VOLCANIC_ROCK'
    DAYS_LEFT_MAP_R4 = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}
    TIMESTAMPS_PER_DAY = 1_000_000; TRADING_DAYS_PER_YEAR = 252.0; MIN_TTE = 1e-9
    MIN_POINTS_FOR_VT_FIT = 3
    VOUCHER_PRICE_DIFF_THRESHOLD = 9.0 # Reduced from 15.0
    VOUCHER_TRADE_QTY = 2; VOUCHER_NEUTRALITY_SCALE = 0.5
    DISABLED_VOUCHERS = set()

    # Macaron Params (Reduce buffer)
    MACARON_PRODUCT = 'MAGNIFICENT_MACARONS'; MACARON_CSI_THRESHOLD = 64.0; MACARON_STORAGE_COST = 0.1; MACARON_CONVERSION_LIMIT = 10
    MACARON_CONV_BUFFER = 0.5 # Reduced from 1.0
    MACARON_EST_HOLD_TIME = 10

    # Momentum Params (Unchanged)
    MARKET_TRADE_VOL_THRESHOLD = 30; MARKET_TRADE_IMBALANCE_THRESHOLD = 0.3; MARKET_TRADE_PRICE_ADJUST = 1

    # Counterparty Tracking Params (Unchanged)
    MAX_TRADE_HISTORY = 100; COUNTERPARTY_VOLUME_THRESHOLD = 50

# --- Helper Functions --- (Keep as before)
def calculate_skewed_quantity(base_qty: int, current_pos: int, limit: int, neutrality_scale: float, is_buy_order: bool) -> int:
    if limit <= 0: return base_qty
    scale = max(0.0, min(1.0, neutrality_scale)); position_fraction = 0.0
    if is_buy_order and current_pos > 0: position_fraction = current_pos / limit
    elif not is_buy_order and current_pos < 0: position_fraction = abs(current_pos) / limit
    elif (is_buy_order and current_pos < 0) or (not is_buy_order and current_pos > 0): position_fraction = 0.0
    position_fraction = max(0.0, min(1.0, position_fraction)); quantity_factor = max(0.0, 1.0 - (position_fraction * scale))
    return math.floor(base_qty * quantity_factor)

# --- Trader Class ---
class Trader:

    # Keep DEQUE_CONFIGS as before
    DEQUE_CONFIGS = [
        ('resin_prices', 'RESIN_STD_WINDOW'), ('squid_prices', 'SQUID_WINDOW'),
        ('kelp_prices', 'KELP_WINDOW'), ('basket1_prices', 'BASKET_VOL_WINDOW'),
        ('basket2_prices', 'BASKET_VOL_WINDOW'), ('trade_history', 'MAX_TRADE_HISTORY')
    ]

    def __init__(self):
        self.constants = Constants()
        self._voucher_strategy_enabled = NUMPY_AVAILABLE

    # --- State Management --- (Keep as before)
    def load_trader_data(self, traderData: str) -> dict:
        state = {}
        if traderData:
            try: state = json.loads(traderData)
            except Exception as e: print(f"Error loading traderData: {e}"); state = {}
        if not isinstance(state, dict): state = {}

        for key, maxlen_attr_or_val in self.DEQUE_CONFIGS:
            maxlen = getattr(self.constants, maxlen_attr_or_val, 20) if isinstance(maxlen_attr_or_val, str) else maxlen_attr_or_val
            loaded_list = state.get(key, [])
            if key == 'trade_history':
                 valid_vals = [item for item in loaded_list if isinstance(item, dict)]
                 state[key] = collections.deque(valid_vals, maxlen=maxlen)
            else:
                 if not isinstance(loaded_list, list): loaded_list = []
                 valid_vals = [v for v in loaded_list if isinstance(v, (int, float))]
                 state[key] = collections.deque(valid_vals, maxlen=maxlen)

        if 'resin_ema' not in state or not isinstance(state['resin_ema'], (float, int, type(None))): state['resin_ema'] = None
        elif isinstance(state['resin_ema'], int): state['resin_ema'] = float(state['resin_ema'])

        if 'squid_holding_info' not in state or not isinstance(state['squid_holding_info'], (dict, type(None))): state['squid_holding_info'] = None

        state.setdefault('current_day', 1)
        if not isinstance(state['current_day'], int) or state['current_day'] < 1 or state['current_day'] > 4: state['current_day'] = 1

        if 'last_vt_fit_coeffs' in state:
            coeffs_list = state['last_vt_fit_coeffs']
            if self._voucher_strategy_enabled and isinstance(coeffs_list, list):
                try: state['last_vt_fit_coeffs'] = np.array(coeffs_list)
                except Exception: state['last_vt_fit_coeffs'] = None
            elif not isinstance(coeffs_list, (list, type(None))): state['last_vt_fit_coeffs'] = None
        else: state['last_vt_fit_coeffs'] = None

        if 'counterparty_stats' not in state or not isinstance(state['counterparty_stats'], dict):
            state['counterparty_stats'] = {}

        return state

    def save_trader_data(self, data: dict) -> str:
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, collections.deque):
                serializable_data[key] = list(value)
            elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                 serializable_data[key] = value.tolist()
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                 serializable_data[key] = value
        if 'counterparty_stats' in data and isinstance(data['counterparty_stats'], dict):
            serializable_data['counterparty_stats'] = data['counterparty_stats']
        else: serializable_data['counterparty_stats'] = {}
        try: return json.dumps(serializable_data, separators=(',', ':'))
        except Exception as e:
            print(f"Error saving traderData: {e}"); minimal_data = {'current_day': data.get('current_day', 1)}
            try: return json.dumps(minimal_data)
            except: return "{}"

    # --- Calculation Helpers --- (Keep as before)
    def _calculate_tte(self, day: int, timestamp: int) -> float:
        days_remaining_at_start = self.constants.DAYS_LEFT_MAP_R4.get(day, 0)
        if days_remaining_at_start <= 0: return self.constants.MIN_TTE
        time_elapsed_fraction_day = timestamp / self.constants.TIMESTAMPS_PER_DAY
        remaining_days = (days_remaining_at_start - 1) + (1 - time_elapsed_fraction_day)
        tte_years = remaining_days / self.constants.TRADING_DAYS_PER_YEAR
        return max(tte_years, self.constants.MIN_TTE)
    def _calculate_m_t(self, St: float, K: int, TTE: float) -> Optional[float]:
        if St is None or K is None or TTE is None or TTE <= 0 or St <= 0 or K <= 0 or any(isinstance(v, float) and math.isnan(v) for v in [St, float(K), TTE]): return None
        try: return math.log(float(K) / St) / math.sqrt(TTE)
        except (ValueError, ZeroDivisionError): return None
    def _fit_parabola_vt(self, mt_vt_pairs: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        if not NUMPY_AVAILABLE: return None
        valid_points = [(m, v) for m, v in mt_vt_pairs if m is not None and v is not None and not math.isnan(m) and not math.isnan(v)]
        if len(valid_points) < self.constants.MIN_POINTS_FOR_VT_FIT: return None
        m_values = np.array([p[0] for p in valid_points]); v_values = np.array([p[1] for p in valid_points])
        try: coeffs = np.polyfit(m_values, v_values, 2); return coeffs
        except (np.linalg.LinAlgError, ValueError): return None
    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None; best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask
    def calculate_weighted_mid_price(self, order_depth: OrderDepth, best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
         if best_bid is None or best_ask is None: return None
         bid_vol = order_depth.buy_orders.get(best_bid, 0.0); ask_vol = abs(order_depth.sell_orders.get(best_ask, 0.0)); total_vol = bid_vol + ask_vol
         if total_vol < self.constants.EPSILON: return (best_ask + best_bid) / 2.0
         return (best_bid * ask_vol + best_ask * bid_vol) / total_vol
    def get_volume_imbalance(self, order_depth: OrderDepth, levels: int = 3) -> float:
        bid_vol, ask_vol = 0.0, 0.0;
        try: sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True); sorted_asks = sorted(order_depth.sell_orders.keys())
        except Exception: return 0.0
        for i in range(min(levels, len(sorted_bids))): bid_vol += order_depth.buy_orders[sorted_bids[i]]
        for i in range(min(levels, len(sorted_asks))): ask_vol += abs(order_depth.sell_orders[sorted_asks[i]])
        total_vol = bid_vol + ask_vol
        if total_vol < self.constants.EPSILON: return 0.0
        return (bid_vol - ask_vol) / total_vol
    def manage_orders(self, product: str, current_pos: int, limit: int, desired_orders: List[Order]) -> List[Order]:
        final_orders: List[Order] = []; potential_buy_qty = sum(o.quantity for o in desired_orders if o.quantity > 0); potential_sell_qty = sum(abs(o.quantity) for o in desired_orders if o.quantity < 0)
        allowed_to_buy = limit - current_pos; allowed_to_sell = limit + current_pos; buy_scaling_factor = 1.0; sell_scaling_factor = 1.0
        if potential_buy_qty > allowed_to_buy: buy_scaling_factor = allowed_to_buy / potential_buy_qty if potential_buy_qty > 0 else 0
        if potential_sell_qty > allowed_to_sell: sell_scaling_factor = allowed_to_sell / potential_sell_qty if potential_sell_qty > 0 else 0
        for order in desired_orders:
            original_qty = order.quantity; capped_qty = 0
            if original_qty > 0: capped_qty = math.floor(original_qty * buy_scaling_factor)
            elif original_qty < 0: capped_qty = -math.floor(abs(original_qty) * sell_scaling_factor)
            if abs(capped_qty) > 0: final_orders.append(Order(order.symbol, order.price, capped_qty))
        return final_orders
    def calculate_basket_values(self, components: Dict[str, int], product_metrics: Dict[str, Dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        theoretical_fv = 0.0; replication_buy_cost = 0.0; replication_sell_value = 0.0; valid = True
        for comp, qty in components.items():
            comp_metrics = product_metrics.get(comp)
            if not comp_metrics or comp_metrics.get('mid_price') is None or comp_metrics.get('best_ask') is None or comp_metrics.get('best_bid') is None: valid = False; break
            theoretical_fv += qty * comp_metrics['mid_price']; replication_buy_cost += qty * comp_metrics['best_ask']; replication_sell_value += qty * comp_metrics['best_bid']
        return (theoretical_fv, replication_buy_cost, replication_sell_value) if valid else (None, None, None)

    # --- Main Trading Logic ---
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        trader_state = self.load_trader_data(state.traderData)
        all_orders: Dict[str, List[Order]] = {prod: [] for prod in self.constants.POSITION_LIMIT.keys()}
        conversions = 0
        current_positions = state.position
        timestamp = state.timestamp

        # --- Manage Day Assumption & Reset State --- (Keep as before)
        if timestamp == 0:
            assumed_next_day = trader_state.get('current_day', 1) + 1
            if assumed_next_day <= 4:
                trader_state['current_day'] = assumed_next_day
                print(f"Timestamp 0 detected: Advancing to Day {trader_state['current_day']}")
                trader_state['resin_ema'] = None; trader_state['squid_holding_info'] = None; trader_state['last_vt_fit_coeffs'] = None
                for key, _ in self.DEQUE_CONFIGS:
                    if key != 'trade_history' and key in trader_state and isinstance(trader_state[key], collections.deque):
                           trader_state[key].clear()
            else: print(f"Warning: Timestamp 0 detected, but already at day {trader_state.get('current_day', 0)}. Not advancing day.")
        current_day = trader_state.get('current_day', 1)

        # --- Process Own Trades & Update Counterparty Stats --- (Keep as before)
        trade_history = trader_state.get('trade_history', collections.deque(maxlen=self.constants.MAX_TRADE_HISTORY))
        counterparty_stats = trader_state.get('counterparty_stats', {})
        for product, trades_list in state.own_trades.items():
             for trade in trades_list:
                 if not isinstance(trade, Trade): continue
                 actual_counter_party: Optional[UserId] = None
                 if trade.buyer == "SUBMISSION": actual_counter_party = trade.seller
                 elif trade.seller == "SUBMISSION": actual_counter_party = trade.buyer
                 if actual_counter_party and actual_counter_party != "SUBMISSION":
                      trade_info = {'symbol': trade.symbol, 'price': trade.price, 'quantity': trade.quantity,'counter_party': actual_counter_party, 'timestamp': trade.timestamp}
                      trade_history.append(trade_info)
                      cp_id = actual_counter_party
                      if cp_id not in counterparty_stats: counterparty_stats[cp_id] = {'total_volume': 0, 'net_qty': 0}
                      counterparty_stats[cp_id]['total_volume'] += abs(trade.quantity)
                      counterparty_stats[cp_id]['net_qty'] += trade.quantity
        trader_state['trade_history'] = trade_history
        trader_state['counterparty_stats'] = counterparty_stats

        # --- 1. Calculate Metrics --- (Keep as before)
        product_metrics = {}
        basket_std_dev = {'PICNIC_BASKET1': None, 'PICNIC_BASKET2': None}
        all_available_products = list(state.order_depths.keys())
        macaron_conv_obs = state.observations.conversionObservations.get(self.constants.MACARON_PRODUCT)
        macaron_metrics = {'mid_price': None, 'best_bid': None, 'best_ask': None,'eff_conv_buy': None, 'eff_conv_sell': None,'sunlight_index': None, 'sugar_price': None,'total_market_vol': 0, 'net_market_buy_vol': 0}
        if macaron_conv_obs:
            macaron_metrics['sunlight_index'] = getattr(macaron_conv_obs, 'sunlightIndex', None); macaron_metrics['sugar_price'] = getattr(macaron_conv_obs, 'sugarPrice', None)
            required_attrs = ['askPrice', 'transportFees', 'importTariff', 'bidPrice', 'exportTariff']
            if all(hasattr(macaron_conv_obs, attr) and getattr(macaron_conv_obs, attr) is not None for attr in required_attrs):
                 try:
                     macaron_metrics['eff_conv_buy'] = (macaron_conv_obs.askPrice + macaron_conv_obs.transportFees + macaron_conv_obs.importTariff);
                     macaron_metrics['eff_conv_sell'] = (macaron_conv_obs.bidPrice - macaron_conv_obs.transportFees - macaron_conv_obs.exportTariff)
                 except Exception as e: print(f"Error calculating Macaron conversion prices: {e}")
        for product in all_available_products:
            metrics = macaron_metrics if product == self.constants.MACARON_PRODUCT else {'mid_price': None, 'best_bid': None, 'best_ask': None,'std_dev': None, 'average': None,'total_market_vol': 0, 'net_market_buy_vol': 0}
            order_depth = state.order_depths.get(product)
            if order_depth:
                best_bid, best_ask = self.get_best_bid_ask(order_depth); metrics['best_bid'] = best_bid; metrics['best_ask'] = best_ask; mid_price = None
                if best_bid is not None and best_ask is not None: mid_price = self.calculate_weighted_mid_price(order_depth, best_bid, best_ask); mid_price = (best_bid + best_ask) / 2.0 if mid_price is None else mid_price
                elif best_bid is not None: mid_price = best_bid
                elif best_ask is not None: mid_price = best_ask
                metrics['mid_price'] = mid_price
                if mid_price is not None:
                    stats_configs = {'RAINFOREST_RESIN': ('resin_prices', 'RESIN_STD_WINDOW', True), 'SQUID_INK': ('squid_prices', 'SQUID_WINDOW', True),'KELP': ('kelp_prices', 'KELP_WINDOW', True), 'PICNIC_BASKET1': ('basket1_prices', 'BASKET_VOL_WINDOW', False),'PICNIC_BASKET2': ('basket2_prices', 'BASKET_VOL_WINDOW', False)}
                    if product in stats_configs:
                        deque_key, window_attr, stats_needed = stats_configs[product]
                        if deque_key in trader_state and isinstance(trader_state[deque_key], collections.deque):
                             trader_state[deque_key].append(mid_price); prices = list(trader_state[deque_key])
                             if len(prices) >= 5:
                                  try:
                                       p_std = np.std(prices); p_mean = np.mean(prices)
                                       if stats_needed: metrics['average'] = p_mean; metrics['std_dev'] = p_std
                                       if product in basket_std_dev: basket_std_dev[product] = p_std
                                  except Exception: metrics['average'] = None; metrics['std_dev'] = None; basket_std_dev[product] = None
            product_metrics[product] = metrics
        for product, trades in state.market_trades.items():
            if product in product_metrics and product_metrics[product]['mid_price'] is not None:
                product_mid = product_metrics[product]['mid_price']; total_vol = 0; net_buy_vol = 0
                for trade in trades:
                    if not isinstance(trade, Trade): continue
                    vol = abs(trade.quantity); total_vol += vol;
                    if isinstance(trade.price, (int, float)):
                         if trade.price > product_mid: net_buy_vol += vol
                         elif trade.price < product_mid: net_buy_vol -= vol
                product_metrics[product]['total_market_vol'] = total_vol; product_metrics[product]['net_market_buy_vol'] = net_buy_vol

        # --- 2. Calculate Basket Values --- (Use updated threshold)
        theo_fv_b1, rep_buy_b1, rep_sell_b1 = self.calculate_basket_values(self.constants.BASKET1_COMPONENTS, product_metrics)
        theo_fv_b2, rep_buy_b2, rep_sell_b2 = self.calculate_basket_values(self.constants.BASKET2_COMPONENTS, product_metrics)

        # --- 3. Volcanic Voucher Analysis --- (Keep as before, uses updated threshold constant)
        voucher_analysis = {'tte': None, 'st': None, 'fit_coeffs': None, 'data': {}}
        if self._voucher_strategy_enabled:
            voucher_analysis['tte'] = self._calculate_tte(current_day, timestamp)
            st_metrics = product_metrics.get(self.constants.UNDERLYING_PRODUCT); st_price = st_metrics.get('mid_price') if st_metrics else None; voucher_analysis['st'] = st_price
            if voucher_analysis['tte'] is not None and st_price is not None and st_price > 0:
                mt_vt_pairs = [];
                for voucher_prod in self.constants.VOUCHER_PRODUCTS:
                     if voucher_prod in self.constants.DISABLED_VOUCHERS: continue
                     if voucher_prod not in state.order_depths: continue
                     voucher_metrics = product_metrics.get(voucher_prod); vt_price = voucher_metrics.get('mid_price'); strike = self.constants.VOUCHER_INFO[voucher_prod]['strike']
                     if vt_price is not None:
                         m_t = self._calculate_m_t(st_price, strike, voucher_analysis['tte']); voucher_analysis['data'][voucher_prod] = {'Vt': vt_price, 'K': strike, 'm_t': m_t, 'best_bid': voucher_metrics.get('best_bid'), 'best_ask': voucher_metrics.get('best_ask')}
                         if m_t is not None: mt_vt_pairs.append((m_t, vt_price))
                     else: voucher_analysis['data'][voucher_prod] = {'Vt': None, 'K': strike, 'm_t': None, 'best_bid': None, 'best_ask': None}
                fit_coeffs = self._fit_parabola_vt(mt_vt_pairs)
                if fit_coeffs is not None and len(fit_coeffs) == 3 and fit_coeffs[0] > 0: voucher_analysis['fit_coeffs'] = fit_coeffs; trader_state['last_vt_fit_coeffs'] = fit_coeffs
                else:
                     last_coeffs = trader_state.get('last_vt_fit_coeffs', None)
                     if last_coeffs is not None and isinstance(last_coeffs, np.ndarray) and len(last_coeffs) == 3 and last_coeffs[0] > 0: voucher_analysis['fit_coeffs'] = last_coeffs
                     else: voucher_analysis['fit_coeffs'] = None
            else: voucher_analysis['fit_coeffs'] = None

        # --- 4. Generate Orders ---
        for product in all_available_products:
            orders: List[Order] = []
            order_depth = state.order_depths.get(product); current_pos = current_positions.get(product, 0)
            limit = self.constants.POSITION_LIMIT.get(product, 0); metrics = product_metrics.get(product, {})
            best_bid, best_ask = metrics.get('best_bid'), metrics.get('best_ask'); mid_price = metrics.get('mid_price')

            if mid_price is None and product != self.constants.MACARON_PRODUCT: continue
            if (best_bid is None or best_ask is None) and product not in [self.constants.MACARON_PRODUCT]: continue

            # Momentum adjustment (Keep as before)
            price_adjustment = 0; total_mkt_vol = metrics.get('total_market_vol', 0); net_mkt_buy_vol = metrics.get('net_market_buy_vol', 0)
            if total_mkt_vol > self.constants.MARKET_TRADE_VOL_THRESHOLD:
                 imbalance_ratio = net_mkt_buy_vol / total_mkt_vol if total_mkt_vol > 0 else 0
                 if imbalance_ratio > self.constants.MARKET_TRADE_IMBALANCE_THRESHOLD: price_adjustment = self.constants.MARKET_TRADE_PRICE_ADJUST
                 elif imbalance_ratio < -self.constants.MARKET_TRADE_IMBALANCE_THRESHOLD: price_adjustment = -self.constants.MARKET_TRADE_PRICE_ADJUST

            # --- STRATEGIES (with specific adjustments) ---

            # Basket Strategy (Use updated threshold, reverted B2 params)
            if product == 'PICNIC_BASKET1' or product == 'PICNIC_BASKET2':
                 theo_fv, rep_buy, rep_sell = (theo_fv_b1, rep_buy_b1, rep_sell_b1) if product == 'PICNIC_BASKET1' else (theo_fv_b2, rep_buy_b2, rep_sell_b2)
                 std_dev = basket_std_dev.get(product);
                 current_threshold = self.constants.BASKET_BASE_VALUE_THRESHOLD # Use updated constant
                 base_qty = self.constants.BASKET1_PASSIVE_QTY if product == 'PICNIC_BASKET1' else self.constants.BASKET2_PASSIVE_QTY # Use updated constant for B2
                 neutrality_scale = self.constants.BASKET1_NEUTRALITY_SCALE if product == 'PICNIC_BASKET1' else self.constants.BASKET2_NEUTRALITY_SCALE # Use updated constant for B2

                 if std_dev is not None and std_dev > 0: current_threshold += std_dev * self.constants.BASKET_VOL_THRESHOLD_SCALAR
                 if rep_buy is not None and rep_sell is not None and best_ask is not None and best_bid is not None:
                    buy_value_signal = rep_sell - best_ask; sell_value_signal = best_bid - rep_buy;
                    if buy_value_signal > current_threshold:
                        target_price = best_ask + 1 + (price_adjustment if price_adjustment > 0 else 0); target_price = min(target_price, int(rep_sell - current_threshold) if rep_sell is not None else target_price)
                        skewed_qty = calculate_skewed_quantity(base_qty, current_pos, limit, neutrality_scale, True)
                        if skewed_qty > 0: orders.append(Order(product, target_price, skewed_qty))
                    if sell_value_signal > current_threshold:
                        target_price = best_bid - 1 + (price_adjustment if price_adjustment < 0 else 0); target_price = max(target_price, int(rep_buy + current_threshold) if rep_buy is not None else target_price)
                        skewed_qty = calculate_skewed_quantity(base_qty, current_pos, limit, neutrality_scale, False)
                        if skewed_qty > 0: orders.append(Order(product, target_price, -skewed_qty))

            # Component Market Making (Use tighter spread, quote inside BBO)
            elif product in ['CROISSANTS', 'JAMS', 'DJEMBES']:
                 if mid_price is not None and best_bid is not None and best_ask is not None:
                    # Try to quote inside the BBO
                    target_bid = best_bid + 1
                    target_ask = best_ask - 1

                    # Apply momentum adjustment cautiously
                    target_bid += (price_adjustment if price_adjustment > 0 else 0)
                    target_ask += (price_adjustment if price_adjustment < 0 else 0)

                    # Ensure minimum spread and don't cross
                    target_bid = min(target_bid, best_ask - 1) # Don't bid >= ask
                    target_ask = max(target_ask, best_bid + 1) # Don't ask <= bid

                    if target_ask > target_bid : # Check if valid spread remains
                        base_qty = self.constants.COMPONENT_MM_QTY
                        buy_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.COMPONENT_NEUTRALITY_SCALE, True)
                        sell_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.COMPONENT_NEUTRALITY_SCALE, False)
                        if buy_qty > 0: orders.append(Order(product, target_bid, buy_qty))
                        if sell_qty > 0: orders.append(Order(product, target_ask, -sell_qty))
                    # else: print(f"Component MM {product}: No valid spread found after adjustments ({best_bid+1} to {best_ask-1}) -> ({target_bid} to {target_ask})")

            # Rainforest Resin Strategy (Use increased buffer)
            elif product == 'RAINFOREST_RESIN':
                 ema = trader_state.get('resin_ema')
                 if not isinstance(ema, (float, int, type(None))): ema = None
                 current_avg = metrics.get('average'); target_buy_price, target_sell_price = None, None
                 if mid_price is not None:
                    if ema is None and current_avg is not None: ema = current_avg
                    elif ema is not None and mid_price is not None:
                         try: ema = (self.constants.RESIN_EMA_ALPHA * mid_price + (1 - self.constants.RESIN_EMA_ALPHA) * ema)
                         except TypeError: ema=None
                    trader_state['resin_ema'] = ema
                 fair_value_est = ema if ema is not None else mid_price
                 if fair_value_est is not None:
                     adjusted_fv = fair_value_est + price_adjustment
                     # Use updated RESIN_SPREAD_BUFFER
                     target_buy_price = math.floor(adjusted_fv - self.constants.RESIN_SPREAD_BUFFER)
                     target_sell_price = math.ceil(adjusted_fv + self.constants.RESIN_SPREAD_BUFFER)
                     if target_buy_price is not None and target_sell_price is not None and target_sell_price > target_buy_price and best_bid is not None and best_ask is not None:
                         base_qty = self.constants.RESIN_PASSIVE_QTY
                         buy_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.RESIN_NEUTRALITY_SCALE, True)
                         if buy_qty > 0 and target_buy_price < best_ask: orders.append(Order(product, int(target_buy_price), int(buy_qty)))
                         sell_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.RESIN_NEUTRALITY_SCALE, False)
                         if sell_qty > 0 and target_sell_price > best_bid: orders.append(Order(product, int(target_sell_price), -int(sell_qty)))

            # Squid Ink Strategy (Entry logic disabled)
            elif product == 'SQUID_INK':
                 holding_info = trader_state.get('squid_holding_info'); current_avg = metrics.get('average'); current_std = metrics.get('std_dev')
                 dynamic_support, dynamic_resistance = None, None
                 if mid_price is not None and current_avg is not None and current_std is not None and best_bid is not None and best_ask is not None:
                     current_std = max(self.constants.SQUID_MIN_STD, current_std)
                     # Calculate S/R for potential context, but entry is disabled
                     support = round(current_avg - self.constants.SQUID_SR_STD_FACTOR * current_std); resistance = round(current_avg + self.constants.SQUID_SR_STD_FACTOR * current_std)
                     if resistance > support + self.constants.SQUID_MIN_SR_SPREAD: dynamic_support, dynamic_resistance = support, resistance

                     # Exit Logic (Remains active to manage any existing/stray positions)
                     if holding_info:
                         entry_price = holding_info.get('entry_price'); side = holding_info.get('side'); trade_qty = holding_info.get('qty'); exit_order_placed = False
                         if entry_price is not None and side is not None and trade_qty is not None and isinstance(trade_qty, int) and trade_qty > 0:
                             stop_loss_buy_trigger = entry_price - self.constants.SQUID_STOP_LOSS_STD_FACTOR * current_std; stop_loss_sell_trigger = entry_price + self.constants.SQUID_STOP_LOSS_STD_FACTOR * current_std
                             profit_take_buy_trigger = entry_price + self.constants.SQUID_EXIT_STD_FACTOR * current_std; profit_take_sell_trigger = entry_price - self.constants.SQUID_EXIT_STD_FACTOR * current_std
                             if side == 'buy' and mid_price < stop_loss_buy_trigger:
                                 exit_price = best_bid
                                 if exit_price is not None: orders.append(Order(product, int(exit_price), -trade_qty)); trader_state['squid_holding_info'] = None; holding_info = None; exit_order_placed = True
                             elif side == 'sell' and mid_price > stop_loss_sell_trigger:
                                 exit_price = best_ask
                                 if exit_price is not None: orders.append(Order(product, int(exit_price), trade_qty)); trader_state['squid_holding_info'] = None; holding_info = None; exit_order_placed = True
                             if not exit_order_placed:
                                 if side == 'buy' and mid_price > profit_take_buy_trigger:
                                     exit_price = best_bid
                                     if exit_price is not None: orders.append(Order(product, int(exit_price), -trade_qty)); trader_state['squid_holding_info'] = None; holding_info = None; exit_order_placed = True
                                 elif side == 'sell' and mid_price < profit_take_sell_trigger:
                                     exit_price = best_ask
                                     if exit_price is not None: orders.append(Order(product, int(exit_price), trade_qty)); trader_state['squid_holding_info'] = None; holding_info = None; exit_order_placed = True
                         else: trader_state['squid_holding_info'] = None; holding_info = None

                     # --- Entry Logic --- (DISABLED) ---
                     # if holding_info is None and dynamic_support is not None and dynamic_resistance is not None:
                     #     base_entry_qty = self.constants.SQUID_BASE_AGGRESSION_QTY; rem_buy_capacity = max(0, limit - current_pos); rem_sell_capacity = max(0, limit + current_pos)
                     #     potential_buy_qty = math.floor(base_entry_qty * self.constants.SQUID_AGGRESSION_POS_SCALE); scaled_buy_qty = min(potential_buy_qty, rem_buy_capacity)
                     #     potential_sell_qty = math.floor(base_entry_qty * self.constants.SQUID_AGGRESSION_POS_SCALE); scaled_sell_qty = min(potential_sell_qty, rem_sell_capacity)
                     #     if best_ask <= dynamic_support and scaled_buy_qty > 0:
                     #         entry_price = best_ask; orders.append(Order(product, int(entry_price), int(scaled_buy_qty))); trader_state['squid_holding_info'] = {'entry_price': entry_price, 'qty': scaled_buy_qty, 'side': 'buy'}; holding_info = trader_state['squid_holding_info']
                     #     elif best_bid >= dynamic_resistance and scaled_sell_qty > 0:
                     #          entry_price = best_bid; orders.append(Order(product, int(entry_price), -int(scaled_sell_qty))); trader_state['squid_holding_info'] = {'entry_price': entry_price, 'qty': scaled_sell_qty, 'side': 'sell'}; holding_info = trader_state['squid_holding_info']

                 # --- Neutrality Logic (Remains active) ---
                 if holding_info is None and abs(current_pos) > self.constants.SQUID_NEUTRAL_POS_THRESH and best_bid is not None and best_ask is not None:
                       neutral_qty = self.constants.SQUID_NEUTRAL_QTY
                       if current_pos > 0: orders.append(Order(product, best_bid, -int(min(neutral_qty, current_pos))))
                       elif current_pos < 0: orders.append(Order(product, best_ask, int(min(neutral_qty, abs(current_pos)))))

            # Kelp Strategy (Keep as before)
            elif product == 'KELP':
                 agg_order_placed = False; current_avg = metrics.get('average'); current_std = metrics.get('std_dev')
                 if mid_price is not None and current_avg is not None and current_std is not None and best_bid is not None and best_ask is not None: # Aggressive
                     current_std = max(self.constants.KELP_MIN_STD, current_std); deviation = mid_price - current_avg; deviation_threshold = self.constants.KELP_AGGRESSION_STD_FACTOR * current_std
                     base_agg_qty = self.constants.KELP_BASE_AGGRESSION_QTY; rem_buy_capacity = max(0, limit - current_pos); rem_sell_capacity = max(0, limit + current_pos)
                     scaled_buy_qty = math.floor(base_agg_qty * min(1.0, rem_buy_capacity / base_agg_qty if base_agg_qty > 0 else 0) * self.constants.KELP_AGGRESSION_POS_SCALE); scaled_sell_qty = math.floor(base_agg_qty * min(1.0, rem_sell_capacity / base_agg_qty if base_agg_qty > 0 else 0) * self.constants.KELP_AGGRESSION_POS_SCALE)
                     if deviation < -deviation_threshold and scaled_buy_qty > 0: orders.append(Order(product, best_ask, int(scaled_buy_qty))); agg_order_placed = True
                     elif deviation > deviation_threshold and scaled_sell_qty > 0: orders.append(Order(product, best_bid, -int(scaled_sell_qty))); agg_order_placed = True
                 if not agg_order_placed and best_bid is not None and best_ask is not None: # Passive
                     spread = best_ask - best_bid
                     if spread >= self.constants.KELP_MIN_PASSIVE_SPREAD:
                         our_bid = best_bid + 1 + (price_adjustment if price_adjustment > 0 else 0); our_ask = best_ask - 1 + (price_adjustment if price_adjustment < 0 else 0)
                         our_bid = min(our_bid, best_ask - 1); our_ask = max(our_ask, best_bid + 1)
                         if our_ask > our_bid:
                             buy_qty, sell_qty = self.constants.KELP_PASSIVE_QTY, self.constants.KELP_PASSIVE_QTY
                             pos_limit_fraction = abs(current_pos) / limit if limit > 0 else 0
                             if pos_limit_fraction > self.constants.KELP_POS_ADJUST_THRESH:
                                 if current_pos > 0: sell_qty = round(sell_qty * self.constants.KELP_POS_ADJUST_FACTOR)
                                 elif current_pos < 0: buy_qty = round(buy_qty * self.constants.KELP_POS_ADJUST_FACTOR)
                             final_buy_qty = calculate_skewed_quantity(buy_qty, current_pos, limit, 0.5, True); final_sell_qty = calculate_skewed_quantity(sell_qty, current_pos, limit, 0.5, False)
                             if final_buy_qty > 0: orders.append(Order(product, int(our_bid), int(final_buy_qty)))
                             if final_sell_qty > 0: orders.append(Order(product, int(our_ask), -int(final_sell_qty)))

            # Volcanic Voucher Strategy (Use updated threshold)
            elif product in self.constants.VOUCHER_PRODUCTS and self._voucher_strategy_enabled and product not in self.constants.DISABLED_VOUCHERS:
                coeffs = voucher_analysis.get('fit_coeffs'); voucher_data = voucher_analysis.get('data', {}).get(product)
                if coeffs is not None and voucher_data:
                    m_t = voucher_data.get('m_t'); Vt_market = voucher_data.get('Vt'); voucher_best_bid = voucher_data.get('best_bid'); voucher_best_ask = voucher_data.get('best_ask')
                    if m_t is not None and Vt_market is not None and voucher_best_bid is not None and voucher_best_ask is not None:
                        try: Vt_fitted = np.polyval(coeffs, m_t)
                        except Exception: Vt_fitted = None
                        if isinstance(Vt_fitted, (int, float)) and not math.isnan(Vt_fitted) and Vt_fitted > 0:
                            price_diff = Vt_market - Vt_fitted; base_trade_qty = self.constants.VOUCHER_TRADE_QTY
                            threshold = self.constants.VOUCHER_PRICE_DIFF_THRESHOLD # Use updated constant
                            if price_diff > threshold: # Sell Signal
                                target_price = voucher_best_bid; skewed_qty = calculate_skewed_quantity(base_trade_qty, current_pos, limit, self.constants.VOUCHER_NEUTRALITY_SCALE, False)
                                if skewed_qty > 0: orders.append(Order(product, target_price, -skewed_qty))
                            elif price_diff < -threshold: # Buy Signal
                                target_price = voucher_best_ask; skewed_qty = calculate_skewed_quantity(base_trade_qty, current_pos, limit, self.constants.VOUCHER_NEUTRALITY_SCALE, True)
                                if skewed_qty > 0: orders.append(Order(product, target_price, skewed_qty))

            # Magnificent Macarons Strategy (Use updated buffer)
            elif product == self.constants.MACARON_PRODUCT:
                eff_conv_buy = metrics.get('eff_conv_buy')
                # Conversion (Buy) Logic
                if eff_conv_buy is not None and best_ask is not None:
                    # Use updated MACARON_CONV_BUFFER
                    buy_signal = best_ask > (eff_conv_buy + self.constants.MACARON_CONV_BUFFER)
                    if buy_signal and current_pos < limit:
                        max_conv_request = self.constants.MACARON_CONVERSION_LIMIT; needed_to_reach_limit = limit - current_pos
                        conversion_request_qty = max(0, min(max_conv_request, needed_to_reach_limit))
                        if conversion_request_qty > 0: conversions = conversion_request_qty
                # Selling Logic
                if current_pos > 0 and eff_conv_buy is not None and best_bid is not None:
                    estimated_storage = self.constants.MACARON_STORAGE_COST * self.constants.MACARON_EST_HOLD_TIME
                    # Use updated MACARON_CONV_BUFFER
                    target_sell_price = math.ceil(eff_conv_buy + estimated_storage + self.constants.MACARON_CONV_BUFFER)
                    # Prioritize target price, try to improve on best bid
                    final_ask_price = max(best_bid + 1, target_sell_price)
                    # Do NOT apply negative momentum adjustment when selling inventory
                    # final_ask_price += (price_adjustment if price_adjustment < 0 else 0) # Removed this line
                    final_ask_price = max(final_ask_price, best_bid + 1) # Ensure still above best bid
                    sell_qty = current_pos
                    if sell_qty > 0: orders.append(Order(product, final_ask_price, -sell_qty))

            # Store orders
            if orders:
                all_orders[product] = orders

        # --- 5. Apply Position Limits --- (Keep as before)
        final_result: Dict[str, List[Order]] = {}
        for product, limit_val in self.constants.POSITION_LIMIT.items():
             current_pos = current_positions.get(product, 0); desired_orders_for_product = all_orders.get(product, [])
             if desired_orders_for_product:
                 managed_orders = self.manage_orders(product, current_pos, limit_val, desired_orders_for_product)
                 if managed_orders: final_result[product] = managed_orders

        # --- 6. Save State and Return ---
        trader_state['current_day'] = current_day
        traderData = self.save_trader_data(trader_state)
        return final_result, conversions, traderData