# --- IMPORTS ---
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import collections
import copy
from datamodel import Order, OrderDepth, TradingState, Trade, Listing, Observation, ProsperityEncoder

# Try importing necessary libraries, provide fallbacks if unavailable
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Using basic fallbacks for stats and fitting.")
    class dummy_np:
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0.0
        @staticmethod
        def std(x):
            n = len(x); mean = sum(x) / n if n > 0 else 0.0
            if n < 2: return 0.0
            var = sum([(val - mean)**2 for val in x]) / (n - 1) if n > 1 else 0 # Use sample std dev
            return math.sqrt(var) if var > 0 else 0.0
        @staticmethod
        def polyfit(x, y, deg):
            # Basic polyfit is complex to replicate simply, warn user
            print("Warning: numpy.polyfit unavailable. Parabolic fitting for vouchers will fail.")
            return None # Indicate failure
        @staticmethod
        def polyval(p, x):
            print("Warning: numpy.polyval unavailable. Using fitted voucher IV will fail.")
            return x # Return input x to avoid crashing, but calculation is wrong
    np = dummy_np()

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
    PYVOLLIB_AVAILABLE = True
except ImportError:
    PYVOLLIB_AVAILABLE = False
    print("Warning: py_vollib not available. Implied Volatility calculation for vouchers will fail.")
    # Define dummy functions if needed, or just check PYVOLLIB_AVAILABLE later
    def iv(price, S, K, t, r, flag): return 0.15 # Return a dummy value
    def bs(flag, S, K, t, r, sigma): return S * 0.1 # Return a dummy value

# --- CONSTANTS ---
class Constants:
    # --- Position Limits ---
    POSITION_LIMIT = {
        'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
        'CROISSANTS': 250, 'JAMS': 350, 'DJEMBES': 60,
        'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
        # Volcanic Products - Limits from prompt
        'VOLCANIC_ROCK': 400,
        'VOLCANIC_ROCK_VOUCHER_9500': 200,
        'VOLCANIC_ROCK_VOUCHER_9750': 200,
        'VOLCANIC_ROCK_VOUCHER_10000': 200,
        'VOLCANIC_ROCK_VOUCHER_10250': 200,
        'VOLCANIC_ROCK_VOUCHER_10500': 200,
    }
    EPSILON = 1e-6

    # --- Basket Composition --- (Unchanged)
    BASKET1_COMPONENTS = {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1}
    BASKET2_COMPONENTS = {'CROISSANTS': 4, 'JAMS': 2}

    # --- Basket Value Strategy --- (Unchanged)
    BASKET_BASE_VALUE_THRESHOLD = 6.0; BASKET_VOL_WINDOW = 15
    BASKET_VOL_THRESHOLD_SCALAR = 0.5; BASKET_PASSIVE_QTY = 4
    BASKET_NEUTRALITY_SCALE = 0.7

    # --- Component Market Making --- (Unchanged)
    COMPONENT_MM_QTY = 18; COMPONENT_SPREAD = 2
    COMPONENT_NEUTRALITY_SCALE = 0.6

    # --- RAINFOREST_RESIN --- (Unchanged)
    RESIN_EMA_ALPHA = 0.05; RESIN_STD_WINDOW = 20; RESIN_PASSIVE_QTY = 6; RESIN_NEUTRALITY_SCALE = 0.7; RESIN_SPREAD_BUFFER = 4

    # --- SQUID_INK --- (Unchanged)
    SQUID_WINDOW = 12; SQUID_MIN_STD = 1.5; SQUID_SR_STD_FACTOR = 1.5; SQUID_MIN_SR_SPREAD = 6
    SQUID_BASE_AGGRESSION_QTY = 18; SQUID_AGGRESSION_POS_SCALE = 0.8
    SQUID_EXIT_STD_FACTOR = 0.4; SQUID_STOP_LOSS_STD_FACTOR = 1.2; SQUID_NEUTRAL_QTY = 4; SQUID_NEUTRAL_POS_THRESH = 6

    # --- KELP --- (Unchanged)
    KELP_WINDOW = 15; KELP_MIN_STD = 0.7; KELP_AGGRESSION_STD_FACTOR = 2.0
    KELP_BASE_AGGRESSION_QTY = 22; KELP_AGGRESSION_POS_SCALE = 0.7
    KELP_PASSIVE_QTY = 12; KELP_IMBALANCE_THRESH = 0.25; KELP_POS_ADJUST_THRESH = 0.5; KELP_POS_ADJUST_FACTOR = 1.6; KELP_MIN_PASSIVE_SPREAD = 2

    # --- Volcanic Voucher Strategy ---
    VOUCHER_PRODUCTS = [p for p in POSITION_LIMIT if 'VOUCHER' in p]
    VOUCHER_INFO = {
        'VOLCANIC_ROCK_VOUCHER_9500': {'strike': 9500},
        'VOLCANIC_ROCK_VOUCHER_9750': {'strike': 9750},
        'VOLCANIC_ROCK_VOUCHER_10000': {'strike': 10000},
        'VOLCANIC_ROCK_VOUCHER_10250': {'strike': 10250},
        'VOLCANIC_ROCK_VOUCHER_10500': {'strike': 10500},
    }
    UNDERLYING_PRODUCT = 'VOLCANIC_ROCK'
    RISK_FREE_RATE = 0.0 # Assuming 0% risk-free rate
    # TTE Calculation - Assuming Round 3. Start of R3 Day 0 -> 5 days left.
    # Each day has 1M timestamps. 252 trading days/year.
    INITIAL_DAYS_LEFT_R3 = 5.0
    TIMESTAMPS_PER_DAY = 1_000_000
    TRADING_DAYS_PER_YEAR = 252.0
    MIN_TTE = 1e-9 # Minimum TTE to avoid math errors

    # Strategy Parameters
    MIN_POINTS_FOR_FIT = 3 # Need at least 3 vouchers with valid IV for parabolic fit
    VOL_DIFF_THRESHOLD = 0.015 # Required difference between market IV and fitted IV to trade (e.g., 1.5% vol)
    VOUCHER_TRADE_QTY = 3 # Base quantity to trade per signal
    VOUCHER_NEUTRALITY_SCALE = 0.5 # Scale down trades based on position

# --- Helper Functions ---
# calculate_skewed_quantity remains unchanged

# --- Trader Class ---
class Trader:

    def __init__(self):
        self.constants = Constants()
        self.encoder = ProsperityEncoder()
        # Check availability of required libraries
        if not NUMPY_AVAILABLE or not PYVOLLIB_AVAILABLE:
             print("CRITICAL WARNING: Numpy or PyVollib missing. Voucher strategy disabled.")
             self._voucher_strategy_enabled = False
        else:
             self._voucher_strategy_enabled = True

    # --- State Management ---
    def load_trader_data(self, traderData: str) -> dict:
        state = {}
        if traderData:
            try: state = json.loads(traderData)
            except Exception as e: print(f"Error loading traderData: {e}"); state = {}
        # Restore deques & ensure keys exist
        deque_configs = [
            ('resin_prices', 'RESIN_STD_WINDOW'), ('squid_prices', 'SQUID_WINDOW'), ('kelp_prices', 'KELP_WINDOW'),
            ('basket1_prices', 'BASKET_VOL_WINDOW'), ('basket2_prices', 'BASKET_VOL_WINDOW'),
            # Add deques for underlying price if needed for other strategies
            # ('volcanic_rock_prices', 20), # Example if you need rock price history
            # ('voucher_base_ivs', 50) # Example if you want to track base IV history
        ]
        for key, maxlen_attr_or_val in deque_configs:
            maxlen = getattr(self.constants, maxlen_attr_or_val, 20) if isinstance(maxlen_attr_or_val, str) else maxlen_attr_or_val
            current_val = state.get(key, [])
            # Ensure values are appropriate for deque (e.g., numbers)
            if current_val and isinstance(current_val[0], dict): # Handle case where complex dicts were stored (less likely needed now)
                 state[key] = collections.deque(maxlen=maxlen) # Reset if format changed
                 print(f"Warning: Resetting deque '{key}' due to unexpected format.")
            else:
                 state[key] = collections.deque([v for v in current_val if isinstance(v, (int, float))], maxlen=maxlen)

        state.setdefault('resin_ema', None); state.setdefault('squid_holding_info', None)
        state.setdefault('last_fit_coeffs', None) # Store last valid coefficients
        return state

    def save_trader_data(self, data: dict) -> str:
        # Convert deques to lists for JSON serialization
        serializable_data = copy.deepcopy(data)
        for key, value in serializable_data.items():
            if isinstance(value, collections.deque):
                serializable_data[key] = list(value)
            elif isinstance(value, np.ndarray): # Handle numpy arrays if coeffs are stored
                serializable_data[key] = value.tolist()
        try:
            return self.encoder.encode(serializable_data)
        except Exception as e:
            print(f"Error encoding traderData: {e}")
            # Attempt to encode without problematic keys if error persists
            fallback_data = {k:v for k,v in serializable_data.items() if isinstance(v, (list, dict, str, int, float, bool, type(None)))}
            try: return self.encoder.encode(fallback_data)
            except: return "{}" # Final fallback

    # --- Calculation Helpers ---
    def _calculate_tte(self, timestamp: int) -> float:
        """Calculates Time To Expiry (TTE) in years, assuming Round 3 start."""
        time_elapsed_fraction_day = timestamp / self.constants.TIMESTAMPS_PER_DAY
        remaining_days = self.constants.INITIAL_DAYS_LEFT_R3 - time_elapsed_fraction_day
        tte_years = remaining_days / self.constants.TRADING_DAYS_PER_YEAR
        return max(tte_years, self.constants.MIN_TTE) # Ensure positive TTE

    def _calculate_m_t(self, St: float, K: int, TTE: float) -> Optional[float]:
        """Calculates m_t = log(K/St) / sqrt(TTE)."""
        if pd.isna(St) or pd.isna(K) or pd.isna(TTE) or St <= 0 or K <= 0 or TTE <= 0: return None
        try: return math.log(K / St) / math.sqrt(TTE)
        except (ValueError, ZeroDivisionError): return None

    def _calculate_iv_wrapper(self, Vt: float, St: float, K: int, TTE: float) -> Optional[float]:
        """Wrapper for py_vollib's implied volatility calculation."""
        if not PYVOLLIB_AVAILABLE: return None # Cannot calculate
        if pd.isna(Vt) or pd.isna(St) or pd.isna(K) or pd.isna(TTE) or \
           Vt <= 0 or St <= 0 or K <= 0 or TTE <= 0: return None

        # Basic arbitrage check (Call option price >= St - K*exp(-rT) ~= St - K for r=0)
        intrinsic_value = max(0, St - K)
        if Vt < intrinsic_value - self.constants.EPSILON: # Allow tiny numerical diff
             # print(f"Price violation: Vt={Vt:.2f} < Intrinsic={intrinsic_value:.2f} (St={St:.2f}, K={K}, TTE={TTE:.4f})")
             return None # Price below intrinsic value

        try:
            # py_vollib expects price, S, K, t, r, flag ('c' for call)
            vol = iv(Vt, St, K, TTE, self.constants.RISK_FREE_RATE, 'c')
            if vol < 1e-4 or vol > 2.0: return None # Filter out extreme / likely error IV values
            return vol
        except Exception as e:
            # print(f"IV calculation error for Vt={Vt}, St={St}, K={K}, TTE={TTE}: {e}")
            return None

    def _fit_parabola(self, mt_vt_pairs: List[Tuple[float, float]]) -> Optional[Tuple[Any, float]]:
        """Fits v_t = a*m_t^2 + b*m_t + c and returns (coeffs, base_iv=c)."""
        if not NUMPY_AVAILABLE: return None # Cannot fit
        valid_points = [(m, v) for m, v in mt_vt_pairs if m is not None and v is not None]

        if len(valid_points) < self.constants.MIN_POINTS_FOR_FIT:
            return None # Not enough data

        m_values = np.array([p[0] for p in valid_points])
        v_values = np.array([p[1] for p in valid_points])

        try:
            coeffs = np.polyfit(m_values, v_values, 2) # degree 2 for parabola
            base_iv = coeffs[2] # This is v_t(m_t=0)
            # Basic sanity check on fitted base IV
            if base_iv <= 0 or base_iv > 2.0: # Similar check as raw IV
                 return None
            return coeffs, base_iv
        except (np.linalg.LinAlgError, ValueError):
            # print(f"Warning: Parabolic fit failed for {len(valid_points)} points.")
            return None


    # --- get_best_bid_ask, calculate_weighted_mid_price, get_volume_imbalance, manage_orders, calculate_basket_values ---
    # (These helper functions remain unchanged)
    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask
    def calculate_weighted_mid_price(self, order_depth: OrderDepth, best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
         if best_bid is None or best_ask is None: return None
         bid_vol = order_depth.buy_orders.get(best_bid, 0.0); ask_vol = abs(order_depth.sell_orders.get(best_ask, 0.0))
         total_vol = bid_vol + ask_vol
         if total_vol < self.constants.EPSILON: return (best_ask + best_bid) / 2.0
         return (best_bid * ask_vol + best_ask * bid_vol) / total_vol
    def get_volume_imbalance(self, order_depth: OrderDepth, levels: int = 3) -> float:
        bid_vol, ask_vol = 0.0, 0.0;
        try:
            sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True); sorted_asks = sorted(order_depth.sell_orders.keys())
            for i in range(min(levels, len(sorted_bids))): bid_vol += order_depth.buy_orders[sorted_bids[i]]
            for i in range(min(levels, len(sorted_asks))): ask_vol += abs(order_depth.sell_orders[sorted_asks[i]])
        except Exception: return 0.0
        total_vol = bid_vol + ask_vol
        if total_vol < self.constants.EPSILON: return 0.0
        return (bid_vol - ask_vol) / total_vol
    def manage_orders(self, product: str, current_pos: int, limit: int, desired_orders: List[Order]) -> List[Order]:
        final_orders: List[Order] = []; potential_buy_qty = sum(o.quantity for o in desired_orders if o.quantity > 0); potential_sell_qty = sum(abs(o.quantity) for o in desired_orders if o.quantity < 0)
        allowed_to_buy = limit - current_pos; allowed_to_sell = limit + current_pos; buy_scaling_factor = 1.0; sell_scaling_factor = 1.0
        if potential_buy_qty > allowed_to_buy: buy_scaling_factor = allowed_to_buy / potential_buy_qty if potential_buy_qty > 0 else 0
        if potential_sell_qty > allowed_to_sell: sell_scaling_factor = allowed_to_sell / potential_sell_qty if potential_sell_qty > 0 else 0
        for order in desired_orders:
            if order.quantity > 0: capped_qty = math.floor(order.quantity * buy_scaling_factor);
            elif order.quantity < 0: capped_qty = math.floor(abs(order.quantity) * sell_scaling_factor)
            else: capped_qty = 0
            # Ensure capped_qty respects original sign
            if order.quantity < 0: capped_qty = -capped_qty
            if abs(capped_qty) > 0:
                final_orders.append(Order(order.symbol, order.price, capped_qty))
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

        # --- 1. Calculate Metrics ---
        product_metrics = {}
        basket_std_dev = {'PICNIC_BASKET1': None, 'PICNIC_BASKET2': None}
        all_available_products = list(state.order_depths.keys())

        for product in all_available_products:
            metrics = {'mid_price': None, 'best_bid': None, 'best_ask': None, 'std_dev': None, 'average': None}
            order_depth = state.order_depths.get(product)
            if order_depth:
                best_bid, best_ask = self.get_best_bid_ask(order_depth)
                metrics['best_bid'] = best_bid; metrics['best_ask'] = best_ask
                mid_price = None
                if best_bid is not None and best_ask is not None:
                    # Use weighted mid-price preferably, fallback to simple avg
                    mid_price = self.calculate_weighted_mid_price(order_depth, best_bid, best_ask)
                    if mid_price is None: mid_price = (best_bid + best_ask) / 2.0
                elif best_bid is not None: mid_price = best_bid # Fallback if only one side exists
                elif best_ask is not None: mid_price = best_ask # Fallback
                metrics['mid_price'] = mid_price

                # Update price history & stats (Unchanged logic)
                stats_configs = { # Product -> (deque_key, window_attr, stats_needed)
                    'RAINFOREST_RESIN': ('resin_prices', 'RESIN_STD_WINDOW', True),
                    'SQUID_INK': ('squid_prices', 'SQUID_WINDOW', True),
                    'KELP': ('kelp_prices', 'KELP_WINDOW', True),
                    'PICNIC_BASKET1': ('basket1_prices', 'BASKET_VOL_WINDOW', False),
                    'PICNIC_BASKET2': ('basket2_prices', 'BASKET_VOL_WINDOW', False),
                }
                if product in stats_configs and mid_price is not None:
                    deque_key, window_attr, stats_needed = stats_configs[product]
                    if deque_key in trader_state:
                         trader_state[deque_key].append(mid_price)
                         prices = list(trader_state[deque_key])
                         if len(prices) >= 5 and NUMPY_AVAILABLE: # Require numpy for stats
                             try:
                                 p_std = np.std(prices)
                                 if stats_needed:
                                     metrics['average'] = np.mean(prices)
                                     metrics['std_dev'] = p_std
                                 if product in basket_std_dev: basket_std_dev[product] = p_std
                             except Exception: pass # Ignore stats calculation errors
            product_metrics[product] = metrics

        # --- 2. Calculate Basket Values --- (Unchanged)
        theo_fv_b1, rep_buy_b1, rep_sell_b1 = self.calculate_basket_values(self.constants.BASKET1_COMPONENTS, product_metrics)
        theo_fv_b2, rep_buy_b2, rep_sell_b2 = self.calculate_basket_values(self.constants.BASKET2_COMPONENTS, product_metrics)

        # --- 3. Volcanic Voucher Analysis ---
        voucher_analysis = {'tte': None, 'st': None, 'fit_coeffs': None, 'base_iv': None, 'data': {}}
        if self._voucher_strategy_enabled:
            # Calculate TTE once per timestamp
            voucher_analysis['tte'] = self._calculate_tte(timestamp)

            # Get Underlying Price (St)
            st_metrics = product_metrics.get(self.constants.UNDERLYING_PRODUCT)
            st_price = st_metrics.get('mid_price') if st_metrics else None
            voucher_analysis['st'] = st_price

            if voucher_analysis['tte'] is not None and st_price is not None:
                mt_vt_pairs = []
                for voucher_prod in self.constants.VOUCHER_PRODUCTS:
                    if voucher_prod not in state.order_depths: continue # Skip if voucher not trading

                    voucher_metrics = product_metrics.get(voucher_prod)
                    vt_price = voucher_metrics.get('mid_price') if voucher_metrics else None
                    strike = self.constants.VOUCHER_INFO[voucher_prod]['strike']

                    if vt_price is not None:
                        m_t = self._calculate_m_t(st_price, strike, voucher_analysis['tte'])
                        v_t = self._calculate_iv_wrapper(vt_price, st_price, strike, voucher_analysis['tte'])

                        voucher_analysis['data'][voucher_prod] = {
                            'Vt': vt_price,
                            'K': strike,
                            'm_t': m_t,
                            'market_v_t': v_t,
                            'best_bid': voucher_metrics.get('best_bid'),
                            'best_ask': voucher_metrics.get('best_ask')
                        }
                        if m_t is not None and v_t is not None:
                             mt_vt_pairs.append((m_t, v_t))
                    else:
                         voucher_analysis['data'][voucher_prod] = {'Vt': None, 'K': strike, 'm_t': None, 'market_v_t': None, 'best_bid': None, 'best_ask': None}


                # Fit the parabola if enough data points
                fit_result = self._fit_parabola(mt_vt_pairs)
                if fit_result:
                    voucher_analysis['fit_coeffs'], voucher_analysis['base_iv'] = fit_result
                    trader_state['last_fit_coeffs'] = fit_result[0] # Store the latest good coefficients
                    # Optional: Track base IV history
                    # if 'voucher_base_ivs' in trader_state:
                    #      trader_state['voucher_base_ivs'].append(fit_result[1])
                else:
                    # Use last known good coefficients if fit fails? Or disable trading?
                    voucher_analysis['fit_coeffs'] = trader_state.get('last_fit_coeffs', None) # Use last good fit if available
                    voucher_analysis['base_iv'] = None # Indicate current fit failed


        # --- 4. Generate Orders ---
        for product in all_available_products:
            orders: List[Order] = []
            order_depth = state.order_depths.get(product)
            current_pos = current_positions.get(product, 0)
            limit = self.constants.POSITION_LIMIT.get(product, 0)
            metrics = product_metrics.get(product, {})
            best_bid, best_ask = metrics.get('best_bid'), metrics.get('best_ask')

            # --- Existing Strategies (Baskets, Components, Resin, Squid, Kelp) ---
            # (Keep the logic for these products exactly as it was in the input code)
            # ... [Omitted for Brevity - Paste your existing logic for these products here] ...
            # --- Basket Value Strategy (with Dynamic Threshold) ---
            if product == 'PICNIC_BASKET1' or product == 'PICNIC_BASKET2':
                 # ... [Your Basket Logic Here] ...
                 theo_fv, rep_buy, rep_sell = (theo_fv_b1, rep_buy_b1, rep_sell_b1) if product == 'PICNIC_BASKET1' else (theo_fv_b2, rep_buy_b2, rep_sell_b2)
                 std_dev = basket_std_dev.get(product)
                 current_threshold = self.constants.BASKET_BASE_VALUE_THRESHOLD
                 if std_dev is not None and std_dev > 0: current_threshold += std_dev * self.constants.BASKET_VOL_THRESHOLD_SCALAR
                 if rep_buy is not None and rep_sell is not None and best_ask is not None and best_bid is not None:
                    buy_value_signal = rep_sell - best_ask
                    if buy_value_signal > current_threshold:
                        target_price = best_ask + 1
                        base_qty = self.constants.BASKET_PASSIVE_QTY
                        skewed_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.BASKET_NEUTRALITY_SCALE, True)
                        if skewed_qty > 0: orders.append(Order(product, target_price, skewed_qty))
                    sell_value_signal = best_bid - rep_buy
                    if sell_value_signal > current_threshold:
                        target_price = best_bid - 1
                        base_qty = self.constants.BASKET_PASSIVE_QTY
                        skewed_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.BASKET_NEUTRALITY_SCALE, False)
                        if skewed_qty > 0: orders.append(Order(product, target_price, -skewed_qty))

            # --- Component MM (Quoting around Mid-Price) ---
            elif product in ['CROISSANTS', 'JAMS', 'DJEMBES']:
                 # ... [Your Component MM Logic Here] ...
                 mid_price = metrics.get('mid_price')
                 if mid_price is not None and best_bid is not None and best_ask is not None:
                    spread_needed = self.constants.COMPONENT_SPREAD
                    target_bid = math.floor(mid_price - spread_needed / 2.0)
                    target_ask = math.ceil(mid_price + spread_needed / 2.0)
                    target_bid = min(target_bid, best_bid)
                    target_ask = max(target_ask, best_ask)
                    if target_ask - target_bid < spread_needed:
                         target_bid = math.floor(mid_price - spread_needed / 2.0)
                         target_ask = math.ceil(mid_price + spread_needed / 2.0)
                    if target_ask > target_bid :
                        base_qty = self.constants.COMPONENT_MM_QTY
                        buy_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.COMPONENT_NEUTRALITY_SCALE, True)
                        sell_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.COMPONENT_NEUTRALITY_SCALE, False)
                        if buy_qty > 0 and target_bid < best_ask: orders.append(Order(product, target_bid, buy_qty))
                        if sell_qty > 0 and target_ask > best_bid: orders.append(Order(product, target_ask, -sell_qty))

            # --- RAINFOREST_RESIN ---
            elif product == 'RAINFOREST_RESIN':
                 # ... [Your Resin Logic Here] ...
                 ema = trader_state.get('resin_ema'); mid_price = metrics.get('mid_price'); current_avg = metrics.get('average')
                 target_buy_price, target_sell_price = None, None
                 if mid_price is not None:
                    if ema is None and current_avg is not None: ema = current_avg
                    elif ema is not None: ema = (self.constants.RESIN_EMA_ALPHA * mid_price + (1 - self.constants.RESIN_EMA_ALPHA) * ema)
                    trader_state['resin_ema'] = ema
                 if ema is not None: target_buy_price = math.floor(ema - self.constants.RESIN_SPREAD_BUFFER); target_sell_price = math.ceil(ema + self.constants.RESIN_SPREAD_BUFFER)
                 elif mid_price is not None: target_buy_price = math.floor(mid_price - self.constants.RESIN_SPREAD_BUFFER); target_sell_price = math.ceil(mid_price + self.constants.RESIN_SPREAD_BUFFER)
                 if target_buy_price is not None and target_sell_price is not None and target_sell_price > target_buy_price and best_bid is not None and best_ask is not None:
                     base_qty = self.constants.RESIN_PASSIVE_QTY; buy_qty = 0; sell_qty = 0
                     if target_buy_price < best_ask: buy_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.RESIN_NEUTRALITY_SCALE, True)
                     if buy_qty > 0: orders.append(Order(product, int(target_buy_price), int(buy_qty)))
                     if target_sell_price > best_bid: sell_qty = calculate_skewed_quantity(base_qty, current_pos, limit, self.constants.RESIN_NEUTRALITY_SCALE, False)
                     if sell_qty > 0: orders.append(Order(product, int(target_sell_price), -int(sell_qty)))


            # --- SQUID_INK ---
            elif product == 'SQUID_INK':
                 # ... [Your Squid Logic Here, ensure it uses metrics dictionary] ...
                 holding_info = trader_state.get('squid_holding_info'); mid_price = metrics.get('mid_price'); current_avg = metrics.get('average'); current_std = metrics.get('std_dev')
                 dynamic_support, dynamic_resistance = None, None
                 if mid_price is not None and current_avg is not None and current_std is not None and best_bid is not None and best_ask is not None:
                     current_std = max(self.constants.SQUID_MIN_STD, current_std); support = round(current_avg - self.constants.SQUID_SR_STD_FACTOR * current_std); resistance = round(current_avg + self.constants.SQUID_SR_STD_FACTOR * current_std)
                     if resistance > support + self.constants.SQUID_MIN_SR_SPREAD: dynamic_support, dynamic_resistance = support, resistance
                     if holding_info:
                         entry_price, side, trade_qty = holding_info['entry_price'], holding_info['side'], int(holding_info['qty']); stop_loss_triggered, profit_take_triggered = False, False; exit_price, exit_qty = None, None
                         stop_loss_buy_trigger = entry_price - self.constants.SQUID_STOP_LOSS_STD_FACTOR * current_std; stop_loss_sell_trigger = entry_price + self.constants.SQUID_STOP_LOSS_STD_FACTOR * current_std
                         if side == 'buy' and mid_price < stop_loss_buy_trigger: stop_loss_triggered, exit_price, exit_qty = True, best_bid, -trade_qty
                         elif side == 'sell' and mid_price > stop_loss_sell_trigger: stop_loss_triggered, exit_price, exit_qty = True, best_ask, trade_qty
                         if not stop_loss_triggered:
                             profit_take_buy_trigger = entry_price + self.constants.SQUID_EXIT_STD_FACTOR * current_std; profit_take_sell_trigger = entry_price - self.constants.SQUID_EXIT_STD_FACTOR * current_std
                             if side == 'buy' and mid_price > profit_take_buy_trigger: profit_take_triggered, exit_price, exit_qty = True, best_bid, -trade_qty
                             elif side == 'sell' and mid_price < profit_take_sell_trigger: profit_take_triggered, exit_price, exit_qty = True, best_ask, trade_qty
                         if (stop_loss_triggered or profit_take_triggered) and exit_price is not None and exit_qty is not None:
                             orders.append(Order(product, int(exit_price), int(exit_qty))); trader_state['squid_holding_info'] = None; holding_info = None
                     if holding_info is None and dynamic_support is not None and dynamic_resistance is not None:
                         base_entry_qty = self.constants.SQUID_BASE_AGGRESSION_QTY
                         rem_buy_capacity = max(0, limit - current_pos); rem_sell_capacity = max(0, limit + current_pos)
                         scaled_buy_qty = math.floor(base_entry_qty * min(1.0, rem_buy_capacity / base_entry_qty if base_entry_qty > 0 else 0) * self.constants.SQUID_AGGRESSION_POS_SCALE)
                         scaled_sell_qty = math.floor(base_entry_qty * min(1.0, rem_sell_capacity / base_entry_qty if base_entry_qty > 0 else 0) * self.constants.SQUID_AGGRESSION_POS_SCALE)
                         if best_ask <= dynamic_support and scaled_buy_qty > 0:
                             entry_price = best_ask
                             orders.append(Order(product, int(entry_price), int(scaled_buy_qty)))
                             trader_state['squid_holding_info'] = {'entry_price': entry_price, 'qty': scaled_buy_qty, 'side': 'buy'}; holding_info = trader_state['squid_holding_info']
                         elif best_bid >= dynamic_resistance and scaled_sell_qty > 0:
                             entry_price = best_bid
                             orders.append(Order(product, int(entry_price), -int(scaled_sell_qty)))
                             trader_state['squid_holding_info'] = {'entry_price': entry_price, 'qty': scaled_sell_qty, 'side': 'sell'}; holding_info = trader_state['squid_holding_info']
                 if holding_info is None and abs(current_pos) > self.constants.SQUID_NEUTRAL_POS_THRESH and best_bid is not None and best_ask is not None:
                       neutral_qty = self.constants.SQUID_NEUTRAL_QTY
                       if current_pos > 0: orders.append(Order(product, best_bid, -int(min(neutral_qty, current_pos)))) # Use min
                       elif current_pos < 0: orders.append(Order(product, best_ask, int(min(neutral_qty, abs(current_pos))))) # Use min


            # --- KELP ---
            elif product == 'KELP':
                 # ... [Your Kelp Logic Here, ensure it uses metrics dictionary] ...
                 agg_order_placed = False; mid_price = metrics.get('mid_price'); current_avg = metrics.get('average'); current_std = metrics.get('std_dev')
                 if mid_price is not None and current_avg is not None and current_std is not None and best_bid is not None and best_ask is not None:
                     current_std = max(self.constants.KELP_MIN_STD, current_std); deviation = mid_price - current_avg; deviation_threshold = self.constants.KELP_AGGRESSION_STD_FACTOR * current_std
                     base_agg_qty = self.constants.KELP_BASE_AGGRESSION_QTY
                     rem_buy_capacity = max(0, limit - current_pos); rem_sell_capacity = max(0, limit + current_pos)
                     scaled_buy_qty = math.floor(base_agg_qty * min(1.0, rem_buy_capacity / base_agg_qty if base_agg_qty > 0 else 0) * self.constants.KELP_AGGRESSION_POS_SCALE)
                     scaled_sell_qty = math.floor(base_agg_qty * min(1.0, rem_sell_capacity / base_agg_qty if base_agg_qty > 0 else 0) * self.constants.KELP_AGGRESSION_POS_SCALE)
                     if deviation < -deviation_threshold and scaled_buy_qty > 0: orders.append(Order(product, best_ask, int(scaled_buy_qty))); agg_order_placed = True
                     elif deviation > deviation_threshold and scaled_sell_qty > 0: orders.append(Order(product, best_bid, -int(scaled_sell_qty))); agg_order_placed = True
                 if not agg_order_placed and best_bid is not None and best_ask is not None:
                     spread = best_ask - best_bid
                     if spread >= self.constants.KELP_MIN_PASSIVE_SPREAD:
                         imbalance = self.get_volume_imbalance(order_depth, levels=3); our_bid, our_ask = best_bid + 1, best_ask - 1
                         if imbalance > self.constants.KELP_IMBALANCE_THRESH: our_bid = min(our_bid + 1, best_ask - 1)
                         elif imbalance < -self.constants.KELP_IMBALANCE_THRESH: our_ask = max(our_ask - 1, best_bid + 1)
                         if our_bid >= our_ask: our_bid, our_ask = best_bid, best_ask # Reset if crossed
                         if our_ask > our_bid:
                             buy_qty, sell_qty = self.constants.KELP_PASSIVE_QTY, self.constants.KELP_PASSIVE_QTY
                             pos_limit_fraction = abs(current_pos) / limit if limit > 0 else 0
                             if pos_limit_fraction > self.constants.KELP_POS_ADJUST_THRESH:
                                 if current_pos > 0: sell_qty = round(sell_qty * self.constants.KELP_POS_ADJUST_FACTOR)
                                 elif current_pos < 0: buy_qty = round(buy_qty * self.constants.KELP_POS_ADJUST_FACTOR)
                             if buy_qty > 0: orders.append(Order(product, int(our_bid), int(buy_qty)))
                             if sell_qty > 0: orders.append(Order(product, int(our_ask), -int(sell_qty)))

            # --- Volcanic Voucher Strategy (IV Smile Arb) ---
            elif product in self.constants.VOUCHER_PRODUCTS and self._voucher_strategy_enabled:
                coeffs = voucher_analysis.get('fit_coeffs')
                voucher_data = voucher_analysis.get('data', {}).get(product)

                if coeffs is not None and voucher_data and NUMPY_AVAILABLE:
                    m_t = voucher_data.get('m_t')
                    market_v_t = voucher_data.get('market_v_t')
                    voucher_best_bid = voucher_data.get('best_bid')
                    voucher_best_ask = voucher_data.get('best_ask')

                    if m_t is not None and market_v_t is not None:
                        try:
                            fitted_v_t = np.polyval(coeffs, m_t)

                            # Check for trading signals based on vol difference
                            vol_diff = market_v_t - fitted_v_t
                            base_trade_qty = self.constants.VOUCHER_TRADE_QTY

                            # Signal: Market IV is too high -> Sell Voucher
                            if vol_diff > self.constants.VOL_DIFF_THRESHOLD and voucher_best_bid is not None:
                                target_price = voucher_best_bid # Hit the bid to sell
                                skewed_qty = calculate_skewed_quantity(base_trade_qty, current_pos, limit, self.constants.VOUCHER_NEUTRALITY_SCALE, False)
                                if skewed_qty > 0:
                                    orders.append(Order(product, target_price, -skewed_qty))
                                    # print(f"SELL {product} @ {target_price} Qty: {skewed_qty} (Market IV: {market_v_t:.3f}, Fitted IV: {fitted_v_t:.3f})")


                            # Signal: Market IV is too low -> Buy Voucher
                            elif vol_diff < -self.constants.VOL_DIFF_THRESHOLD and voucher_best_ask is not None:
                                target_price = voucher_best_ask # Lift the ask to buy
                                skewed_qty = calculate_skewed_quantity(base_trade_qty, current_pos, limit, self.constants.VOUCHER_NEUTRALITY_SCALE, True)
                                if skewed_qty > 0:
                                    orders.append(Order(product, target_price, skewed_qty))
                                    # print(f"BUY {product} @ {target_price} Qty: {skewed_qty} (Market IV: {market_v_t:.3f}, Fitted IV: {fitted_v_t:.3f})")

                        except Exception as e:
                            print(f"Error during voucher order generation for {product}: {e}")


            # Store orders for the product
            # Only add orders if the list is not empty to avoid empty lists in the result
            if orders:
                 all_orders[product] = orders

        # --- 5. Apply Position Limits --- (Unchanged logic, now uses all_orders directly)
        final_result: Dict[str, List[Order]] = {}
        for product, limit in self.constants.POSITION_LIMIT.items():
             current_pos = current_positions.get(product, 0);
             desired_orders_for_product = all_orders.get(product, []) # Get potential orders
             if desired_orders_for_product: # Only manage if there are orders
                 final_result[product] = self.manage_orders(product, current_pos, limit, desired_orders_for_product)

        # --- 6. Save State and Return ---
        traderData = self.save_trader_data(trader_state)
        # Clean the final result dict to remove products with no orders after limit checks
        cleaned_result = {k: v for k, v in final_result.items() if v}
        return cleaned_result, conversions, traderData

# Example calculate_skewed_quantity function (if not defined globally)
def calculate_skewed_quantity(base_qty: int, current_pos: int, limit: int, neutrality_scale: float, is_buy_order: bool) -> int:
    if limit <= 0: return base_qty # Or return 0 if limit is 0? Depends on desired behavior. Let's return base_qty.
    scale = max(0.0, min(1.0, neutrality_scale))
    position_fraction = 0.0

    # If buying, check positive position; if selling, check negative position.
    if is_buy_order and current_pos > 0:
        position_fraction = current_pos / limit
    elif not is_buy_order and current_pos < 0:
        position_fraction = abs(current_pos) / limit
    # If position helps the desired trade (e.g., buying when short), don't penalize quantity
    elif (is_buy_order and current_pos < 0) or (not is_buy_order and current_pos > 0):
         position_fraction = 0.0

    # Ensure position fraction doesn't exceed 1 due to potential overfills etc.
    position_fraction = max(0.0, min(1.0, position_fraction))

    quantity_factor = max(0.0, 1.0 - (position_fraction * scale))
    return math.floor(base_qty * quantity_factor)