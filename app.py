import os
import sys
import json
import logging
import datetime
import re
import warnings
from threading import Lock
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.utils
import urllib.request
import xml.etree.ElementTree as ET
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Proactively download NLTK vader_lexicon
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK vader_lexicon: {e}")

# Add project root directory to path for Kronos imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Kronos'))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
    logger.info("Kronos model classes imported successfully!")
except ImportError as e:
    MODEL_AVAILABLE = False
    logger.error(f"Error importing Kronos model: {e}. Will fall back to simulation if needed.")

app = Flask(__name__)
CORS(app)

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'portfolio.json')

# Global variables for model state
model_lock = Lock()
loaded_model_key = None
loaded_device = None
tokenizer = None
model = None
predictor = None

AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, fast CPU inference'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced accuracy and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, higher prediction accuracy'
    }
}

# --- Helper Functions ---

def load_portfolio():
    """Load portfolio items from JSON file."""
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read portfolio file: {e}")
        return []

def save_portfolio(portfolio):
    """Save portfolio items to JSON file."""
    try:
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to write portfolio file: {e}")
        return False

def get_indian_ticker(symbol):
    """Normalize ticker symbol: check if valid as-is (e.g. US stocks), else append .NS for Indian stocks."""
    symbol = symbol.strip().upper()
    if not symbol:
        return ""
    if '.' in symbol:
        return symbol
    # Try fetching a tiny history to verify if it is valid as-is
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d")
        if df is not None and not df.empty:
            return symbol
    except Exception:
        pass
        
    # Suffix fallback for Indian exchanges
    return f"{symbol}.NS"

def fetch_stock_history(ticker_symbol, period="2y"):
    """Fetch historical daily OHLCV data using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None, "No historical data found for symbol"
        
        df = df.reset_index()
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure date column exists
        if 'date' not in df.columns:
            if 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'date'}, inplace=True)
            elif 'index' in df.columns:
                df.rename(columns={'index': 'date'}, inplace=True)
            else:
                return None, "Date column missing from yfinance response"
                
        # Remove timezone info for model compatibility
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure we have OHLCV
        required = ['open', 'high', 'low', 'close', 'volume']
        for req in required:
            if req not in df.columns:
                return None, f"Required column {req} missing from data"
            df[req] = pd.to_numeric(df[req], errors='coerce')
            
        # Fill missing values
        df = df.dropna(subset=['close'])
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df['close'])
        df['low'] = df['low'].fillna(df['close'])
        df['volume'] = df['volume'].fillna(0)
        
        # Calculate amount = close * volume (Kronos expectation)
        df['amount'] = df['close'] * df['volume']
        
        return df, None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None, str(e)

def calculate_dmsd_drift(symbol, net_sentiment, lookback_days=30):
    """Calculate combined drift rate using sentiment, stock momentum, and market trend."""
    import numpy as np
    
    # 1. Sentiment Drift Rate
    r_sentiment = net_sentiment * 0.003
    
    # 2. Stock Momentum Drift Rate (past 30 business days)
    r_stock = 0.0
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d") # fetch plenty of data
        if not df.empty and len(df) >= lookback_days:
            closes = df['Close'].iloc[-lookback_days:].values
            returns = np.diff(closes) / closes[:-1]
            r_stock = np.mean(returns)
    except Exception as e:
        logger.warning(f"Failed to fetch stock drift for {symbol}: {e}")
        
    # 3. Market Drift Rate (Nifty 50)
    r_nifty = 0.0
    try:
        nifty = yf.Ticker("^NSEI")
        df_n = nifty.history(period="60d")
        if not df_n.empty and len(df_n) >= lookback_days:
            closes_n = df_n['Close'].iloc[-lookback_days:].values
            returns_n = np.diff(closes_n) / closes_n[:-1]
            r_nifty = np.mean(returns_n)
    except Exception as e:
        logger.warning(f"Failed to fetch market drift: {e}")
        
    # Combine drift rates
    r_drift = (0.6 * r_sentiment) + (0.3 * r_stock) + (0.1 * r_nifty)
    # Cap daily drift to +/- 0.2%
    return max(-0.002, min(0.002, r_drift))

def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    """Limit price movements to +/- 10% per step (standard Indian circuit limits)."""
    pred_df = pred_df.reset_index(drop=True)
    cols = ['open', 'high', 'low', 'close']
    pred_df[cols] = pred_df[cols].astype('float64')
    
    current_close = last_close
    for i in range(len(pred_df)):
        limit_up = current_close * (1 + limit_rate)
        limit_down = current_close * (1 - limit_rate)
        
        for col in cols:
            val = pred_df.at[i, col]
            if pd.notna(val):
                clipped = max(min(val, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)
                
        current_close = float(pred_df.at[i, 'close'])
    return pred_df

# --- Routing ---

@app.route('/')
def home():
    return render_template('index.html')

def audit_active_predictions(portfolio):
    """
    Check active predictions in the portfolio, verify if the trading horizon is met,
    and calculate performance metrics.
    """
    modified = False
    for item in portfolio:
        if item.get('prediction_status') != 'ACTIVE':
            continue
            
        symbol = item['symbol']
        purchase_date_str = item.get('purchase_date', '')
        horizon = int(item.get('prediction_horizon_days', 10))
        predicted_target = float(item.get('predicted_target_price', 0.0))
        purchase_price = float(item.get('purchase_price', 0.0))
        predicted_direction = item.get('predicted_direction', 'UP')
        
        if not purchase_date_str:
            continue
            
        try:
            # Fetch daily data for counting trading days post-purchase
            df, err = fetch_stock_history(symbol, period="1y")
            if df is not None and not df.empty:
                purchase_date = datetime.date.fromisoformat(purchase_date_str)
                df['date_only'] = df['date'].dt.date
                
                # Rows after purchase date, sorted chronologically
                post_purchase_df = df[df['date_only'] > purchase_date].sort_values('date_only')
                
                if len(post_purchase_df) >= horizon:
                    # Horizon reached
                    target_row = post_purchase_df.iloc[horizon - 1]
                    actual_target_price = float(target_row['close'])
                    evaluation_date_str = target_row['date_only'].isoformat()
                    
                    # Calculate deviation and directional matching
                    abs_error = abs(predicted_target - actual_target_price)
                    pct_error = (abs_error / actual_target_price) * 100
                    
                    if predicted_direction == 'UP':
                        dir_match = actual_target_price > purchase_price
                    else:
                        dir_match = actual_target_price < purchase_price
                        
                    # Update fields
                    item['actual_target_price'] = actual_target_price
                    item['evaluation_date'] = evaluation_date_str
                    item['absolute_error'] = abs_error
                    item['percentage_error'] = pct_error
                    item['directional_match'] = bool(dir_match)
                    item['prediction_status'] = 'VERIFIED'
                    modified = True
                    logger.info(f"Verified prediction for {symbol}: target={predicted_target:.2f}, actual={actual_target_price:.2f}, match={dir_match}")
                else:
                    # Capture current trading progress
                    item['trading_days_elapsed'] = len(post_purchase_df)
                    item['trading_days_remaining'] = horizon - len(post_purchase_df)
        except Exception as e:
            logger.error(f"Error auditing prediction for {symbol}: {e}")
            
    if modified:
        save_portfolio(portfolio)

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio_api():
    """Retrieve portfolio with current pricing, calculations, and performance metrics."""
    portfolio = load_portfolio()
    
    # Audit predictions
    if portfolio:
        audit_active_predictions(portfolio)
        
    if not portfolio:
        return jsonify({
            'holdings': [],
            'summary': {
                'total_cost': 0.0,
                'total_value': 0.0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'today_pnl': 0.0,
                'today_pnl_percent': 0.0
            }
        })
        
    holdings = []
    total_cost = 0.0
    total_value = 0.0
    today_pnl = 0.0
    
    for item in portfolio:
        symbol = item['symbol']
        shares = float(item['shares'])
        purchase_price = float(item['purchase_price'])
        purchase_date = item.get('purchase_date', '')
        name = item.get('name', symbol)
        
        current_price = purchase_price
        prev_close = purchase_price
        change_pct = 0.0
        
        try:
            # Query Yahoo Finance for live price metrics
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            
            # Extract current price
            if 'last_price' in fast_info and fast_info['last_price'] is not None:
                current_price = float(fast_info['last_price'])
            else:
                # Fallback to fetching last 2 days of history
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    if len(hist) > 1:
                        prev_close = float(hist['Close'].iloc[-2])
            
            # Extract previous close / daily change
            if 'previous_close' in fast_info and fast_info['previous_close'] is not None:
                prev_close = float(fast_info['previous_close'])
            elif 'regularMarketPreviousClose' in ticker.info:
                prev_close = float(ticker.info['regularMarketPreviousClose'])
                
            if prev_close > 0:
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
        except Exception as e:
            logger.error(f"Error fetching live updates for portfolio ticker {symbol}: {e}")
            
        cost = shares * purchase_price
        val = shares * current_price
        pnl = val - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0.0
        
        today_val_change = (current_price - prev_close) * shares
        today_pnl += today_val_change
        
        total_cost += cost
        total_value += val
        
        holdings.append({
            'symbol': symbol,
            'name': name,
            'shares': shares,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date,
            'purchase_datetime': item.get('purchase_datetime', purchase_date),
            'current_price': current_price,
            'prev_close': prev_close,
            'change_percent': change_pct,
            'total_cost': cost,
            'total_value': val,
            'pnl': pnl,
            'pnl_percent': pnl_pct,
            
            # Prediction auditor fields
            'prediction_status': item.get('prediction_status', 'NONE'),
            'prediction_horizon_days': item.get('prediction_horizon_days', 10),
            'predicted_target_price': item.get('predicted_target_price'),
            'predicted_direction': item.get('predicted_direction'),
            'actual_target_price': item.get('actual_target_price'),
            'absolute_error': item.get('absolute_error'),
            'percentage_error': item.get('percentage_error'),
            'directional_match': item.get('directional_match'),
            'evaluation_date': item.get('evaluation_date'),
            'trading_days_elapsed': item.get('trading_days_elapsed', 0),
            'trading_days_remaining': item.get('trading_days_remaining', 0)
        })
        
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0.0
    
    prev_total_value = total_value - today_pnl
    today_pnl_pct = (today_pnl / prev_total_value) * 100 if prev_total_value > 0 else 0.0
    
    return jsonify({
        'holdings': holdings,
        'summary': {
            'total_cost': total_cost,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_pct,
            'today_pnl': today_pnl,
            'today_pnl_percent': today_pnl_pct
        }
    })

@app.route('/api/portfolio', methods=['POST'])
def add_portfolio_item():
    """Add a stock transaction to portfolio."""
    data = request.json
    if not data or 'symbol' not in data or 'shares' not in data or 'purchase_price' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
        
    raw_symbol = data['symbol']
    symbol = get_indian_ticker(raw_symbol)
    shares = float(data['shares'])
    purchase_price = float(data['purchase_price'])
    purchase_date = data.get('purchase_date', datetime.date.today().isoformat())
    purchase_datetime = datetime.datetime.now().isoformat()
    
    # Try fetching stock name
    name = symbol
    try:
        ticker = yf.Ticker(symbol)
        if 'longName' in ticker.info:
            name = ticker.info['longName']
        elif 'shortName' in ticker.info:
            name = ticker.info['shortName']
    except Exception as e:
        logger.warning(f"Could not fetch company name for {symbol}: {e}")
        
    portfolio = load_portfolio()
    
    # Check if transaction already exists for symbol, if so, average it out or add as new
    existing_item = next((item for item in portfolio if item['symbol'] == symbol), None)
    if existing_item:
        total_shares = existing_item['shares'] + shares
        total_cost = (existing_item['shares'] * existing_item['purchase_price']) + (shares * purchase_price)
        existing_item['shares'] = total_shares
        existing_item['purchase_price'] = total_cost / total_shares
        
        # If prediction details are passed in payload, update to active prediction
        if 'predicted_target_price' in data:
            existing_item['prediction_status'] = 'ACTIVE'
            existing_item['purchase_date'] = purchase_date
            existing_item['purchase_datetime'] = purchase_datetime
            existing_item['predicted_target_price'] = float(data['predicted_target_price'])
            existing_item['predicted_direction'] = data.get('predicted_direction', 'UP')
            existing_item['prediction_horizon_days'] = int(data.get('prediction_horizon_days', 10))
            existing_item['prediction_vector'] = data.get('prediction_vector', [])
            
            # Clear historical verified metrics
            existing_item.pop('actual_target_price', None)
            existing_item.pop('absolute_error', None)
            existing_item.pop('percentage_error', None)
            existing_item.pop('directional_match', None)
            existing_item.pop('evaluation_date', None)
            existing_item.pop('trading_days_elapsed', None)
            existing_item.pop('trading_days_remaining', None)
    else:
        new_item = {
            'symbol': symbol,
            'name': name,
            'shares': shares,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date,
            'purchase_datetime': purchase_datetime,
            'prediction_status': 'NONE'
        }
        if 'predicted_target_price' in data:
            new_item['prediction_status'] = 'ACTIVE'
            new_item['predicted_target_price'] = float(data['predicted_target_price'])
            new_item['predicted_direction'] = data.get('predicted_direction', 'UP')
            new_item['prediction_horizon_days'] = int(data.get('prediction_horizon_days', 10))
            new_item['prediction_vector'] = data.get('prediction_vector', [])
            
        portfolio.append(new_item)
        
    if save_portfolio(portfolio):
        return jsonify({'success': True, 'message': f'Successfully added {symbol} to portfolio'})
    else:
        return jsonify({'error': 'Failed to save portfolio'}), 500

@app.route('/api/portfolio', methods=['DELETE'])
def remove_portfolio_item():
    """Remove all shares of a stock from the portfolio."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
        
    symbol = symbol.strip().upper()
    portfolio = load_portfolio()
    new_portfolio = [item for item in portfolio if item['symbol'] != symbol]
    
    if len(portfolio) == len(new_portfolio):
        return jsonify({'error': f'Symbol {symbol} not found in portfolio'}), 404
        
    if save_portfolio(new_portfolio):
        return jsonify({'success': True, 'message': f'Successfully removed {symbol} from portfolio'})
    else:
        return jsonify({'error': 'Failed to update portfolio'}), 500

LOCAL_PORTFOLIO_STOCKS = [
    {'symbol': 'TRENT.NS', 'name': 'Trent Limited', 'exchange': 'NSE'},
    {'symbol': 'SWSOLAR.NS', 'name': 'Sterling and Wilson Renewable Energy (Sterling)', 'exchange': 'NSE'},
    {'symbol': 'STERLINGTOOLS.NS', 'name': 'Sterling Tools Limited', 'exchange': 'NSE'},
    {'symbol': 'SUNPHARMA.NS', 'name': 'Sun Pharmaceutical Industries (Sun Pharma)', 'exchange': 'NSE'},
    {'symbol': 'SBIN.NS', 'name': 'State Bank Of India (SBI)', 'exchange': 'NSE'},
    {'symbol': 'M&M.NS', 'name': 'Mahindra & Mahindra (M&M)', 'exchange': 'NSE'},
    {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank Limited', 'exchange': 'NSE'},
    {'symbol': 'COCHINSHIP.NS', 'name': 'Cochin Shipyard Limited', 'exchange': 'NSE'},
    {'symbol': 'CDSL.NS', 'name': 'CDSL (Central Depository Services)', 'exchange': 'NSE'},
    {'symbol': 'BSOFT.NS', 'name': 'Birlasoft Limited', 'exchange': 'NSE'},
    {'symbol': 'BSE.NS', 'name': 'BSE Limited', 'exchange': 'NSE'},
    {'symbol': 'BIOCON.NS', 'name': 'Biocon Limited', 'exchange': 'NSE'},
    {'symbol': 'UNIENTER.NS', 'name': 'Uniphos Enterprises Limited', 'exchange': 'NSE'},
    {'symbol': 'SALORA.NS', 'name': 'Salora International Limited', 'exchange': 'NSE'},
    {'symbol': 'GOODLUCK.NS', 'name': 'Goodluck India Limited', 'exchange': 'NSE'},
    {'symbol': 'NCC.NS', 'name': 'NCC Limited', 'exchange': 'NSE'},
    {'symbol': 'TATAPOWER.NS', 'name': 'Tata Power Company Limited', 'exchange': 'NSE'},
    {'symbol': 'NMDC.NS', 'name': 'NMDC Limited', 'exchange': 'NSE'},
    {'symbol': 'STOVEKRAFT.NS', 'name': 'Stove Kraft Limited', 'exchange': 'NSE'},
    {'symbol': 'ITC.NS', 'name': 'ITC Limited', 'exchange': 'NSE'},
    {'symbol': 'NESTLEIND.NS', 'name': 'Nestlé India Limited', 'exchange': 'NSE'},
    {'symbol': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever (HUL)', 'exchange': 'NSE'},
    {'symbol': 'TCS.NS', 'name': 'TCS (Tata Consultancy Services)', 'exchange': 'NSE'}
]

@app.route('/api/search', methods=['GET'])
def search_stocks():
    """Search for stock symbols using local portfolio definitions and Yahoo Finance suggestions."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'suggestions': []})
        
    # 1. Fuzzy match against our local portfolio definitions first (to ensure user-sheet stocks always appear)
    q_clean = query.lower()
    local_matches = []
    seen_symbols = set()
    
    for s in LOCAL_PORTFOLIO_STOCKS:
        # Check symbol or name (or parts)
        if q_clean in s['symbol'].lower() or q_clean in s['name'].lower() or any(part in s['name'].lower() for part in q_clean.split()):
            local_matches.append({
                'symbol': s['symbol'],
                'name': s['name'],
                'exchange': s['exchange'],
                'is_indian': True
            })
            seen_symbols.add(s['symbol'])
            
    # 2. Query Yahoo Finance
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    
    try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=5)
        
        suggestions = list(local_matches)
        
        if res.status_code == 200:
            data = res.json()
            quotes = data.get('quotes', [])
            
            for quote in quotes:
                symbol = quote.get('symbol', '')
                if symbol in seen_symbols:
                    continue
                quote_type = quote.get('quoteType', '')
                exch = quote.get('exchDisp', '')
                
                # Check if it's an equity and matches Indian exchanges or has Indian suffixes
                is_indian = (exch in ['NSE', 'BSE']) or symbol.endswith('.NS') or symbol.endswith('.BO')
                
                if quote_type == 'EQUITY':
                    suggestions.append({
                        'symbol': symbol,
                        'name': quote.get('shortname') or quote.get('longname') or symbol,
                        'exchange': exch,
                        'is_indian': is_indian
                    })
            
            # Sort remaining suggestions to put Indian stocks first
            # But preserve our direct local matches at the very top of the list
            non_local = suggestions[len(local_matches):]
            non_local.sort(key=lambda x: x['is_indian'], reverse=True)
            suggestions = local_matches + non_local
            
            return jsonify({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Search API error: {e}")
        
    # Simple offline fallback for common Indian tickers
    fallback_stocks = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "INFY.NS", "name": "Infosys Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "TATASTEEL.NS", "name": "Tata Steel Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "ITC.NS", "name": "ITC Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE", "is_indian": True},
        {"symbol": "BHARTIRTAL.NS", "name": "Bharti Airtel Limited", "exchange": "NSE", "is_indian": True},
        {"symbol": "LTIM.NS", "name": "LTIMindtree Limited", "exchange": "NSE", "is_indian": True}
    ]
    query_upper = query.upper()
    suggestions = [s for s in fallback_stocks if query_upper in s['symbol'] or query_upper in s['name'].upper()]
    return jsonify({'suggestions': suggestions})

@app.route('/api/stock-details', methods=['GET'])
def get_stock_details():
    """Retrieve detailed stats and metrics for a stock."""
    raw_symbol = request.args.get('symbol')
    if not raw_symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400
        
    symbol = get_indian_ticker(raw_symbol)
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get recent daily close price & changes
        hist = ticker.history(period="5d")
        
        current_price = 0.0
        change = 0.0
        change_pct = 0.0
        prev_close = 0.0
        
        if not hist.empty:
            current_price = float(hist['Close'].iloc[-1])
            if len(hist) > 1:
                prev_close = float(hist['Close'].iloc[-2])
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
        
        # Construct response dictionary, fallback gracefully if yfinance info fails
        details = {
            'symbol': symbol,
            'name': info.get('longName') or info.get('shortName') or symbol,
            'current_price': current_price,
            'change': change,
            'change_percent': change_pct,
            'currency': info.get('currency', 'INR'),
            'market_cap': info.get('marketCap') or info.get('regularMarketVolume', 0) * current_price, # rough guess if none
            'pe_ratio': info.get('forwardPE') or info.get('trailingPE') or 'N/A',
            'volume': int(info.get('volume') or (hist['Volume'].iloc[-1] if not hist.empty else 0)),
            'avg_volume': info.get('averageVolume') or 'N/A',
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh') or (hist['Close'].max() if not hist.empty else 0.0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow') or (hist['Close'].min() if not hist.empty else 0.0),
            'open_price': info.get('open') or (hist['Open'].iloc[-1] if not hist.empty else 0.0),
            'day_high': info.get('dayHigh') or (hist['High'].iloc[-1] if not hist.empty else 0.0),
            'day_low': info.get('dayLow') or (hist['Low'].iloc[-1] if not hist.empty else 0.0),
            'dividend_yield': f"{info.get('dividendYield', 0.0) * 100:.2f}%" if info.get('dividendYield') else '0.00%',
            'sector': info.get('sector', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No description available.')
        }
        return jsonify(details)
    except Exception as e:
        logger.error(f"Error fetching details for {symbol}: {e}")
        # Return bare-bones info based on hist
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close > 0 else 0.0
                
                return jsonify({
                    'symbol': symbol,
                    'name': symbol,
                    'current_price': current_price,
                    'change': change,
                    'change_percent': change_pct,
                    'currency': 'INR',
                    'market_cap': 'N/A',
                    'pe_ratio': 'N/A',
                    'volume': int(hist['Volume'].iloc[-1]),
                    'avg_volume': 'N/A',
                    'fifty_two_week_high': float(hist['High'].max()),
                    'fifty_two_week_low': float(hist['Low'].min()),
                    'open_price': float(hist['Open'].iloc[-1]),
                    'day_high': float(hist['High'].iloc[-1]),
                    'day_low': float(hist['Low'].iloc[-1]),
                    'dividend_yield': '0.00%',
                    'sector': 'N/A',
                    'summary': 'Live details failed to fetch. Showing base metrics from chart history.'
                })
        except Exception:
            pass
        return jsonify({'error': f'Failed to retrieve details for symbol {symbol}: {str(e)}'}), 500

@app.route('/api/model-status', methods=['GET'])
def get_model_status_api():
    """Retrieve loading status of Kronos models."""
    global loaded_model_key, loaded_device, predictor
    
    with model_lock:
        return jsonify({
            'available': MODEL_AVAILABLE,
            'loaded': predictor is not None,
            'model_key': loaded_model_key,
            'device': loaded_device,
            'available_models': AVAILABLE_MODELS
        })

@app.route('/api/load-model', methods=['POST'])
def load_model_api():
    """Explicitly trigger loading of a Kronos model size and device."""
    global loaded_model_key, loaded_device, tokenizer, model, predictor
    
    if not MODEL_AVAILABLE:
        return jsonify({'error': 'Kronos model classes could not be imported'}), 400
        
    data = request.json or {}
    model_key = data.get('model_key', 'kronos-small')
    device = data.get('device', 'cpu')
    
    if model_key not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model key: {model_key}'}), 400
        
    config = AVAILABLE_MODELS[model_key]
    
    try:
        with model_lock:
            logger.info(f"Loading tokenizer: {config['tokenizer_id']} and model: {config['model_id']} on {device}...")
            tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_id'])
            model = Kronos.from_pretrained(config['model_id'])
            # Move model to selected device (e.g. cuda)
            model.to(device)
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=config['context_length'])
            
            loaded_model_key = model_key
            loaded_device = device
            logger.info("Model loaded successfully!")
            
        return jsonify({
            'success': True,
            'message': f"Successfully loaded {config['name']} on {device}.",
            'details': {
                'name': config['name'],
                'device': device,
                'params': config['params']
            }
        })
    except Exception as e:
        logger.error(f"Failed to load model {model_key}: {e}")
        return jsonify({'error': f"Failed to load model: {str(e)}"}), 500

@app.route('/api/historical-chart', methods=['GET'])
def get_historical_chart():
    """Retrieve historical-only raw stock price data for previewing."""
    raw_symbol = request.args.get('symbol')
    if not raw_symbol:
        return jsonify({'error': 'Symbol is required'}), 400
        
    symbol = get_indian_ticker(raw_symbol)
    df, err = fetch_stock_history(symbol, period="5y")
    if err:
        return jsonify({'error': f"Failed to fetch stock history: {err}"}), 400
        
    history_points = []
    for idx, row in df.iterrows():
        history_points.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
        
    return jsonify({
        'success': True,
        'symbol': symbol,
        'history': history_points
    })

def fetch_google_news(symbol):
    """Fetch and parse Google News RSS feed for a stock symbol."""
    symbol_prefix = symbol.split('.')[0]
    # We query with "stock" keyword to ensure financial context
    url = f"https://news.google.com/rss/search?q={symbol_prefix}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    articles = []
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        # Parse channel -> item elements
        for item in root.findall('.//item')[:15]:  # Limit to top 15 news items
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            source = item.find('source').text if item.find('source') is not None else "Google News"
            
            # Clean up title: Google News titles usually end with " - Source Name"
            clean_title = title
            if " - " in title:
                clean_title = " - ".join(title.split(" - ")[:-1])
                
            articles.append({
                'title': clean_title,
                'link': link,
                'pubDate': pub_date,
                'source': source,
                'summary': clean_title
            })
    except Exception as e:
        logger.error(f"Error fetching Google News for {symbol}: {e}")
    return articles

def get_aggregated_news(symbol):
    """Aggregate Yahoo Finance and Google News feeds for a stock symbol, deduplicating articles."""
    articles = []
    seen_titles = set()
    
    # 1. Fetch Google News RSS
    g_articles = fetch_google_news(symbol)
    for a in g_articles:
        norm_title = a['title'].lower().strip()
        title_key = norm_title[:30]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            articles.append({
                'title': a['title'],
                'link': a['link'],
                'source': a['source'],
                'summary': a['summary']
            })
            
    # 2. Fetch Yahoo Finance News
    try:
        ticker = yf.Ticker(symbol)
        yf_news = ticker.news
        if yf_news:
            for item in yf_news:
                title = item.get('title', '')
                if not title:
                    continue
                norm_title = title.lower().strip()
                title_key = norm_title[:30]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    articles.append({
                        'title': title,
                        'link': item.get('link', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'summary': title
                    })
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance news for {symbol}: {e}")
        
    return articles

def analyze_news_sentiment(articles):
    """Calculate sentiment score for each article and compute aggregate compound score."""
    if not articles:
        return 0.0, []
        
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"Could not load NLTK SentimentIntensityAnalyzer: {e}")
        return 0.0, []
        
    scored_articles = []
    total_compound = 0.0
    
    for a in articles:
        scores = sia.polarity_scores(a['title'])
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment_label = 'POSITIVE'
        elif compound <= -0.05:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
            
        scored_articles.append({
            'title': a['title'],
            'link': a['link'],
            'source': a['source'],
            'sentiment_score': compound,
            'sentiment_label': sentiment_label
        })
        total_compound += compound
        
    net_sentiment = total_compound / len(articles) if articles else 0.0
    return net_sentiment, scored_articles

@app.route('/api/predict-enhanced', methods=['POST'])
def run_enhanced_forecast():
    """Aggregate news, run sentiment NLP, execute Kronos base model, and apply sentiment drift overlay."""
    global loaded_model_key, loaded_device, tokenizer, model, predictor
    
    data = request.json or {}
    raw_symbol = data.get('symbol')
    if not raw_symbol:
        return jsonify({'error': 'Symbol is required'}), 400
        
    symbol = get_indian_ticker(raw_symbol)
    
    # Base configuration
    lookback = 400
    pred_len = int(data.get('pred_len', 120))
    pred_len = max(15, min(120, pred_len))
    
    temperature = float(data.get('temperature', 0.6))
    top_p = float(data.get('top_p', 0.9))
    sample_count = int(data.get('sample_count', 1))
    
    # 1. Load model if needed
    if predictor is None:
        if MODEL_AVAILABLE:
            try:
                default_key = 'kronos-small'
                default_device = 'cpu'
                config = AVAILABLE_MODELS[default_key]
                logger.info(f"Enhanced Endpoint: Loading default {default_key} on CPU...")
                
                with model_lock:
                    tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_id'])
                    model = Kronos.from_pretrained(config['model_id'])
                    predictor = KronosPredictor(model, tokenizer, device=default_device, max_context=config['context_length'])
                    loaded_model_key = default_key
                    loaded_device = default_device
            except Exception as e:
                logger.error(f"Failed to load default model: {e}")
                return jsonify({'error': f"Failed to load default model: {str(e)}"}), 500
        else:
            return jsonify({'error': 'Kronos model dependencies are missing.'}), 500
            
    # 2. Fetch history
    df, err = fetch_stock_history(symbol, period="5y")
    if err:
        return jsonify({'error': f"Failed to fetch stock history: {err}"}), 400
        
    if len(df) < lookback:
        return jsonify({'error': f"Ticker {symbol} only has {len(df)} days of history. Need {lookback} days."}), 400
        
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.iloc[-lookback:]["date"]
    
    last_date = df["date"].iloc[-1]
    y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
    
    # 3. Execute base Kronos model forecast
    try:
        x_ts_series = pd.Series(x_timestamp).reset_index(drop=True)
        y_ts_series = pd.Series(y_timestamp).reset_index(drop=True)
        
        pred_df_raw = predictor.predict(
            df=x_df.reset_index(drop=True),
            x_timestamp=x_ts_series,
            y_timestamp=y_ts_series,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count
        )
        
        # Apply circuits to base forecast
        last_close = float(x_df['close'].iloc[-1])
        pred_df_raw_limited = apply_price_limits(pred_df_raw.copy(), last_close, limit_rate=0.1)
        pred_df_raw_limited['date'] = y_timestamp.values
        
    except Exception as e:
        logger.error(f"Enhanced forecasting base model crash: {e}")
        return jsonify({'error': f"Inference pipeline failed: {str(e)}"}), 500
        
    # 4. Fetch and analyze live news sentiment
    logger.info(f"Aggregating live news and analyzing sentiment for {symbol}...")
    news_articles = get_aggregated_news(symbol)
    net_sentiment, scored_news = analyze_news_sentiment(news_articles)
    
    # 5. Apply Dynamic Momentum & Sentiment Drift (DMSD) Math
    r_drift = calculate_dmsd_drift(symbol, net_sentiment)
    pred_df_enhanced = pred_df_raw.copy().reset_index(drop=True)
    
    for t in range(len(pred_df_enhanced)):
        drift_factor = (1.0 + r_drift) ** (t + 1)
        
        cols = ['open', 'high', 'low', 'close']
        for col in cols:
            val = pred_df_enhanced.at[t, col]
            if pd.notna(val):
                pred_df_enhanced.at[t, col] = float(val * drift_factor)
                
    # Apply standard circuit limits to enhanced forecast
    pred_df_enhanced_limited = apply_price_limits(pred_df_enhanced, last_close, limit_rate=0.1)
    pred_df_enhanced_limited['date'] = y_timestamp.values
    
    # 6. Format historical, raw forecast, and enhanced forecast points
    history_points = []
    for idx, row in df.iterrows():
        history_points.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
        
    raw_forecast_points = []
    for idx, row in pred_df_raw_limited.iterrows():
        raw_forecast_points.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
        
    enhanced_forecast_points = []
    for idx, row in pred_df_enhanced_limited.iterrows():
        enhanced_forecast_points.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
        
    # Formulate trend signals
    raw_final_close = float(raw_forecast_points[-1]['close'])
    enhanced_final_close = float(enhanced_forecast_points[-1]['close'])
    
    raw_change_pct = ((raw_final_close - last_close) / last_close) * 100
    enhanced_change_pct = ((enhanced_final_close - last_close) / last_close) * 100
    
    # Detect signal based on enhanced forecast
    if enhanced_change_pct > 4.0:
        trend_signal = "BULLISH (UPWARD MOMENTUM)"
        signal_color = "#2e9f52"  # Sherlock Forest Green
        summary_desc = f"StockHolmes aggregates news sentiment and ML projections indicating strong upward momentum over the next {pred_len} trading days. Net news sentiment compound is {net_sentiment:+.2f}. Projected close: {enhanced_final_close:.2f} (+{enhanced_change_pct:.2f}%)."
    elif enhanced_change_pct < -4.0:
        trend_signal = "BEARISH (DOWNWARD DRIFT)"
        signal_color = "#d33c3c"  # Deep Oxblood Red
        summary_desc = f"StockHolmes aggregates news sentiment and ML projections indicating downward price momentum. Net news sentiment compound is {net_sentiment:+.2f}. Projected close: {enhanced_final_close:.2f} ({enhanced_change_pct:.2f}%)."
    else:
        trend_signal = "NEUTRAL (CONSOLIDATION)"
        signal_color = "#c5a059"  # StockHolmes Gold
        summary_desc = f"Projections indicate consolidative sideways drift. Net news sentiment compound is {net_sentiment:+.2f}. Projected close: {enhanced_final_close:.2f} ({enhanced_change_pct:+.2f}%)."
        
    return jsonify({
        'success': True,
        'symbol': symbol,
        'history': history_points,
        'forecast': raw_forecast_points,             # Retain raw forecast under 'forecast' for untouched frontend compatibility
        'forecast_enhanced': enhanced_forecast_points, # Save secondary forecast under 'forecast_enhanced'
        'net_sentiment': net_sentiment,
        'news_feed': scored_news,
        'analysis': {
            'trend_signal': trend_signal,
            'signal_color': signal_color,
            'summary': summary_desc,
            'change_percent': enhanced_change_pct,
            'last_price': last_close,
            'target_price': enhanced_final_close,
            'max_predicted': float(pred_df_enhanced_limited['high'].max()),
            'min_predicted': float(pred_df_enhanced_limited['low'].min()),
            'avg_predicted': float(pred_df_enhanced_limited['close'].mean()),
            
            # Save raw details for comparison audit
            'raw_target_price': raw_final_close,
            'raw_change_percent': raw_change_pct
        }
    })

@app.route('/api/predict', methods=['POST'])
def run_forecast():
    """Download historical data, run Kronos model prediction, and return raw history and forecast data."""
    global loaded_model_key, loaded_device, tokenizer, model, predictor
    
    data = request.json or {}
    raw_symbol = data.get('symbol')
    if not raw_symbol:
        return jsonify({'error': 'Symbol is required'}), 400
        
    symbol = get_indian_ticker(raw_symbol)
    
    # Prediction config (pred_len can be 15 to 120, default 120)
    lookback = 400
    pred_len = int(data.get('pred_len', 120))
    pred_len = max(15, min(120, pred_len))
    
    temperature = float(data.get('temperature', 0.6))
    top_p = float(data.get('top_p', 0.9))
    sample_count = int(data.get('sample_count', 1))
    
    # 1. Check if model is loaded, if not, load default kronos-small on cpu
    if predictor is None:
        if MODEL_AVAILABLE:
            try:
                # Load default
                default_key = 'kronos-small'
                default_device = 'cpu'
                config = AVAILABLE_MODELS[default_key]
                logger.info(f"No active model found. Loading default {default_key} on CPU...")
                
                with model_lock:
                    tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_id'])
                    model = Kronos.from_pretrained(config['model_id'])
                    predictor = KronosPredictor(model, tokenizer, device=default_device, max_context=config['context_length'])
                    loaded_model_key = default_key
                    loaded_device = default_device
            except Exception as e:
                logger.error(f"Failed to load default model: {e}")
                return jsonify({'error': f"Failed to load default predictor model: {str(e)}"}), 500
        else:
            return jsonify({'error': 'Kronos model dependencies are missing. Simulation fallback not active.'}), 500
            
    # 2. Fetch history (at least 400 daily data points needed)
    logger.info(f"Fetching history for {symbol} to run prediction...")
    df, err = fetch_stock_history(symbol, period="5y")
    if err:
        return jsonify({'error': f"Failed to fetch stock history: {err}"}), 400
        
    if len(df) < lookback:
        return jsonify({'error': f"Ticker {symbol} only has {len(df)} days of history. Need at least {lookback} trading days for Kronos prediction context."}), 400
        
    # Get inputs for the prediction
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.iloc[-lookback:]["date"]
    
    # Predict future date range (business days starting from next business day after last date)
    last_date = df["date"].iloc[-1]
    y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
    
    # 3. Run model forecasting
    try:
        logger.info(f"Running Kronos forecasting (device={loaded_device}, temp={temperature}, top_p={top_p}, steps={pred_len})...")
        
        # Avoid DatetimeIndex issue in predictor by converting to Series
        x_ts_series = pd.Series(x_timestamp).reset_index(drop=True)
        y_ts_series = pd.Series(y_timestamp).reset_index(drop=True)
        
        pred_df = predictor.predict(
            df=x_df.reset_index(drop=True),
            x_timestamp=x_ts_series,
            y_timestamp=y_ts_series,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count
        )
        
        # Apply circuits (circuit breaker mechanism typical of NSE/BSE: max 10% movement per day)
        last_close = float(x_df['close'].iloc[-1])
        pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)
        pred_df['date'] = y_timestamp.values
        
    except Exception as e:
        logger.error(f"Prediction engine crash on symbol {symbol}: {e}")
        return jsonify({'error': f"Prediction engine failed: {str(e)}"}), 500

    # 4. Formulate Trend Analysis Details
    predicted_close = pred_df['close'].values
    pred_pct_change = ((predicted_close[-1] - last_close) / last_close) * 100
    
    max_pred = float(pred_df['high'].max())
    min_pred = float(pred_df['low'].min())
    avg_pred = float(pred_df['close'].mean())
    
    # Detect signal
    if pred_pct_change > 4.0:
        trend_signal = "BULLISH (UPWARD MOMENTUM)"
        signal_color = "#2e9f52"  # Sherlock Forest Green
        summary_desc = f"The model predicts a strong upward momentum over the next {pred_len} trading days, with a projected close of {predicted_close[-1]:.2f} (a gain of {pred_pct_change:.2f}%)."
    elif pred_pct_change < -4.0:
        trend_signal = "BEARISH (DOWNWARD DRIFT)"
        signal_color = "#d33c3c"  # Deep Oxblood Red
        summary_desc = f"The model predicts a downward price trajectory, projecting a close of {predicted_close[-1]:.2f} (a decrease of {abs(pred_pct_change):.2f}%)."
    else:
        trend_signal = "NEUTRAL (CONSOLIDATION)"
        signal_color = "#c5a059"  # StockHolmes Gold
        summary_desc = f"The model forecasts sideways consolidation, with minor price fluctuations and a final return of {pred_pct_change:+.2f}%."
        
    # 5. Format historical and prediction data arrays
    history_points = []
    for idx, row in df.iterrows():
        history_points.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
        
    forecast_points = []
    for idx, row in pred_df.iterrows():
        forecast_points.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
        
    return jsonify({
        'success': True,
        'symbol': symbol,
        'history': history_points,
        'forecast': forecast_points,
        'analysis': {
            'trend_signal': trend_signal,
            'signal_color': signal_color,
            'summary': summary_desc,
            'change_percent': pred_pct_change,
            'last_price': last_close,
            'target_price': float(predicted_close[-1]),
            'max_predicted': max_pred,
            'min_predicted': min_pred,
            'avg_predicted': avg_pred
        }
    })

@app.route('/api/sync-to-google-sheet', methods=['POST'])
def sync_to_google_sheet():
    from google_sheets import sync_stock_data, get_sheets_service, extract_dates_from_headers, get_first_sheet_title
    
    data = request.json or {}
    raw_symbol = data.get('symbol')
    raw_spreadsheet_id = data.get('spreadsheet_id')
    enhanced = data.get('enhanced', False)
    
    if not raw_symbol or not raw_spreadsheet_id:
        return jsonify({'error': 'Symbol and Spreadsheet ID are required'}), 400
        
    # Extract Google Sheets ID if a full URL is provided
    spreadsheet_id = raw_spreadsheet_id.strip()
    url_match = re.search(r'/d/([a-zA-Z0-9-_]+)', spreadsheet_id)
    if url_match:
        spreadsheet_id = url_match.group(1)
        
    symbol = get_indian_ticker(raw_symbol)
    
    # 1. Verify service account exists
    service = get_sheets_service()
    if not service:
        return jsonify({
            'error': 'service_account_missing',
            'message': 'service_account.json not found in the project root directory. Please generate and save a Google Cloud Service Account key file to automate this spreadsheet.'
        }), 400
        
    # 2. Get sheet headers to find dates we need
    try:
        sheet_title = get_first_sheet_title(service, spreadsheet_id)
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=f"{sheet_title}!A1:Z2").execute()
        rows = result.get('values', [])
        if not rows or len(rows) < 2:
            return jsonify({'error': 'Invalid sheet structure: needs at least 2 header rows.'}), 400
            
        row1 = rows[0]
        dates_map = extract_dates_from_headers(row1)
        if not dates_map:
            return jsonify({'error': 'No dates found in Sheet Row 1 headers (e.g., 8/24/2026).'}), 400
            
    except Exception as e:
        return jsonify({'error': f"Failed to access Google Sheet: {str(e)}"}), 500
        
    # 3. Load historical data
    df, err = fetch_stock_history(symbol, period="5y")
    if err:
        return jsonify({'error': f"Failed to fetch stock history: {err}"}), 400
        
    # Find today's close price (last close in history)
    last_close = float(df['close'].iloc[-1])
    
    # 4. Generate predictions or parse client-provided predictions
    client_predictions = data.get('client_predictions')
    predictions = {}
    
    if client_predictions:
        # Use client-provided predictions (guarantees matching numbers and instant sync)
        for pt in client_predictions:
            try:
                dt_obj = datetime.datetime.strptime(pt['date'], "%Y-%m-%d")
                key = f"{dt_obj.month}/{dt_obj.day}/{dt_obj.year}"
                predictions[key] = float(pt['close'])
            except Exception as e:
                logger.warning(f"Failed to parse client prediction point {pt}: {e}")
    else:
        # Fallback: Generate predictions using the model
        max_days = 40
        global predictor, loaded_model_key, loaded_device, tokenizer, model
        if predictor is None:
            if MODEL_AVAILABLE:
                try:
                    default_key = 'kronos-small'
                    default_device = 'cpu'
                    config = AVAILABLE_MODELS[default_key]
                    with model_lock:
                        tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_id'])
                        model = Kronos.from_pretrained(config['model_id'])
                        predictor = KronosPredictor(model, tokenizer, device=default_device, max_context=config['context_length'])
                        loaded_model_key = default_key
                        loaded_device = default_device
                except Exception as e:
                    return jsonify({'error': f"Failed to load default predictor model: {str(e)}"}), 500
            else:
                return jsonify({'error': 'Model dependencies are missing.'}), 500
                
        lookback = 400
        x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df.iloc[-lookback:]["date"]
        last_date = df["date"].iloc[-1]
        y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max_days)
        
        try:
            x_ts_series = pd.Series(x_timestamp).reset_index(drop=True)
            y_ts_series = pd.Series(y_timestamp).reset_index(drop=True)
            
            pred_df = predictor.predict(
                df=x_df.reset_index(drop=True),
                x_timestamp=x_ts_series,
                y_timestamp=y_ts_series,
                pred_len=max_days,
                T=0.6,
                top_p=0.9,
                sample_count=1
            )
            pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)
            pred_df['date'] = y_timestamp.values
            
            if enhanced:
                # Run enhanced compound drift calculation using DMSD
                articles = get_aggregated_news(symbol)
                net_sentiment, scored_news = analyze_news_sentiment(articles)
                r_drift = calculate_dmsd_drift(symbol, net_sentiment)
                pred_df_enhanced = pred_df.copy().reset_index(drop=True)
                for t in range(len(pred_df_enhanced)):
                    factor = (1.0 + r_drift) ** (t + 1)
                    for col in ['open', 'high', 'low', 'close']:
                        pred_df_enhanced.at[t, col] = pred_df.at[t, col] * factor
                pred_df = pred_df_enhanced
                
            # Map predictions by formatted date string
            pred_df['date_str'] = pred_df['date'].apply(lambda x: f"{x.month}/{x.day}/{x.year}")
            for idx, row in pred_df.iterrows():
                predictions[row['date_str']] = float(row['close'])
                
        except Exception as e:
            return jsonify({'error': f"Forecasting failed: {str(e)}"}), 500
        
    # Map actuals from history
    df['date_str'] = df['date'].apply(lambda x: f"{x.month}/{x.day}/{x.year}")
    actuals = {}
    for idx, row in df.iterrows():
        actuals[row['date_str']] = float(row['close'])
        
    # Query yfinance for historical values not present in dataframe but requested
    today_dt = datetime.datetime.now()
    for date_str in dates_map.values():
        if date_str not in actuals:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%m/%d/%Y")
                if date_obj < today_dt:
                    start_date = date_obj.strftime("%Y-%m-%d")
                    end_date = (date_obj + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    ticker_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not ticker_data.empty:
                        actuals[date_str] = float(ticker_data['Close'].iloc[0])
            except Exception as ex:
                logger.warning(f"Failed to fetch actual price for date {date_str}: {ex}")
                
    # 5. Call google sheets sync function
    success, msg = sync_stock_data(spreadsheet_id, symbol, last_close, predictions, actuals)
    if not success:
        if msg == "service_account_missing":
            return jsonify({
                'error': 'service_account_missing',
                'message': 'service_account.json not found in the project root directory. Please generate and save a Google Cloud Service Account key file to automate this spreadsheet.'
            }), 400
        return jsonify({'error': f"Sync failed: {msg}"}), 500
        
    return jsonify({
        'success': True,
        'message': msg
    })

if __name__ == '__main__':
    # Create templates/static folders if they do not exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'js'), exist_ok=True)
    
    logger.info("Starting StockHolmes Quant Backend on port 7070...")
    app.run(debug=True, host='0.0.0.0', port=7070)
