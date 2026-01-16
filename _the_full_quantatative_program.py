#!/usr/bin/env python3
"""
Quantitative Trading System Improved
Institutional-grade analysis with FRED, Alpha Vantage, advanced ML, and AI consultation.

Requirements:
pip install yfinance pandas numpy scikit-learn xgboost lightgbm catboost colorama ta requests python-dotenv scipy statsmodels fredapi alpha_vantage tensorflow keras

Setup (Windows PowerShell):
    $env:GEMINI_API_KEY="your_key"
    $env:FRED_API_KEY="your_key"
    $env:ALPHA_VANTAGE_KEY="your_key"

Usage:
    python _the_full_quantatative_program.py AAPL MSFT GOOGL
"""

import sys
import os
import warnings
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from colorama import Fore, Style, init
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
import ta
import requests
from dotenv import load_dotenv
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import jarque_bera
from fredapi import Fred
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

warnings.filterwarnings('ignore')
init(autoreset=True)
load_dotenv()

class GoldmanSachsQuantSystem:
    """Goldman Sachs-level quantitative trading system with institutional data sources."""
    
    def __init__(self, tickers, initial_capital=1000000):
        self.tickers = [t.upper() for t in tickers]
        self.initial_capital = initial_capital
        self.data = {}
        self.features = {}
        self.fred_data = {}
        self.alpha_vantage_data = {}
        self.company_info = {}
        self.fundamental_data = {}
        self.ml_results = {}
        self.risk_metrics = {}
        self.portfolio_allocation = {}
        self.strategy_recommendations = {}
        self.correlation_matrix = None
        self.covariance_matrix = None
        self.macro_regime = {}
        
        # API Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        
        if not self.gemini_api_key:
            print(f"{Fore.RED}⚠️  GEMINI_API_KEY not found!{Style.RESET_ALL}")
        if not self.fred_api_key:
            print(f"{Fore.YELLOW}⚠️  FRED_API_KEY not found - macro data limited{Style.RESET_ALL}")
        if not self.alpha_vantage_key:
            print(f"{Fore.YELLOW}⚠️  ALPHA_VANTAGE_KEY not found - fundamental data limited{Style.RESET_ALL}")
        
        self.gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        
        # Initialize APIs
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        if self.alpha_vantage_key:
            self.av_fundamental = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
            self.av_timeseries = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.av_technical = TechIndicators(key=self.alpha_vantage_key, output_format='pandas')
        
    def print_header(self, text, style="=", color=Fore.CYAN):
        width = 100
        print(f"\n{color}{style * width}")
        print(f"{Fore.YELLOW}{text.center(width)}")
        print(f"{color}{style * width}")
    
    def print_status(self, text, status="INFO"):
        colors = {"INFO": Fore.CYAN, "SUCCESS": Fore.GREEN, "WARNING": Fore.YELLOW, "ERROR": Fore.RED}
        symbols = {"INFO": "ℹ", "SUCCESS": "✅", "WARNING": "⚠", "ERROR": "❌"}
        print(f"{colors[status]}{symbols[status]} {text}{Style.RESET_ALL}")
    
    def fetch_fred_macro_data(self):
        """Fetch comprehensive macroeconomic data from FRED."""
        if not self.fred_api_key:
            self.print_status("FRED API key not configured - skipping macro data", "WARNING")
            return
        
        self.print_status("Fetching macroeconomic data from FRED...", "INFO")
        
        fred_series = {
            'GDP': 'GDP',                           # GDP
            'UNRATE': 'Unemployment Rate',          # Unemployment
            'CPIAUCSL': 'CPI',                      # Inflation
            'FEDFUNDS': 'Fed Funds Rate',           # Interest Rate
            'DGS10': '10Y Treasury',                # 10-Year Treasury
            'DGS2': '2Y Treasury',                  # 2-Year Treasury
            'T10Y2Y': 'Yield Curve',                # Yield Spread
            'DEXUSEU': 'USD/EUR',                   # Currency
            'DTWEXBGS': 'Dollar Index',             # Trade Weighted Dollar
            'VIXCLS': 'VIX',                        # Volatility
            'UMCSENT': 'Consumer Sentiment',        # Sentiment
            'INDPRO': 'Industrial Production',      # Production
            'HOUST': 'Housing Starts',              # Housing
            'RSXFS': 'Retail Sales',                # Retail
            'PPIACO': 'PPI',                        # Producer Prices
            'M2SL': 'M2 Money Supply',              # Money Supply
            'BOGMBASE': 'Monetary Base',            # Fed Balance Sheet
            'TERMCBPER24NS': 'Credit Spread',       # Credit Risk
            'BAMLH0A0HYM2': 'High Yield Spread',    # Junk Bond Spread
            'WALCL': 'Fed Assets',                  # Fed Total Assets
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3650)  # 10 years
        
        for series_id, description in fred_series.items():
            try:
                data = self.fred.get_series(series_id, start_date=start_date, end_date=end_date)
                if not data.empty:
                    self.fred_data[series_id] = data
                    self.print_status(f"✓ {description}: {len(data)} observations", "SUCCESS")
                    time.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.print_status(f"✗ {description} ({series_id}): {str(e)}", "WARNING")
        
        # Calculate macro regime
        self.calculate_macro_regime()
    
    def calculate_macro_regime(self):
        """Determine current macroeconomic regime."""
        if not self.fred_data:
            return
        
        regime = {
            'growth': 'Unknown',
            'inflation': 'Unknown',
            'monetary_policy': 'Unknown',
            'risk_sentiment': 'Unknown'
        }
        
        # Growth regime
        if 'GDP' in self.fred_data and len(self.fred_data['GDP']) > 4:
            gdp_growth = self.fred_data['GDP'].pct_change(4).iloc[-1]
            if gdp_growth > 0.03:
                regime['growth'] = 'Expansion'
            elif gdp_growth > 0:
                regime['growth'] = 'Moderate Growth'
            else:
                regime['growth'] = 'Contraction'
        
        # Inflation regime
        if 'CPIAUCSL' in self.fred_data and len(self.fred_data['CPIAUCSL']) > 12:
            cpi_growth = self.fred_data['CPIAUCSL'].pct_change(12).iloc[-1]
            if cpi_growth > 0.04:
                regime['inflation'] = 'High Inflation'
            elif cpi_growth > 0.02:
                regime['inflation'] = 'Moderate Inflation'
            else:
                regime['inflation'] = 'Low Inflation'
        
        # Monetary policy
        if 'FEDFUNDS' in self.fred_data and len(self.fred_data['FEDFUNDS']) > 12:
            fed_funds = self.fred_data['FEDFUNDS'].iloc[-1]
            fed_trend = self.fred_data['FEDFUNDS'].iloc[-1] - self.fred_data['FEDFUNDS'].iloc[-12]
            if fed_trend > 0.5:
                regime['monetary_policy'] = 'Tightening'
            elif fed_trend < -0.5:
                regime['monetary_policy'] = 'Easing'
            else:
                regime['monetary_policy'] = 'Neutral'
        
        # Risk sentiment
        if 'VIXCLS' in self.fred_data:
            vix = self.fred_data['VIXCLS'].iloc[-1]
            if vix > 30:
                regime['risk_sentiment'] = 'High Fear'
            elif vix > 20:
                regime['risk_sentiment'] = 'Elevated Fear'
            else:
                regime['risk_sentiment'] = 'Complacent'
        
        self.macro_regime = regime
        self.print_status(f"Macro Regime: Growth={regime['growth']}, Inflation={regime['inflation']}, Policy={regime['monetary_policy']}, Sentiment={regime['risk_sentiment']}", "INFO")
    
    def fetch_alpha_vantage_data(self):
        """Fetch fundamental and technical data from Alpha Vantage."""
        if not self.alpha_vantage_key:
            self.print_status("Alpha Vantage API key not configured - skipping", "WARNING")
            return
        
        self.print_status("Fetching data from Alpha Vantage...", "INFO")
        
        for ticker in self.tickers:
            try:
                # Company Overview
                overview, _ = self.av_fundamental.get_company_overview(ticker)
                if not overview.empty:
                    self.alpha_vantage_data[ticker] = {
                        'overview': overview
                    }
                    self.print_status(f"✓ {ticker}: Company overview fetched", "SUCCESS")
                
                time.sleep(12)  # Alpha Vantage rate limit: 5 calls/min
                
                # Income Statement
                try:
                    income_statement, _ = self.av_fundamental.get_income_statement_annual(ticker)
                    if not income_statement.empty:
                        self.alpha_vantage_data[ticker]['income_statement'] = income_statement
                        self.print_status(f"✓ {ticker}: Income statement fetched", "SUCCESS")
                    time.sleep(12)
                except:
                    pass
                
                # Balance Sheet
                try:
                    balance_sheet, _ = self.av_fundamental.get_balance_sheet_annual(ticker)
                    if not balance_sheet.empty:
                        self.alpha_vantage_data[ticker]['balance_sheet'] = balance_sheet
                        self.print_status(f"✓ {ticker}: Balance sheet fetched", "SUCCESS")
                    time.sleep(12)
                except:
                    pass
                
            except Exception as e:
                self.print_status(f"✗ {ticker} Alpha Vantage: {str(e)}", "WARNING")
    
    def fetch_market_data(self, period="10y"):
        """Fetch comprehensive market data."""
        self.print_status("Fetching market data from Yahoo Finance...", "INFO")
        
        benchmark_symbols = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100', 
            '^VIX': 'VIX',
            'TLT': '20Y Treasury',
            'GLD': 'Gold',
            'DXY': 'Dollar Index',
            'HYG': 'High Yield',
            'LQD': 'Investment Grade',
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLB': 'Materials',
        }
        
        all_symbols = self.tickers + list(benchmark_symbols.keys())
        
        for symbol in all_symbols:
            try:
                ticker_obj = yf.Ticker(symbol)
                data = ticker_obj.history(period=period, interval="1d")
                
                if not data.empty and len(data) > 252:
                    if symbol in self.tickers:
                        self.data[symbol] = data
                        try:
                            info = ticker_obj.info
                            self.company_info[symbol] = info
                            self.fundamental_data[symbol] = self.extract_fundamentals(info)
                        except:
                            self.company_info[symbol] = {"shortName": symbol}
                            self.fundamental_data[symbol] = {}
                    else:
                        self.data[symbol] = data
                    
                    self.print_status(f"✓ {symbol}: {len(data)} days", "SUCCESS")
            except Exception as e:
                self.print_status(f"✗ {symbol}: {str(e)}", "WARNING")
        
        return len([t for t in self.tickers if t in self.data]) > 0
    
    def extract_fundamentals(self, info):
        """Extract fundamental metrics."""
        return {
            'marketCap': info.get('marketCap'),
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'priceToBook': info.get('priceToBook'),
            'priceToSales': info.get('priceToSalesTrailing12Months'),
            'pegRatio': info.get('pegRatio'),
            'dividendYield': info.get('dividendYield'),
            'profitMargins': info.get('profitMargins'),
            'operatingMargins': info.get('operatingMargins'),
            'returnOnEquity': info.get('returnOnEquity'),
            'returnOnAssets': info.get('returnOnAssets'),
            'debtToEquity': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'beta': info.get('beta'),
            'earningsGrowth': info.get('earningsGrowth'),
            'revenueGrowth': info.get('revenueGrowth'),
        }
    
    def calculate_advanced_features(self):
        """Calculate 200+ advanced features."""
        self.print_status("Engineering advanced features...", "INFO")
        
        for ticker in self.tickers:
            if ticker not in self.data:
                continue
            
            df = self.data[ticker].copy()
            
            # Price features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
            df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
            
            # Multi-timeframe momentum
            for period in [1, 2, 3, 5, 10, 15, 21, 30, 42, 63, 90, 126, 189, 252]:
                df[f'Mom_{period}'] = df['Close'].pct_change(period)
                df[f'PriceRatio_{period}'] = df['Close'] / df['Close'].shift(period)
            
            # Moving averages
            for ma in [5, 10, 20, 30, 50, 100, 150, 200]:
                df[f'SMA_{ma}'] = df['Close'].rolling(ma).mean()
                df[f'EMA_{ma}'] = df['Close'].ewm(span=ma).mean()
                df[f'Price_SMA_{ma}'] = df['Close'] / df[f'SMA_{ma}']
                df[f'Price_EMA_{ma}'] = df['Close'] / df[f'EMA_{ma}']
            
            # MA crossovers
            df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)
            
            # Volatility
            for window in [5, 10, 21, 42, 63, 126, 252]:
                df[f'Vol_{window}'] = df['Returns'].rolling(window).std() * np.sqrt(252)
                df[f'VolRatio_{window}'] = df[f'Vol_{window}'] / df[f'Vol_{window}'].shift(21)
            
            # Parkinson volatility
            df['Parkinson_Vol'] = np.sqrt((1/(4*21*np.log(2))) * ((np.log(df['High']/df['Low'])**2).rolling(21).sum())) * np.sqrt(252)
            
            # Technical indicators
            df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
            df['RSI_30'] = ta.momentum.RSIIndicator(df['Close'], 30).rsi()
            df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
            df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
            df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], 20, 2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
            
            # ADX
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
            # ATR
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['ATR_Pct'] = df['ATR'] / df['Close']
            
            # Volume
            df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
            
            # VWAP
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df['Price_VWAP'] = df['Close'] / df['VWAP']
            
            # Statistical features
            for window in [10, 20, 50]:
                df[f'ZScore_{window}'] = (df['Close'] - df['Close'].rolling(window).mean()) / df['Close'].rolling(window).std()
                df[f'Skew_{window}'] = df['Returns'].rolling(window).skew()
                df[f'Kurt_{window}'] = df['Returns'].rolling(window).kurt()
            
            # Benchmark correlations
            if 'SPY' in self.data:
                spy_returns = self.data['SPY']['Close'].pct_change().reindex(df.index)
                for window in [21, 63, 126, 252]:
                    df[f'SPY_Corr_{window}'] = df['Returns'].rolling(window).corr(spy_returns)
                    df[f'SPY_Beta_{window}'] = df['Returns'].rolling(window).cov(spy_returns) / spy_returns.rolling(window).var()
            
            # Macro features from FRED
            if self.fred_data:
                for series_id, series_data in self.fred_data.items():
                    aligned_data = series_data.reindex(df.index, method='ffill')
                    df[f'FRED_{series_id}'] = aligned_data
                    df[f'FRED_{series_id}_Change'] = aligned_data.pct_change()
            
            # Clean data
            df = df.loc[:, df.isnull().mean() < 0.3]
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            self.features[ticker] = df
            
            feature_count = len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])
            self.print_status(f"✓ {ticker}: {feature_count} features engineered", "SUCCESS")
    
    def calculate_risk_metrics(self):
        """Calculate institutional-grade risk metrics."""
        self.print_status("Calculating risk metrics...", "INFO")
        
        for ticker in self.tickers:
            if ticker not in self.features:
                continue
            
            df = self.features[ticker]
            returns = df['Returns'].dropna()
            
            if len(returns) < 252:
                continue
            
            mean_ret = returns.mean() * 252
            std_ret = returns.std() * np.sqrt(252)
            
            # VaR & CVaR
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Drawdown
            cum_ret = (1 + returns).cumprod()
            running_max = cum_ret.expanding().max()
            drawdown = (cum_ret - running_max) / running_max
            max_dd = drawdown.min()
            
            # Ratios
            sharpe = (mean_ret - 0.02) / std_ret if std_ret > 0 else 0
            downside = returns[returns < 0].std() * np.sqrt(252)
            sortino = (mean_ret - 0.02) / downside if downside > 0 else 0
            calmar = mean_ret / abs(max_dd) if max_dd != 0 else 0
            
            # Win rate
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean()
            avg_loss = returns[returns < 0].mean()
            profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() else 0
            
            # Kelly
            if avg_loss != 0:
                kelly = win_rate - ((1 - win_rate) / abs(avg_loss / avg_win))
            else:
                kelly = 0
            
            # Beta & Alpha
            if 'SPY' in self.data:
                spy_ret = self.data['SPY']['Close'].pct_change().dropna()
                common = returns.index.intersection(spy_ret.index)
                if len(common) > 252:
                    beta = returns.loc[common].cov(spy_ret.loc[common]) / spy_ret.loc[common].var()
                    alpha = mean_ret - (0.02 + beta * (spy_ret.mean() * 252 - 0.02))
                else:
                    beta = alpha = np.nan
            else:
                beta = alpha = np.nan
            
            self.risk_metrics[ticker] = {
                'mean_return': mean_ret,
                'volatility': std_ret,
                'sharpe': sharpe,
                'sortino': sortino,
                'calmar': calmar,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'kelly': kelly,
                'beta': beta,
                'alpha': alpha,
            }
            
            self.print_status(f"✓ {ticker}: Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}", "SUCCESS")
    
    def train_ensemble_models(self):
        """Train institutional-grade ensemble models."""
        self.print_status("Training ML ensemble...", "INFO")
        
        for ticker in self.tickers:
            if ticker not in self.features:
                continue
            
            df = self.features[ticker].copy()
            
            # Prepare features
            exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns']
            feature_cols = [c for c in df.columns if c not in exclude]
            X = df[feature_cols].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
            X = X.dropna()
            
            if len(X) < 500:
                continue
            
            # Multiple horizons
            horizons = {
                '1d': 1,
                '3d': 3,
                '5d': 5,
                '10d': 10,
                '21d': 21
            }
            
            ticker_results = {}
            
            for horizon_name, days in horizons.items():
                # Create target
                y = (df['Close'].shift(-days) > df['Close']).astype(int)
                
                # Align
                common = X.index.intersection(y.index)
                X_aligned = X.loc[common]
                y_aligned = y.loc[common].dropna()
                
                common = X_aligned.index.intersection(y_aligned.index)
                X_final = X_aligned.loc[common]
                y_final = y_aligned.loc[common]
                
                if len(X_final) < 500:
                    continue
                
                # Feature selection
                if len(X_final.columns) > 80:
                    selector = SelectKBest(mutual_info_classif, k=80)
                    X_selected = selector.fit_transform(X_final, y_final)
                    selected_features = X_final.columns[selector.get_support()]
                    X_final = X_final[selected_features]
                
                # Scale
                scaler = RobustScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X_final),
                    index=X_final.index,
                    columns=X_final.columns
                )
                
                # Train models
                models = {
                    'XGBoost': xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        eval_metric='logloss'
                    ),
                    'LightGBM': lgb.LGBMClassifier(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.03,
                        random_state=42,
                        verbose=-1
                    ),
                    'GradientBoosting': GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42
                    ),
                }
                
                if CATBOOST_AVAILABLE:
                    models['CatBoost'] = cb.CatBoostClassifier(
                        iterations=300,
                        depth=8,
                        learning_rate=0.03,
                        random_state=42,
                        verbose=False
                    )
                
                # Train and evaluate
                tscv = TimeSeriesSplit(n_splits=5)
                best_model = None
                best_score = 0
                
                for name, model in models.items():
                    try:
                        scores = cross_val_score(model, X_scaled, y_final, cv=tscv, scoring='accuracy')
                        avg_score = scores.mean()
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_model = model
                            best_name = name
                    except Exception as e:
                        self.print_status(f"Model {name} failed: {e}", "WARNING")
                
                if best_model:
                    best_model.fit(X_scaled, y_final)
                    pred = best_model.predict(X_scaled.iloc[-1:])
                    proba = best_model.predict_proba(X_scaled.iloc[-1:])
                    
                    ticker_results[horizon_name] = {
                        'model': best_model,
                        'model_name': best_name,
                        'accuracy': best_score,
                        'prediction': pred[0],
                        'probability': proba[0],
                        'scaler': scaler
                    }
                    
                    direction = "📈 BULLISH" if pred[0] == 1 else "📉 BEARISH"
                    confidence = proba[0][pred[0]] * 100
                    self.print_status(f"{ticker} {horizon_name}: {direction} ({confidence:.1f}%, Acc: {best_score:.3f})", "SUCCESS")
            
            self.ml_results[ticker] = ticker_results
    
    def calculate_position_sizing(self):
        """Calculate optimal position sizing."""
        self.print_status("Calculating position sizing...", "INFO")
        
        for ticker in self.tickers:
            if ticker not in self.risk_metrics:
                continue
            
            rm = self.risk_metrics[ticker]
            current_price = self.data[ticker]['Close'].iloc[-1]
            
            # Kelly sizing
            kelly_full = rm['kelly']
            kelly_half = kelly_full * 0.5
            
            # Volatility sizing
            target_vol = 0.15
            vol_size = target_vol / rm['volatility'] if rm['volatility'] > 0 else 0.1
            
            # Risk-based sizing (2% risk per trade)
            max_dd = abs(rm['max_drawdown'])
            risk_size = 0.02 / max_dd if max_dd > 0 else 0.1
            
            # ML confidence sizing
            ml_confidences = []
            if ticker in self.ml_results:
                for horizon, result in self.ml_results[ticker].items():
                    conf = result['probability'][result['prediction']]
                    ml_confidences.append(conf)
            avg_confidence = np.mean(ml_confidences) if ml_confidences else 0.5
            ml_size = avg_confidence * 0.2
            
            # Recommended size
            recommended = min(kelly_half, vol_size, risk_size, ml_size, 0.25)
            recommended = max(recommended, 0)  # No shorting
            
            position_dollars = self.initial_capital * recommended
            shares = int(position_dollars / current_price)
            
            # Stop loss & targets
            atr = self.features[ticker]['ATR'].iloc[-1]
            stop_loss = current_price - (2 * atr)
            
            rr_ratio = rm['profit_factor'] if rm['profit_factor'] > 0 else 2
            target_1 = current_price * (1 + rr_ratio * 0.01)
            target_2 = current_price * (1 + rr_ratio * 0.02)
            target_3 = current_price * (1 + rr_ratio * 0.03)
            
            self.strategy_recommendations[ticker] = {
                'recommended_size': recommended,
                'position_dollars': position_dollars,
                'shares': shares,
                'stop_loss': stop_loss,
                'target_1': target_1,
                'target_2': target_2,
                'target_3': target_3,
                'kelly_full': kelly_full,
                'kelly_half': kelly_half,
            }
            
            self.print_status(f"✓ {ticker}: {recommended:.1%} (${position_dollars:,.0f}, {shares} shares)", "SUCCESS")
    
    def calculate_portfolio_optimization(self):
        """Modern Portfolio Theory optimization."""
        if len(self.tickers) < 2:
            return
        
        self.print_status("Optimizing portfolio allocation...", "INFO")
        
        returns_data = {}
        for ticker in self.tickers:
            if ticker in self.features:
                returns_data[ticker] = self.features[ticker]['Returns'].dropna()
        
        if len(returns_data) < 2:
            return
        
        returns_df = pd.DataFrame(returns_data).dropna()
        if len(returns_df) < 252:
            return
        
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        self.correlation_matrix = returns_df.corr()
        
        num_assets = len(self.tickers)
        
        def portfolio_return(weights):
            return np.sum(mean_returns * weights)
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def neg_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_vol(weights)
            return -(ret - 0.02) / vol
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial = np.array([1/num_assets] * num_assets)
        
        # Max Sharpe
        result_sharpe = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        max_sharpe_weights = result_sharpe.x
        
        # Min Volatility
        result_minvol = minimize(portfolio_vol, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        min_vol_weights = result_minvol.x
        
        self.portfolio_allocation = {
            'max_sharpe': {
                'weights': dict(zip(self.tickers, max_sharpe_weights)),
                'expected_return': portfolio_return(max_sharpe_weights),
                'volatility': portfolio_vol(max_sharpe_weights),
                'sharpe': -neg_sharpe(max_sharpe_weights)
            },
            'min_volatility': {
                'weights': dict(zip(self.tickers, min_vol_weights)),
                'expected_return': portfolio_return(min_vol_weights),
                'volatility': portfolio_vol(min_vol_weights),
                'sharpe': (portfolio_return(min_vol_weights) - 0.02) / portfolio_vol(min_vol_weights)
            }
        }
        
        self.print_status("✓ Portfolio optimization complete", "SUCCESS")
    
    def query_gemini_fast(self, prompt):
        """Fast, witty AI responses."""
        if not self.gemini_api_key:
            return "⚠️  Gemini API key not configured"
        
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.8,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(self.gemini_api_url, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and data['candidates']:
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    return self.format_ai_response(text)
            
            return f"⚠️  API Error: {response.status_code}"
        
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def format_ai_response(self, text):
        """Format AI response with colors and structure."""
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted.append("")
                continue
            
            # Headers
            if line.startswith('#'):
                formatted.append(f"\n{Fore.YELLOW}{'='*60}")
                formatted.append(f"{line.replace('#', '').strip()}")
                formatted.append(f"{'='*60}{Style.RESET_ALL}")
            # Bullet points
            elif line.startswith('- ') or line.startswith('• '):
                formatted.append(f"{Fore.CYAN}  • {line[2:]}{Style.RESET_ALL}")
            # Numbers
            elif line[0].isdigit() and '. ' in line:
                formatted.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
            # Bold (simulated)
            elif '**' in line:
                line = line.replace('**', f'{Fore.YELLOW}').replace('**', Style.RESET_ALL)
                formatted.append(line)
            else:
                formatted.append(line)
        
        return '\n'.join(formatted)
    
    def ai_chat_interface(self):
        """Fast, witty AI chat about your portfolio."""
        self.print_header("💬 AI ANALYST CHAT", color=Fore.MAGENTA)
        
        print(f"{Fore.CYAN}Chat with your AI analyst. Type 'exit' to quit, 'summary' for data overview.{Style.RESET_ALL}\n")
        
        context = self.prepare_context()
        
        while True:
            try:
                user_input = input(f"{Fore.GREEN}You → {Style.RESET_ALL}").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print(f"{Fore.YELLOW}👋 Closing chat...{Style.RESET_ALL}")
                    break
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'summary':
                    self.display_quick_summary()
                    continue
                
                prompt = f"""You are a witty, sharp Goldman Sachs quantitative analyst. Be concise, insightful, and use emojis sparingly.

CURRENT DATA:
{context}

MACRO REGIME:
{json.dumps(self.macro_regime, indent=2)}

USER: {user_input}

Respond in 3-5 sentences max. Be direct and actionable."""
                
                print(f"\n{Fore.YELLOW}AI → {Style.RESET_ALL}", end="", flush=True)
                response = self.query_gemini_fast(prompt)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Chat interrupted.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")
    
    def prepare_context(self):
        """Prepare concise context for AI."""
        context = []
        
        for ticker in self.tickers:
            ticker_info = [f"\n{ticker}:"]
            
            if ticker in self.data:
                price = self.data[ticker]['Close'].iloc[-1]
                ticker_info.append(f"  Price: ${price:.2f}")
            
            if ticker in self.risk_metrics:
                rm = self.risk_metrics[ticker]
                ticker_info.append(f"  Sharpe: {rm['sharpe']:.2f}")
                ticker_info.append(f"  MaxDD: {rm['max_drawdown']:.2%}")
            
            if ticker in self.ml_results:
                for horizon, result in self.ml_results[ticker].items():
                    direction = "UP" if result['prediction'] == 1 else "DOWN"
                    conf = result['probability'][result['prediction']] * 100
                    ticker_info.append(f"  ML {horizon}: {direction} ({conf:.0f}%)")
            
            if ticker in self.strategy_recommendations:
                sr = self.strategy_recommendations[ticker]
                ticker_info.append(f"  Position: {sr['recommended_size']:.1%}")
            
            context.extend(ticker_info)
        
        return '\n'.join(context)
    
    def display_quick_summary(self):
        """Quick data summary."""
        print(f"\n{Fore.YELLOW}{'='*60}")
        print("PORTFOLIO SNAPSHOT")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        for ticker in self.tickers:
            print(f"{Fore.CYAN}{ticker}:{Style.RESET_ALL}")
            
            if ticker in self.data:
                price = self.data[ticker]['Close'].iloc[-1]
                print(f"  💵 Price: ${price:.2f}")
            
            if ticker in self.risk_metrics:
                rm = self.risk_metrics[ticker]
                sharpe_color = Fore.GREEN if rm['sharpe'] > 1 else Fore.YELLOW
                print(f"  📊 Sharpe: {sharpe_color}{rm['sharpe']:.2f}{Style.RESET_ALL}")
                print(f"  📉 MaxDD: {Fore.RED}{rm['max_drawdown']:.2%}{Style.RESET_ALL}")
            
            if ticker in self.ml_results:
                print(f"  🤖 ML Predictions:")
                for horizon, result in self.ml_results[ticker].items():
                    direction = "📈" if result['prediction'] == 1 else "📉"
                    conf = result['probability'][result['prediction']] * 100
                    print(f"     {horizon}: {direction} ({conf:.0f}%)")
            
            if ticker in self.strategy_recommendations:
                sr = self.strategy_recommendations[ticker]
                print(f"  💼 Position: {Fore.GREEN}{sr['recommended_size']:.1%}{Style.RESET_ALL} (${sr['position_dollars']:,.0f})")
            
            print()
    
    def display_terminal_analysis(self):
        """Comprehensive terminal display."""
        self.print_header("📊 IMPROVED QUANT ANALYSIS", color=Fore.CYAN)
        
        # Macro overview
        if self.macro_regime:
            print(f"\n{Fore.YELLOW}🌍 MACRO REGIME{Style.RESET_ALL}")
            print(f"  Growth: {Fore.CYAN}{self.macro_regime.get('growth', 'Unknown')}{Style.RESET_ALL}")
            print(f"  Inflation: {Fore.CYAN}{self.macro_regime.get('inflation', 'Unknown')}{Style.RESET_ALL}")
            print(f"  Policy: {Fore.CYAN}{self.macro_regime.get('monetary_policy', 'Unknown')}{Style.RESET_ALL}")
            print(f"  Sentiment: {Fore.CYAN}{self.macro_regime.get('risk_sentiment', 'Unknown')}{Style.RESET_ALL}")
        
        # Individual stocks
        for ticker in self.tickers:
            print(f"\n{Fore.YELLOW}{'='*80}")
            print(f"{ticker} ANALYSIS")
            print(f"{'='*80}{Style.RESET_ALL}")
            
            if ticker not in self.data:
                print(f"{Fore.RED}No data available{Style.RESET_ALL}")
                continue
            
            # Current price
            current = self.data[ticker]['Close'].iloc[-1]
            prev = self.data[ticker]['Close'].iloc[-2]
            change = (current - prev) / prev
            
            change_color = Fore.GREEN if change > 0 else Fore.RED
            print(f"\n💵 Price: ${current:.2f} ({change_color}{change:+.2%}{Style.RESET_ALL})")
            
            # Technical levels
            if ticker in self.features:
                df = self.features[ticker]
                if 'RSI_14' in df.columns:
                    rsi = df['RSI_14'].iloc[-1]
                    rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
                    print(f"📈 RSI: {rsi:.1f} {rsi_status}")
                
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    macd = df['MACD'].iloc[-1]
                    signal = df['MACD_Signal'].iloc[-1]
                    macd_status = "🟢 Bullish" if macd > signal else "🔴 Bearish"
                    print(f"📊 MACD: {macd_status}")
            
            # Risk metrics
            if ticker in self.risk_metrics:
                rm = self.risk_metrics[ticker]
                print(f"\n{Fore.CYAN}⚖️  RISK METRICS{Style.RESET_ALL}")
                
                sharpe_color = Fore.GREEN if rm['sharpe'] > 1 else Fore.YELLOW if rm['sharpe'] > 0 else Fore.RED
                print(f"  Sharpe: {sharpe_color}{rm['sharpe']:.3f}{Style.RESET_ALL}")
                print(f"  Sortino: {rm['sortino']:.3f}")
                print(f"  VaR(95%): {Fore.RED}{rm['var_95']:.2%}{Style.RESET_ALL}")
                print(f"  MaxDD: {Fore.RED}{rm['max_drawdown']:.2%}{Style.RESET_ALL}")
                print(f"  Win Rate: {rm['win_rate']:.1%}")
                print(f"  Kelly: {rm['kelly']:.1%}")
                
                if not np.isnan(rm['beta']):
                    print(f"  Beta: {rm['beta']:.2f}")
                    print(f"  Alpha: {rm['alpha']:.2%}")
            
            # ML predictions
            if ticker in self.ml_results:
                print(f"\n{Fore.CYAN}🤖 ML PREDICTIONS{Style.RESET_ALL}")
                for horizon, result in self.ml_results[ticker].items():
                    direction = "📈 BULLISH" if result['prediction'] == 1 else "📉 BEARISH"
                    conf = result['probability'][result['prediction']] * 100
                    color = Fore.GREEN if result['prediction'] == 1 else Fore.RED
                    print(f"  {horizon}: {color}{direction}{Style.RESET_ALL} (Confidence: {conf:.1f}%, Model: {result['model_name']})")
            
            # Position sizing
            if ticker in self.strategy_recommendations:
                sr = self.strategy_recommendations[ticker]
                print(f"\n{Fore.CYAN}💼 POSITION STRATEGY{Style.RESET_ALL}")
                print(f"  Recommended: {Fore.GREEN}{sr['recommended_size']:.2%}{Style.RESET_ALL} (${sr['position_dollars']:,.2f})")
                print(f"  Shares: {sr['shares']:,}")
                print(f"  Stop Loss: ${sr['stop_loss']:.2f}")
                print(f"  Targets: ${sr['target_1']:.2f} → ${sr['target_2']:.2f} → ${sr['target_3']:.2f}")
        
        # Portfolio allocation
        if len(self.tickers) > 1 and self.portfolio_allocation:
            print(f"\n{Fore.YELLOW}{'='*80}")
            print("PORTFOLIO OPTIMIZATION")
            print(f"{'='*80}{Style.RESET_ALL}\n")
            
            for strategy, alloc in self.portfolio_allocation.items():
                print(f"{Fore.CYAN}{strategy.upper().replace('_', ' ')}:{Style.RESET_ALL}")
                print(f"  Return: {Fore.GREEN}{alloc['expected_return']:.2%}{Style.RESET_ALL}")
                print(f"  Volatility: {alloc['volatility']:.2%}")
                print(f"  Sharpe: {alloc['sharpe']:.2f}")
                print(f"  Allocation:")
                for ticker, weight in alloc['weights'].items():
                    if weight > 0.01:
                        bar = "█" * int(weight * 50)
                        print(f"    {ticker}: {weight:>6.1%} {Fore.CYAN}{bar}{Style.RESET_ALL}")
                print()
    
    def run_interactive_menu(self):
        """Interactive menu system."""
        while True:
            self.print_header("⚡ IMPROVED QUANT SYSTEM", color=Fore.MAGENTA)
            print(f"{Fore.CYAN}Portfolio: {', '.join(self.tickers)}")
            print(f"Capital: ${self.initial_capital:,.0f}{Style.RESET_ALL}\n")
            
            print(f"{Fore.YELLOW}OPTIONS:{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}1.{Style.RESET_ALL} 📊 Full Analysis")
            print(f"  {Fore.GREEN}2.{Style.RESET_ALL} 💬 AI Chat")
            print(f"  {Fore.GREEN}3.{Style.RESET_ALL} 📈 Quick Summary")
            print(f"  {Fore.GREEN}4.{Style.RESET_ALL} 🎯 ML Predictions Only")
            print(f"  {Fore.GREEN}5.{Style.RESET_ALL} ⚖️  Risk Metrics Only")
            print(f"  {Fore.RED}6.{Style.RESET_ALL} 🚪 Exit\n")
            
            try:
                choice = input(f"{Fore.CYAN}Select (1-6): {Style.RESET_ALL}").strip()
                
                if choice == '1':
                    self.display_terminal_analysis()
                    input(f"\n{Fore.YELLOW}Press Enter...{Style.RESET_ALL}")
                elif choice == '2':
                    self.ai_chat_interface()
                elif choice == '3':
                    self.display_quick_summary()
                    input(f"\n{Fore.YELLOW}Press Enter...{Style.RESET_ALL}")
                elif choice == '4':
                    self.display_ml_only()
                    input(f"\n{Fore.YELLOW}Press Enter...{Style.RESET_ALL}")
                elif choice == '5':
                    self.display_risk_only()
                    input(f"\n{Fore.YELLOW}Press Enter...{Style.RESET_ALL}")
                elif choice == '6':
                    print(f"\n{Fore.GREEN}✅ Exiting. Good trading!{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Invalid option{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Exiting...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def display_ml_only(self):
        """Display ML predictions only."""
        print(f"\n{Fore.YELLOW}🤖 ML PREDICTIONS{Style.RESET_ALL}\n")
        for ticker in self.tickers:
            if ticker in self.ml_results:
                print(f"{Fore.CYAN}{ticker}:{Style.RESET_ALL}")
                for horizon, result in self.ml_results[ticker].items():
                    direction = "📈 UP" if result['prediction'] == 1 else "📉 DOWN"
                    conf = result['probability'][result['prediction']] * 100
                    color = Fore.GREEN if result['prediction'] == 1 else Fore.RED
                    print(f"  {horizon}: {color}{direction}{Style.RESET_ALL} ({conf:.0f}%, {result['model_name']}, Acc: {result['accuracy']:.3f})")
                print()
    
    def display_risk_only(self):
        """Display risk metrics only."""
        print(f"\n{Fore.YELLOW}⚖️  RISK METRICS{Style.RESET_ALL}\n")
        for ticker in self.tickers:
            if ticker in self.risk_metrics:
                rm = self.risk_metrics[ticker]
                print(f"{Fore.CYAN}{ticker}:{Style.RESET_ALL}")
                print(f"  Sharpe: {rm['sharpe']:.3f} | Sortino: {rm['sortino']:.3f}")
                print(f"  VaR(95%): {rm['var_95']:.2%} | CVaR(95%): {rm['cvar_95']:.2%}")
                print(f"  MaxDD: {rm['max_drawdown']:.2%} | Win Rate: {rm['win_rate']:.1%}")
                print()
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        try:
            self.print_header("🚀 INITIALIZING IMPROVED QUANT SYSTEM", color=Fore.MAGENTA)
            
            # Fetch all data
            self.fetch_fred_macro_data()
            self.fetch_alpha_vantage_data()
            
            if not self.fetch_market_data():
                self.print_status("Failed to fetch market data", "ERROR")
                return
            
            # Analysis pipeline
            self.calculate_advanced_features()
            self.calculate_risk_metrics()
            self.train_ensemble_models()
            self.calculate_position_sizing()
            
            if len(self.tickers) > 1:
                self.calculate_portfolio_optimization()
            
            self.print_status("✅ Analysis complete!", "SUCCESS")
            
            # Interactive menu
            self.run_interactive_menu()
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted{Style.RESET_ALL}")
        except Exception as e:
            self.print_status(f"Critical error: {e}", "ERROR")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point."""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}⚡ IMPROVED QUANTITATIVE SYSTEM")
    print(f"{Fore.CYAN}Institutional-grade analysis with FRED, Alpha Vantage, ML & AI")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    # Get tickers
    tickers = []
    if len(sys.argv) > 1:
        tickers = [arg.upper() for arg in sys.argv[1:]]
    else:
        try:
            ticker_input = input(f"{Fore.CYAN}Enter tickers (e.g., AAPL MSFT GOOGL): {Style.RESET_ALL}").strip().upper()
            if not ticker_input:
                print(f"{Fore.RED}No tickers provided{Style.RESET_ALL}")
                return
            tickers = ticker_input.split()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Cancelled{Style.RESET_ALL}")
            return
    
    # Get capital
    try:
        capital_input = input(f"{Fore.CYAN}Initial capital (default $1M): {Style.RESET_ALL}").strip()
        capital = float(capital_input.replace(',', '').replace(', '')) if capital_input else 1000000
    except:
        capital = 1000000
    
    # Validate
    valid_tickers = [t for t in tickers if len(t) >= 1 and t.replace('.', '').replace('-', '').isalnum()]
    if not valid_tickers:
        print(f"{Fore.RED}No valid tickers{Style.RESET_ALL}")
        return
    
    # Run
    system = GoldmanSachsQuantSystem(valid_tickers, capital)
    system.run_complete_analysis()

if __name__ == "__main__":
    main()
