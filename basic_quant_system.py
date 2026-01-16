# streamlined_quant_system.py
"""
Streamlined Quantitative Stock Trading System
20 Essential Features | ML Ensemble | LLM Insights | Personal Trading Focus
"""

import os
import warnings
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from colorama import Fore, Style, init
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from dotenv import load_dotenv
from google import genai
import traceback

warnings.filterwarnings('ignore')
init(autoreset=True)
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'ANALYSIS_PERIOD': '2y',
    'MIN_DATA_POINTS': 300,
    'RISK_FREE_RATE': 0.045,
    'CONFIDENCE_LEVEL': 0.95,
}

# ============================================================================
# CORE METRICS ENGINE (20 ESSENTIAL FEATURES)
# ============================================================================
class CoreMetrics:
    """20 essential quantitative features for trading decisions"""
    
    @staticmethod
    def calculate_all_metrics(df):
        """Calculate all 20 essential metrics"""
        metrics = {}
        
        # 1-5: Price & Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        metrics['Current_Price'] = df['Close'].iloc[-1]
        metrics['Price_Change_1M'] = ((df['Close'].iloc[-1] / df['Close'].iloc[-21]) - 1) * 100 if len(df) > 21 else np.nan
        metrics['Price_Change_3M'] = ((df['Close'].iloc[-1] / df['Close'].iloc[-63]) - 1) * 100 if len(df) > 63 else np.nan
        
        # 6-8: Volatility Measures
        metrics['Volatility_Ann'] = df['Returns'].std() * np.sqrt(252)
        metrics['Volatility_20D'] = df['Returns'].rolling(20).std().iloc[-1] * np.sqrt(252) if len(df) > 20 else np.nan
        metrics['Realized_Vol'] = np.sqrt((df['Returns'][-20:]**2).sum()) * np.sqrt(252) if len(df) > 20 else np.nan
        
        # 9-11: Technical Indicators
        df['RSI'] = CoreMetrics._calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = CoreMetrics._calculate_macd(df['Close'])
        metrics['RSI_Latest'] = df['RSI'].iloc[-1]
        metrics['MACD_Signal'] = 'Bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'Bearish'
        
        # 12-14: Moving Averages
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_200'] = df['Close'].rolling(200).mean()
        metrics['Price_vs_MA50'] = ((df['Close'].iloc[-1] / df['MA_50'].iloc[-1]) - 1) * 100 if df['MA_50'].iloc[-1] > 0 else np.nan
        
        # 15-17: Risk-Adjusted Returns
        returns_clean = df['Returns'].dropna()
        metrics['Sharpe_Ratio'] = CoreMetrics._sharpe_ratio(returns_clean)
        metrics['Sortino_Ratio'] = CoreMetrics._sortino_ratio(returns_clean)
        metrics['Max_Drawdown'] = CoreMetrics._max_drawdown(df['Close'])
        
        # 18-19: Statistical Moments
        metrics['Skewness'] = returns_clean.skew()
        metrics['Kurtosis'] = returns_clean.kurtosis()
        
        # 20: Momentum Score
        metrics['Momentum_Score'] = CoreMetrics._momentum_score(df)
        
        return metrics, df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    @staticmethod
    def _sharpe_ratio(returns, rf_rate=0.045):
        """Calculate Sharpe Ratio"""
        excess = returns - (rf_rate / 252)
        return (excess.mean() * 252) / (returns.std() * np.sqrt(252))
    
    @staticmethod
    def _sortino_ratio(returns, rf_rate=0.045):
        """Calculate Sortino Ratio"""
        excess = returns - (rf_rate / 252)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return np.nan
        downside_std = downside.std() * np.sqrt(252)
        return (excess.mean() * 252) / downside_std if downside_std != 0 else np.nan
    
    @staticmethod
    def _max_drawdown(prices):
        """Calculate Maximum Drawdown"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min() * 100
    
    @staticmethod
    def _momentum_score(df):
        """Calculate composite momentum score (0-100)"""
        score = 0
        
        # RSI momentum (0-25 points)
        rsi = df['RSI'].iloc[-1]
        if 40 < rsi < 70:
            score += 25
        elif 30 < rsi < 80:
            score += 15
        
        # MACD momentum (0-25 points)
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            score += 25
        
        # Price vs MA (0-25 points)
        if df['Close'].iloc[-1] > df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1]:
            score += 25
        elif df['Close'].iloc[-1] > df['MA_50'].iloc[-1]:
            score += 15
        
        # Recent trend (0-25 points)
        if len(df) > 20:
            recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
            if recent_return > 5:
                score += 25
            elif recent_return > 2:
                score += 15
            elif recent_return > 0:
                score += 10
        
        return score


# ============================================================================
# FUNDAMENTAL ANALYZER
# ============================================================================
class FundamentalAnalyzer:
    """Analyze fundamental metrics"""
    
    @staticmethod
    def get_fundamentals(ticker_obj):
        """Fetch key fundamental metrics"""
        info = ticker_obj.info
        
        fundamentals = {
            'PE_Ratio': info.get('trailingPE', np.nan),
            'Forward_PE': info.get('forwardPE', np.nan),
            'PEG_Ratio': info.get('pegRatio', np.nan),
            'Price_to_Book': info.get('priceToBook', np.nan),
            'ROE': info.get('returnOnEquity', np.nan),
            'ROA': info.get('returnOnAssets', np.nan),
            'Profit_Margin': info.get('profitMargins', np.nan),
            'Debt_to_Equity': info.get('debtToEquity', np.nan),
            'Current_Ratio': info.get('currentRatio', np.nan),
            'Revenue_Growth': info.get('revenueGrowth', np.nan),
            'Market_Cap': info.get('marketCap', np.nan),
            'Beta': info.get('beta', np.nan),
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown'),
        }
        
        # Calculate quality score
        fundamentals['Quality_Score'] = FundamentalAnalyzer._calculate_quality_score(fundamentals)
        
        return fundamentals
    
    @staticmethod
    def _calculate_quality_score(fundamentals):
        """Calculate quality score (0-100)"""
        score = 0
        
        # Profitability (0-30)
        roe = fundamentals.get('ROE', 0) or 0
        if roe > 0.20:
            score += 15
        elif roe > 0.15:
            score += 10
        elif roe > 0.10:
            score += 5
        
        profit_margin = fundamentals.get('Profit_Margin', 0) or 0
        if profit_margin > 0.20:
            score += 15
        elif profit_margin > 0.10:
            score += 10
        elif profit_margin > 0.05:
            score += 5
        
        # Value (0-30)
        pe = fundamentals.get('PE_Ratio', 0) or 0
        if 0 < pe < 15:
            score += 15
        elif 0 < pe < 25:
            score += 10
        elif 0 < pe < 35:
            score += 5
        
        pb = fundamentals.get('Price_to_Book', 0) or 0
        if 0 < pb < 2:
            score += 15
        elif 0 < pb < 4:
            score += 10
        elif 0 < pb < 6:
            score += 5
        
        # Financial Health (0-40)
        current_ratio = fundamentals.get('Current_Ratio', 0) or 0
        if current_ratio > 2:
            score += 20
        elif current_ratio > 1.5:
            score += 15
        elif current_ratio > 1:
            score += 10
        
        debt_to_equity = fundamentals.get('Debt_to_Equity', 100) or 100
        if debt_to_equity < 0.3:
            score += 20
        elif debt_to_equity < 0.5:
            score += 15
        elif debt_to_equity < 1.0:
            score += 10
        
        return min(score, 100)


# ============================================================================
# ML PREDICTION ENGINE
# ============================================================================
class MLPredictor:
    """Ensemble ML prediction using RF, LightGBM, and GBM"""
    
    @staticmethod
    def prepare_features(df):
        """Prepare features for ML"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['Returns']
        features['log_returns'] = df['Log_Returns']
        features['volatility'] = df['Returns'].rolling(20).std()
        
        # Technical features
        features['rsi'] = df['RSI']
        features['macd'] = df['MACD']
        features['price_to_ma20'] = df['Close'] / df['MA_20']
        features['price_to_ma50'] = df['Close'] / df['MA_50']
        
        # Momentum features
        features['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        features['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        features['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Target: 5-day forward return > 0
        features['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        
        return features.dropna()
    
    @staticmethod
    def train_ensemble(df):
        """Train ensemble and return predictions"""
        feature_df = MLPredictor.prepare_features(df)
        
        if len(feature_df) < 100:
            return None
        
        # Split data
        feature_cols = [col for col in feature_df.columns if col != 'target']
        X = feature_df[feature_cols].values
        y = feature_df['target'].values
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
        }
        
        predictions = []
        accuracies = []
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            accuracy = model.score(X_test_scaled, y_test)
            
            # Latest prediction
            X_latest = scaler.transform(X[-1:])
            pred_proba = model.predict_proba(X_latest)[0][1]
            
            predictions.append(pred_proba)
            accuracies.append(accuracy)
        
        return {
            'Bullish_Probability': np.mean(predictions),
            'Prediction_Confidence': 1 - np.std(predictions),
            'Ensemble_Accuracy': np.mean(accuracies),
            'Signal': 'BUY' if np.mean(predictions) > 0.6 else 'SELL' if np.mean(predictions) < 0.4 else 'HOLD'
        }


# ============================================================================
# MAIN ANALYZER
# ============================================================================
class StreamlinedAnalyzer:
    """Main analysis engine"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.client = None
        self.model = None
        
        if self.gemini_api_key:
            try:
                self.client = genai.Client(api_key=self.gemini_api_key)
                self.model = 'gemini-2.0-flash-exp'
            except:
                print(f"{Fore.YELLOW}Warning: Gemini AI not available{Style.RESET_ALL}")
        
        self.results = {}
        self.benchmark = None
    
    def print_header(self, text):
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}{text.center(80)}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    def print_status(self, text, status="INFO"):
        colors = {"INFO": Fore.BLUE, "SUCCESS": Fore.GREEN, "WARNING": Fore.YELLOW, "ERROR": Fore.RED}
        print(f"{colors.get(status, Fore.WHITE)}[{status}] {text}{Style.RESET_ALL}")
    
    def fetch_benchmark(self):
        """Fetch SPY benchmark"""
        try:
            spy = yf.Ticker('SPY')
            self.benchmark = spy.history(period=CONFIG['ANALYSIS_PERIOD'])
            self.benchmark['Returns'] = self.benchmark['Close'].pct_change()
            self.print_status("Benchmark (SPY) loaded", "SUCCESS")
        except Exception as e:
            self.print_status(f"Benchmark error: {str(e)}", "WARNING")
    
    def analyze_stock(self, ticker):
        """Comprehensive stock analysis"""
        self.print_status(f"Analyzing {ticker}...", "INFO")
        
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period=CONFIG['ANALYSIS_PERIOD'])
            
            if df.empty or len(df) < CONFIG['MIN_DATA_POINTS']:
                self.print_status(f"Insufficient data for {ticker}", "WARNING")
                return None
            
            # Calculate metrics
            metrics, df = CoreMetrics.calculate_all_metrics(df)
            metrics['Ticker'] = ticker
            
            # Get fundamentals
            fundamentals = FundamentalAnalyzer.get_fundamentals(stock)
            metrics.update(fundamentals)
            
            # ML predictions
            ml_results = MLPredictor.train_ensemble(df)
            if ml_results:
                metrics.update(ml_results)
            
            # Calculate beta vs SPY
            if self.benchmark is not None:
                aligned = pd.DataFrame({
                    'stock': df['Returns'],
                    'market': self.benchmark['Returns']
                }).dropna()
                
                if len(aligned) > 30:
                    cov = np.cov(aligned['stock'], aligned['market'])[0, 1]
                    var = np.var(aligned['market'])
                    metrics['Beta_Calculated'] = cov / var if var != 0 else np.nan
            
            # Calculate composite score
            metrics['Composite_Score'] = self._calculate_composite_score(metrics)
            
            self.print_status(f"✓ {ticker} - Score: {metrics['Composite_Score']:.1f}/100", "SUCCESS")
            return metrics
            
        except Exception as e:
            self.print_status(f"Error analyzing {ticker}: {str(e)}", "ERROR")
            return None
    
    def _calculate_composite_score(self, metrics):
        """Calculate overall score (0-100)"""
        score = 0
        
        # Technical strength (40 points)
        score += min(metrics.get('Momentum_Score', 0) * 0.4, 40)
        
        # Risk-adjusted returns (30 points)
        sharpe = metrics.get('Sharpe_Ratio', 0)
        if sharpe > 2.0:
            score += 30
        elif sharpe > 1.5:
            score += 25
        elif sharpe > 1.0:
            score += 20
        elif sharpe > 0.5:
            score += 10
        
        # Quality (20 points)
        score += min(metrics.get('Quality_Score', 0) * 0.2, 20)
        
        # ML confidence (10 points)
        if 'Bullish_Probability' in metrics:
            ml_score = (metrics['Bullish_Probability'] - 0.5) * 20  # Scale from 0-1 to 0-10
            score += max(0, min(ml_score, 10))
        
        return min(score, 100)
    
    def generate_report(self, results):
        """Generate comprehensive report"""
        self.print_header("QUANTITATIVE ANALYSIS REPORT")
        
        report = []
        report.append("="*80)
        report.append(f"STREAMLINED QUANT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("20 Essential Features | ML Ensemble | LLM Insights")
        report.append("="*80)
        report.append("")
        
        # Sort by composite score
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('Composite_Score', 0), reverse=True)
        
        for ticker, metrics in sorted_results:
            recommendation = self._get_recommendation(metrics['Composite_Score'])
            
            report.append(f"\n{ticker} - [{recommendation}] - Score: {metrics['Composite_Score']:.1f}/100")
            report.append("-"*80)
            
            # Price & Performance
            report.append(f"  Price: ${metrics['Current_Price']:.2f} | "
                         f"1M: {metrics.get('Price_Change_1M', np.nan):.1f}% | "
                         f"3M: {metrics.get('Price_Change_3M', np.nan):.1f}%")
            
            # Risk Metrics
            report.append(f"  Risk: Sharpe={metrics.get('Sharpe_Ratio', np.nan):.2f} | "
                         f"Sortino={metrics.get('Sortino_Ratio', np.nan):.2f} | "
                         f"MaxDD={metrics.get('Max_Drawdown', np.nan):.1f}% | "
                         f"Vol={metrics.get('Volatility_Ann', np.nan):.2%}")
            
            # Technical
            report.append(f"  Technical: RSI={metrics.get('RSI_Latest', np.nan):.1f} | "
                         f"MACD={metrics.get('MACD_Signal', 'N/A')} | "
                         f"Momentum={metrics.get('Momentum_Score', 0):.0f}/100")
            
            # Fundamentals
            report.append(f"  Fundamentals: PE={metrics.get('PE_Ratio', np.nan):.1f} | "
                         f"ROE={metrics.get('ROE', np.nan):.1%} | "
                         f"Quality={metrics.get('Quality_Score', 0):.0f}/100")
            
            # ML Prediction
            if 'Bullish_Probability' in metrics:
                report.append(f"  ML Ensemble: {metrics['Signal']} | "
                             f"Bullish={metrics['Bullish_Probability']:.1%} | "
                             f"Confidence={metrics['Prediction_Confidence']:.2f} | "
                             f"Accuracy={metrics['Ensemble_Accuracy']:.1%}")
            
            report.append("")
        
        # Summary statistics
        report.append("="*80)
        report.append("PORTFOLIO SUMMARY")
        report.append("="*80)
        
        scores = [m.get('Composite_Score', 0) for m in results.values()]
        report.append(f"  Average Score: {np.mean(scores):.1f}/100")
        report.append(f"  Stocks Analyzed: {len(results)}")
        
        strong_buys = sum(1 for m in results.values() if m.get('Composite_Score', 0) >= 75)
        buys = sum(1 for m in results.values() if 65 <= m.get('Composite_Score', 0) < 75)
        report.append(f"  Strong Buy: {strong_buys} | Buy: {buys}")
        
        report.append("")
        
        full_report = "\n".join(report)
        print(full_report)
        
        # Save to file
        with open('quant_report.txt', 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        return full_report
    
    def _get_recommendation(self, score):
        """Get recommendation based on score"""
        if score >= 75:
            return "STRONG BUY"
        elif score >= 65:
            return "BUY"
        elif score >= 50:
            return "HOLD"
        elif score >= 40:
            return "SELL"
        else:
            return "STRONG SELL"
    
    def get_llm_insights(self, report):
        """Get LLM-powered insights"""
        if not self.client:
            self.print_status("LLM not available - set GEMINI_API_KEY", "WARNING")
            return
        
        self.print_header("🤖 AI-POWERED INSIGHTS")
        
        prompt = f"""You are an expert quantitative analyst. Analyze this stock report and provide:

1. Top 3 investment opportunities with probability-driven reasoning
2. Key risks and concerns
3. Suggested portfolio allocation strategy
4. Market timing considerations

Report:
{report}

Be concise, quantitative, and actionable. Focus on probability and risk-adjusted returns."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            print(f"{Fore.CYAN}{response.text}{Style.RESET_ALL}\n")
            
            # Save insights
            with open('ai_insights.txt', 'w', encoding='utf-8') as f:
                f.write(response.text)
            
        except Exception as e:
            self.print_status(f"LLM error: {str(e)}", "ERROR")
    
    def interactive_mode(self):
        """Interactive command mode"""
        self.print_header("📊 INTERACTIVE MODE")
        print(f"{Fore.GREEN}Commands:{Style.RESET_ALL}")
        print("  'analyze [TICKER1 TICKER2 ...]' - Analyze stocks")
        print("  'show' - Show current results")
        print("  'insights' - Get AI insights")
        print("  'compare [TICKER1] [TICKER2]' - Compare two stocks")
        print("  'export' - Export to CSV")
        print("  'exit' - Quit")
        print()
        
        while True:
            try:
                cmd = input(f"{Fore.YELLOW}> {Style.RESET_ALL}").strip().lower()
                
                if cmd == 'exit':
                    print(f"{Fore.GREEN}Happy trading!{Style.RESET_ALL}")
                    break
                
                elif cmd.startswith('analyze'):
                    tickers = cmd.split()[1:]
                    if not tickers:
                        print(f"{Fore.RED}Usage: analyze TICKER1 TICKER2 ...{Style.RESET_ALL}")
                        continue
                    
                    self.fetch_benchmark()
                    
                    for ticker in tickers:
                        result = self.analyze_stock(ticker.upper())
                        if result:
                            self.results[ticker.upper()] = result
                    
                    if self.results:
                        report = self.generate_report(self.results)
                
                elif cmd == 'show':
                    if not self.results:
                        print(f"{Fore.RED}No results yet. Use 'analyze' first.{Style.RESET_ALL}")
                    else:
                        self.generate_report(self.results)
                
                elif cmd == 'insights':
                    if not self.results:
                        print(f"{Fore.RED}No results yet. Use 'analyze' first.{Style.RESET_ALL}")
                    else:
                        report = self.generate_report(self.results)
                        self.get_llm_insights(report)
                
                elif cmd.startswith('compare'):
                    parts = cmd.split()
                    if len(parts) != 3:
                        print(f"{Fore.RED}Usage: compare TICKER1 TICKER2{Style.RESET_ALL}")
                        continue
                    
                    t1, t2 = parts[1].upper(), parts[2].upper()
                    if t1 in self.results and t2 in self.results:
                        self._compare_stocks(t1, t2)
                    else:
                        print(f"{Fore.RED}Stocks not analyzed yet{Style.RESET_ALL}")
                
                elif cmd == 'export':
                    if not self.results:
                        print(f"{Fore.RED}No results to export{Style.RESET_ALL}")
                    else:
                        df = pd.DataFrame(self.results).T
                        df.to_csv('analysis_results.csv')
                        print(f"{Fore.GREEN}Exported to analysis_results.csv{Style.RESET_ALL}")
                
                else:
                    print(f"{Fore.RED}Unknown command{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n{Fore.GREEN}Exiting...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def _compare_stocks(self, t1, t2):
        """Compare two stocks"""
        m1 = self.results[t1]
        m2 = self.results[t2]
        
        print(f"\n{Fore.CYAN}Comparing {t1} vs {t2}{Style.RESET_ALL}\n")
        
        metrics_to_compare = [
            ('Composite_Score', 'Score'),
            ('Sharpe_Ratio', 'Sharpe'),
            ('Sortino_Ratio', 'Sortino'),
            ('Volatility_Ann', 'Volatility'),
            ('Momentum_Score', 'Momentum'),
            ('Quality_Score', 'Quality'),
            ('Bullish_Probability', 'ML Bullish %'),
        ]
        
        for key, label in metrics_to_compare:
            v1 = m1.get(key, np.nan)
            v2 = m2.get(key, np.nan)
            
            if pd.notna(v1) and pd.notna(v2):
                if key == 'Bullish_Probability':
                    v1, v2 = v1 * 100, v2 * 100
                
                winner = t1 if v1 > v2 else t2
                print(f"  {label:15s}: {t1}={v1:7.2f} | {t2}={v2:7.2f} | Winner: {winner}")
        
        print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution"""
    print(f"{Fore.CYAN}")
    print("="*80)
    print("STREAMLINED QUANTITATIVE TRADING SYSTEM".center(80))
    print("20 Essential Features | ML Ensemble | LLM Insights".center(80))
    print("="*80)
    print(f"{Style.RESET_ALL}\n")
    
    analyzer = StreamlinedAnalyzer()
    
    # Check for command-line arguments
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode
        tickers = [t.upper() for t in sys.argv[1:]]
        print(f"{Fore.GREEN}Analyzing: {', '.join(tickers)}{Style.RESET_ALL}\n")
        
        analyzer.fetch_benchmark()
        
        for ticker in tickers:
            result = analyzer.analyze_stock(ticker)
            if result:
                analyzer.results[ticker] = result
        
        if analyzer.results:
            report = analyzer.generate_report(analyzer.results)
            
            # Ask for AI insights
            if analyzer.client:
                response = input(f"\n{Fore.YELLOW}Get AI insights? (y/n): {Style.RESET_ALL}").strip().lower()
                if response == 'y':
                    analyzer.get_llm_insights(report)
            
            # Export option
            response = input(f"\n{Fore.YELLOW}Export to CSV? (y/n): {Style.RESET_ALL}").strip().lower()
            if response == 'y':
                df = pd.DataFrame(analyzer.results).T
                df.to_csv('analysis_results.csv')
                print(f"{Fore.GREEN}✓ Exported to analysis_results.csv{Style.RESET_ALL}")
    else:
        # Interactive mode
        analyzer.interactive_mode()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.GREEN}Session ended. Happy trading!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        traceback.print_exc()
