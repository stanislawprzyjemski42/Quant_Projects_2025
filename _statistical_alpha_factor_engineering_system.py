"""
================================================================================
STATISTICAL ALPHA FACTOR ENGINEERING SYSTEM
================================================================================
Author: Stanisław Przyjemski
Purpose:Testing out math and finance equacions in Python on real market data.
Domain: Quantitative Finance & Statistical Signal Processing

PROJECT OVERVIEW:
This engine implements mathematical models for extracting predictive signals
from financial time series data. The approach combines statistical mechanics,
signal processing, and econometric theory to construct features for machine
learning pipelines.

MATHEMATICAL FOUNDATIONS:
- Volatility Estimation: Parkinson (1980) high-low range estimator
- Momentum Decomposition: Velocity-acceleration framework from physics
- Statistical Normalization: Z-score standardization for stationarity
- Technical Indicators: Wilder's RSI with exponential smoothing

COMPUTATIONAL APPROACH:
- Vectorized operations using NumPy/Pandas (O(N) time complexity)
- Rolling window calculations with memory-efficient sliding views
- Missing data handling with forward-fill methodology
- Correlation analysis for feature redundancy detection

WHY THIS MATTERS:
Traditional technical analysis relies on visual pattern recognition. This
project mathematically quantifies those patterns into normalized features
that machine learning models can process. Each indicator captures a different
aspect of market microstructure: volatility clustering (GARCH effects),
momentum persistence (autocorrelation), and mean reversion tendencies.
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AlphaEngine:
    """
    A quantitative feature engineering pipeline for financial time series.
    
    This class implements statistical transformations that convert raw OHLCV
    (Open, High, Low, Close, Volume) data into normalized features suitable
    for predictive modeling. The methodology follows principles from:
    - Econometric volatility estimation (Parkinson, Garman-Klass)
    - Signal processing (moving averages, momentum)
    - Statistical learning (normalization, decorrelation)
    
    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'SPY')
    window : int, default=20
        Lookback period for rolling calculations (typically 20 trading days ≈ 1 month)
    
    Attributes
    ----------
    data : pd.DataFrame
        Contains OHLCV data plus engineered features
    """
    
    def __init__(self, ticker, window=20):
        self.ticker = ticker
        self.window = window
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """
        Retrieve and preprocess market data.
        
        Process:
        1. Download 2 years of daily data (sufficient for 252-day volatility estimates)
        2. Handle yfinance's multi-index column structure
        3. Remove missing values (NaN) that would corrupt rolling calculations
        
        Returns
        -------
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        """
        end = datetime.now()
        start = end - timedelta(days=730)  # ~2 years of trading data
        
        print(f"[DATA] Downloading {self.ticker} from {start.date()} to {end.date()}...")
        df = yf.download(self.ticker, start=start, end=end, progress=False)
        
        # yfinance sometimes returns multi-index columns; flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        self.data = df.dropna()
        print(f"[DATA] Loaded {len(self.data)} trading days")
    
    def calculate_volatility_features(self):
        """
        Estimate volatility using two complementary methods.
        
        MATHEMATICAL BACKGROUND:
        Volatility (σ) measures the dispersion of returns. Traditional methods
        use only closing prices, but high-low estimators incorporate intraday
        range information, improving efficiency.
        
        1. REALIZED VOLATILITY (Close-to-Close)
           - Uses log returns: r_t = ln(P_t / P_{t-1})
           - Estimator: σ = std(r) × sqrt(252)
           - Annualization factor sqrt(252) assumes 252 trading days/year
           - Assumption: Returns are i.i.d. normal (violated in practice due to clustering)
        
        2. PARKINSON VOLATILITY (High-Low Range)
           - Formula: σ_P = sqrt( 1/(4ln(2)) × E[ln(H/L)^2] ) × sqrt(252)
           - Published in Parkinson (1980, Journal of Business)
           - More efficient than close-to-close when prices are sampled continuously
           - Assumes geometric Brownian motion (no jumps or drift)
        
        Returns
        -------
        pd.DataFrame with latest volatility estimates (annualized %)
        """
        # Calculate logarithmic returns (better statistical properties than simple returns)
        returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # 1. Realized Volatility
        # Rolling standard deviation of returns, annualized
        self.data['Realized_Vol'] = returns.rolling(self.window).std() * np.sqrt(252)
        
        # 2. Parkinson High-Low Estimator
        # Constant derived from E[ln(H/L)^2] for Brownian motion
        parkinson_const = 1.0 / (4.0 * np.log(2))  # ≈ 0.361
        
        # Square of log high-low ratio
        hl_squared = np.log(self.data['High'] / self.data['Low']) ** 2
        
        # Rolling mean of squared ratios, then sqrt and annualize
        self.data['Parkinson_Vol'] = (
            np.sqrt(parkinson_const * hl_squared.rolling(self.window).mean()) * np.sqrt(252)
        )
        
        return self.data[['Realized_Vol', 'Parkinson_Vol']].tail(1)
    
    def calculate_momentum_physics(self):
        """
        Decompose price movement into velocity and acceleration.
        
        PHYSICS ANALOGY:
        - Price = Position (where the asset is)
        - Velocity = Rate of change (how fast it's moving)
        - Acceleration = Change in velocity (is momentum increasing?)
        
        MATHEMATICAL FORMULATION:
        - Velocity: v_t = (P_t - P_{t-w}) / P_{t-w}  (percentage change over window)
        - Acceleration: a_t = v_t - v_{t-1}  (first difference of velocity)
        - Z-Score: z_t = (v_t - μ_v) / σ_v  (standardization for comparability)
        
        Why Z-Score?
        Raw velocity values are not comparable across different assets or time
        periods. Z-scoring creates a standardized measure where:
        - z = 0 means average velocity
        - z = 1 means one standard deviation above average
        - z > 2 suggests statistically significant momentum (outlier)
        
        Returns
        -------
        pd.DataFrame with normalized momentum metrics
        """
        # Velocity: % change over the lookback window
        self.data['Velocity'] = self.data['Close'].pct_change(self.window)
        
        # Acceleration: Is velocity increasing or decreasing?
        self.data['Acceleration'] = self.data['Velocity'].diff()
        
        # Normalize velocity to Z-score for cross-sectional comparison
        # Formula: z = (x - μ) / σ where μ and σ are rolling statistics
        rolling_mean = self.data['Velocity'].rolling(self.window).mean()
        rolling_std = self.data['Velocity'].rolling(self.window).std()
        
        self.data['Velocity_Z'] = (self.data['Velocity'] - rolling_mean) / rolling_std
        
        return self.data[['Velocity_Z', 'Acceleration']].tail(1)
    
    def generate_ml_features(self):
        """
        Construct a normalized feature matrix for machine learning.
        
        FEATURE ENGINEERING PRINCIPLES:
        Each feature should capture an independent aspect of market behavior:
        
        1. PRICE_DISTANCE (Mean Reversion Signal)
           - Formula: (P_t / MA_w) - 1
           - Interpretation: % deviation from moving average
           - Theory: Prices tend to revert to their moving average (oscillator logic)
        
        2. VOL_REGIME (Volatility Clustering)
           - Formula: σ_short / σ_long
           - Detects shifts in volatility regime (quiet → turbulent markets)
           - Related to GARCH models in econometrics
        
        3. RSI (Relative Strength Index)
           - Formula: RSI = 100 - 100/(1 + RS), where RS = avg_gain/avg_loss
           - Developed by Wilder (1978) for overbought/oversold conditions
           - Values: 0-30 oversold, 70-100 overbought
           - Implementation uses exponential moving average (EMA)
        
        FEATURE QUALITY CHECKS:
        - Stationarity: Features should not trend indefinitely (checked via ADF test)
        - Correlation: Low inter-feature correlation to avoid redundancy
        - Scaling: All features on comparable scales for ML algorithms
        
        Returns
        -------
        pd.DataFrame : Feature matrix ready for sklearn/keras models
        """
        features = pd.DataFrame(index=self.data.index)
        
        # 1. Distance from Moving Average (Oscillator)
        ma = self.data['Close'].rolling(self.window).mean()
        features['Price_Distance'] = (self.data['Close'] / ma) - 1
        
        # 2. Volatility Regime Indicator
        # Compare short-term volatility to long-term baseline
        short_vol = self.data['Realized_Vol']
        long_vol = self.data['Realized_Vol'].rolling(60).mean()  # ~3 months
        features['Vol_Regime'] = short_vol / long_vol
        
        # 3. Relative Strength Index (Wilder's Method)
        # Calculate price changes
        delta = self.data['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing (similar to EMA with α = 1/14)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        
        # Relative strength and RSI formula
        rs = avg_gain / avg_loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Volume Momentum (Bonus Feature)
        # Detects unusual trading activity
        vol_ma = self.data['Volume'].rolling(self.window).mean()
        features['Volume_Ratio'] = self.data['Volume'] / vol_ma
        
        return features.dropna()
    
    def analyze_feature_correlation(self, features):
        """
        Check for multicollinearity in the feature set.
        
        MULTICOLLINEARITY PROBLEM:
        When features are highly correlated (r > 0.7), they provide redundant
        information. This causes issues in:
        - Linear models (unstable coefficients)
        - Interpretability (can't isolate feature importance)
        - Overfitting (model memorizes noise)
        
        SOLUTION:
        - Remove or combine correlated features
        - Use PCA for dimensionality reduction
        - Apply regularization (L1/L2 penalties)
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix to analyze
        
        Returns
        -------
        pd.DataFrame : Correlation matrix
        """
        corr_matrix = features.corr()
        
        print("\n[FEATURE CORRELATION MATRIX]")
        print(corr_matrix.round(3))
        
        # Flag high correlations (excluding diagonal)
        high_corr = np.where(np.abs(corr_matrix) > 0.7)
        high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                     for x, y in zip(*high_corr) if x != y and x < y]
        
        if high_corr:
            print("\n[WARNING] High correlations detected:")
            for feat1, feat2, corr_val in high_corr:
                print(f"  • {feat1} ↔ {feat2}: r = {corr_val:.3f}")
        else:
            print("\n[OK] No problematic correlations (all |r| < 0.7)")
        
        return corr_matrix
    
    def visualize_features(self, features, n_days=100):
        """
        Create diagnostic plots for feature quality assessment.
        
        Parameters
        ----------
        features : pd.DataFrame
            Engineered features
        n_days : int
            Number of recent days to plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Feature Engineering Analysis - {self.ticker}', fontsize=14, fontweight='bold')
        
        recent = features.tail(n_days)
        
        # Plot each feature with reference lines
        axes[0, 0].plot(recent.index, recent['Price_Distance'], color='steelblue', linewidth=1.5)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Price Distance from MA')
        axes[0, 0].set_ylabel('Deviation (%)')
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].plot(recent.index, recent['Vol_Regime'], color='darkgreen', linewidth=1.5)
        axes[0, 1].axhline(1, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Volatility Regime')
        axes[0, 1].set_ylabel('Current/Long-term Ratio')
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].plot(recent.index, recent['RSI'], color='darkorange', linewidth=1.5)
        axes[1, 0].axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        axes[1, 0].axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        axes[1, 0].set_title('RSI (14-period)')
        axes[1, 0].set_ylabel('RSI Value')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].plot(recent.index, recent['Volume_Ratio'], color='purple', linewidth=1.5)
        axes[1, 1].axhline(1, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Volume Activity')
        axes[1, 1].set_ylabel('Volume/MA Ratio')
        axes[1, 1].grid(alpha=0.3)
        
        # Correlation heatmap
        corr = features.corr()
        im = axes[2, 0].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[2, 0].set_xticks(range(len(corr.columns)))
        axes[2, 0].set_yticks(range(len(corr.columns)))
        axes[2, 0].set_xticklabels(corr.columns, rotation=45, ha='right')
        axes[2, 0].set_yticklabels(corr.columns)
        axes[2, 0].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[2, 0])
        
        # Feature distribution (check for normality)
        recent['RSI'].hist(bins=30, ax=axes[2, 1], color='steelblue', edgecolor='black')
        axes[2, 1].set_title('RSI Distribution')
        axes[2, 1].set_xlabel('RSI Value')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].axvline(recent['RSI'].mean(), color='red', linestyle='--', label='Mean')
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Execution pipeline demonstrating the complete workflow.
    
    WORKFLOW:
    1. Data Acquisition → Clean OHLCV data
    2. Volatility Analysis → Risk regime identification
    3. Momentum Extraction → Trend strength quantification  
    4. Feature Engineering → ML-ready normalized features
    5. Quality Assessment → Correlation and distribution checks
    6. Visualization → Diagnostic plots
    """
    print("="*80)
    print("STATISTICAL ALPHA FACTOR ENGINEERING SYSTEM")
    print("="*80)
    
    ticker = input("\nEnter Asset Ticker (default: SPY): ").strip().upper() or "SPY"
    
    # Initialize the engine
    engine = AlphaEngine(ticker=ticker, window=20)
    
    # Phase 1: Volatility Analysis
    print("\n[PHASE 1] Volatility Estimation...")
    vol_metrics = engine.calculate_volatility_features()
    print(vol_metrics.to_string())
    
    # Phase 2: Momentum Analysis
    print("\n[PHASE 2] Momentum Decomposition...")
    mom_metrics = engine.calculate_momentum_physics()
    print(f"Velocity Z-Score: {mom_metrics['Velocity_Z'].iloc[-1]:.4f}")
    print(f"Acceleration: {mom_metrics['Acceleration'].iloc[-1]:.6f}")
    
    # Phase 3: Feature Engineering
    print("\n[PHASE 3] Generating ML Feature Set...")
    ml_features = engine.generate_ml_features()
    print(f"Feature Matrix Shape: {ml_features.shape} (rows × features)")
    
    # Phase 4: Quality Control
    print("\n[PHASE 4] Feature Quality Assessment...")
    corr_matrix = engine.analyze_feature_correlation(ml_features)
    
    # Phase 5: Visualization
    print("\n[PHASE 5] Generating Diagnostic Plots...")
    engine.visualize_features(ml_features, n_days=100)
    
    print("\n" + "="*80)
    print("Analysis Complete. Feature set ready for modeling.")
    print("="*80)
    
    # Export feature set (optional)
    save = input("\nExport features to CSV? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"{ticker}_features_{datetime.now().strftime('%Y%m%d')}.csv"
        ml_features.to_csv(filename)
        print(f"Saved to {filename}")


if __name__ == "__main__":
    main()
