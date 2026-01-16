"""
================================================================================
MACHINE LEARNING PRICE FORECASTING WITH MONTE CARLO SIMULATION
================================================================================
Author: Stanisław Przyjemski
Date: September 2025
Project Type: Pre-built Model Implementation for Financial Markets
Domain: Machine Learning & Stochastic Modeling

PROJECT OVERVIEW:
This system combines supervised learning with probabilistic simulation to
forecast asset prices. Rather than giving a single prediction, it provides
a distribution of possible outcomes, acknowledging the inherent uncertainty
in financial markets.

TECHNICAL COMPONENTS:
1. Machine Learning: Random Forest Regression for pattern recognition
2. Monte Carlo: Geometric Brownian Motion for uncertainty quantification
3. Ensemble Methods: Combining multiple weak learners for robustness
4. Statistical Validation: Out-of-sample testing and confidence intervals

MATHEMATICAL FOUNDATIONS:
- Geometric Brownian Motion: dS = μS dt + σS dW (Wiener process)
- Bootstrap Aggregating: Reduces variance through random sampling
- Walk-Forward Validation: Tests model on unseen future data
- Drift-Diffusion Model: Separates trend (drift) from noise (volatility)

WHY THIS APPROACH:
Traditional ML gives point predictions without uncertainty. Monte Carlo
simulations provide probability distributions. This hybrid approach:
- Uses ML to learn historical patterns
- Uses Monte Carlo to model randomness
- Provides confidence intervals instead of false precision
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MLMonteCarloPricePredictor:
    """
    Hybrid prediction system combining machine learning with stochastic simulation.
    
    This class implements two complementary forecasting methods:
    - Random Forest: Learns complex non-linear patterns from historical data
    - Monte Carlo: Simulates thousands of possible future price paths
    
    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    forecast_days : int, default=30
        Number of trading days to predict forward
    n_simulations : int, default=1000
        Number of Monte Carlo paths to generate
    
    Attributes
    ----------
    data : pd.DataFrame
        Historical price data with engineered features
    model : RandomForestRegressor
        Trained ML model
    """
    
    def __init__(self, ticker, forecast_days=30, n_simulations=1000):
        self.ticker = ticker
        self.forecast_days = forecast_days
        self.n_simulations = n_simulations
        self.data = None
        self.model = None
        self.feature_importance = None
        
    def load_data(self, days=730):
        """
        Download and prepare historical market data.
        
        Parameters
        ----------
        days : int
            Historical data window (default: 2 years)
        
        Returns
        -------
        pd.DataFrame with OHLCV data
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        
        print(f"[DATA] Fetching {self.ticker} from {start.date()} to {end.date()}...")
        df = yf.download(self.ticker, start=start, end=end, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        self.data = df.dropna()
        print(f"[DATA] Loaded {len(self.data)} trading days")
        return self.data
    
    def engineer_features(self):
        """
        Create predictive features from raw price data.
        
        FEATURE ENGINEERING STRATEGY:
        ML models need informative inputs. Raw prices are not stationary
        (they trend indefinitely), so we transform them into:
        
        1. TECHNICAL INDICATORS (Momentum & Trend)
           - SMA: Simple Moving Average (trend direction)
           - RSI: Relative Strength Index (overbought/oversold)
           - MACD: Moving Average Convergence Divergence (momentum)
        
        2. PRICE TRANSFORMS (Stationarity)
           - Returns: % change (removes price level dependency)
           - Volatility: Rolling standard deviation (risk measure)
        
        3. LAGGED FEATURES (Time Dependencies)
           - Previous day's return (autocorrelation)
           - 5-day and 20-day historical returns
        
        Returns
        -------
        pd.DataFrame with feature columns
        """
        df = self.data.copy()
        
        # 1. Moving Averages (Trend Indicators)
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_30'] = df['Close'].rolling(30).mean()
        df['SMA_Ratio'] = df['SMA_10'] / df['SMA_30']  # Golden cross signal
        
        # 2. Returns (Stationarity Transform)
        df['Returns'] = df['Close'].pct_change()
        df['Returns_Lag1'] = df['Returns'].shift(1)
        df['Returns_Lag5'] = df['Returns'].shift(5)
        
        # 3. Volatility (Risk Measure)
        df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        
        # 4. RSI (Momentum Oscillator)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. MACD (Trend Momentum)
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # 6. Volume Features (Market Participation)
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 7. Price Position (Relative Valuation)
        df['Price_to_High'] = df['Close'] / df['High'].rolling(20).max()
        df['Price_to_Low'] = df['Close'] / df['Low'].rolling(20).min()
        
        self.data = df.dropna()
        return self.data
    
    def train_ml_model(self, test_size=0.2):
        """
        Train Random Forest model with walk-forward validation.
        
        ALGORITHM CHOICE: Random Forest
        - Ensemble of decision trees (typically 100-500 trees)
        - Each tree trained on random subset of data (bagging)
        - Reduces overfitting compared to single decision tree
        - Handles non-linear relationships without feature scaling
        - Provides feature importance scores
        
        VALIDATION STRATEGY: Train-Test Split
        - 80% training data (learn patterns)
        - 20% test data (evaluate on unseen future data)
        - Sequential split (no future information leakage)
        
        Parameters
        ----------
        test_size : float
            Fraction of data for testing (default: 0.2)
        
        Returns
        -------
        dict with performance metrics
        """
        # Define feature columns (exclude target and metadata)
        feature_cols = ['SMA_Ratio', 'Returns_Lag1', 'Returns_Lag5', 'Volatility',
                       'RSI', 'MACD_Diff', 'Volume_Ratio', 'Price_to_High', 'Price_to_Low']
        
        # Target: Next day's return
        X = self.data[feature_cols]
        y = self.data['Returns'].shift(-1)  # Predict tomorrow's return
        
        # Remove last row (no future return available)
        X = X[:-1]
        y = y[:-1]
        
        # Sequential train-test split (respects time order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n[ML] Training Random Forest...")
        print(f"  • Training samples: {len(X_train)}")
        print(f"  • Test samples: {len(X_test)}")
        
        # Initialize Random Forest with sensible hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=100,        # Number of trees
            max_depth=10,            # Prevent overfitting
            min_samples_split=20,    # Require sufficient data for splits
            min_samples_leaf=10,     # Smooth predictions
            random_state=42,         # Reproducibility
            n_jobs=-1                # Use all CPU cores
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Feature importance analysis
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n[ML] Model Performance:")
        print(f"  • Test R² Score: {metrics['test_r2']:.4f}")
        print(f"  • Test MAE: {metrics['test_mae']:.4f} ({metrics['test_mae']*100:.2f}% avg error)")
        print(f"  • Test RMSE: {metrics['test_rmse']:.4f}")
        
        return metrics
    
    def ml_forecast(self):
        """
        Generate ML-based price forecast for next N days.
        
        FORECASTING METHOD:
        Uses the trained Random Forest to predict future returns iteratively.
        Each prediction becomes an input for the next step (recursive forecasting).
        
        LIMITATIONS:
        - Error compounds over time (long horizons less reliable)
        - Assumes feature relationships remain stable
        - Single point estimate (no uncertainty quantification)
        
        Returns
        -------
        pd.DataFrame with forecasted prices
        """
        print(f"\n[ML] Generating {self.forecast_days}-day forecast...")
        
        # Start from latest known data
        last_row = self.data.iloc[-1].copy()
        current_price = last_row['Close']
        
        forecast_prices = [current_price]
        forecast_dates = [self.data.index[-1]]
        
        # Iteratively predict each day
        for i in range(self.forecast_days):
            # Prepare features for prediction
            features = np.array([[
                last_row['SMA_Ratio'],
                last_row['Returns_Lag1'],
                last_row['Returns_Lag5'],
                last_row['Volatility'],
                last_row['RSI'],
                last_row['MACD_Diff'],
                last_row['Volume_Ratio'],
                last_row['Price_to_High'],
                last_row['Price_to_Low']
            ]])
            
            # Predict next day's return
            predicted_return = self.model.predict(features)[0]
            
            # Convert return to price
            next_price = current_price * (1 + predicted_return)
            
            # Update for next iteration
            current_price = next_price
            next_date = forecast_dates[-1] + timedelta(days=1)
            
            forecast_prices.append(next_price)
            forecast_dates.append(next_date)
            
            # Update lagged features (simplified)
            last_row['Returns_Lag1'] = predicted_return
        
        return pd.DataFrame({
            'Date': forecast_dates[1:],
            'ML_Forecast': forecast_prices[1:]
        })
    
    def monte_carlo_simulation(self):
        """
        Simulate future price paths using Geometric Brownian Motion.
        
        MATHEMATICAL MODEL:
        Geometric Brownian Motion (GBM) is the standard model in finance:
        
        dS_t = μ S_t dt + σ S_t dW_t
        
        Where:
        - S_t = stock price at time t
        - μ = drift (average return per unit time)
        - σ = volatility (standard deviation of returns)
        - dW_t = Wiener process (random shock)
        
        DISCRETE APPROXIMATION:
        S_{t+1} = S_t × exp((μ - σ²/2)Δt + σ√Δt × Z)
        
        Where Z ~ N(0,1) is a standard normal random variable.
        
        INTERPRETATION:
        - Each simulation is one possible future scenario
        - Distribution shows range of outcomes and probabilities
        - 95% confidence interval captures most likely range
        
        Returns
        -------
        np.ndarray : Matrix of simulated paths (n_simulations × forecast_days)
        """
        print(f"\n[MONTE CARLO] Running {self.n_simulations:,} simulations...")
        
        # Estimate GBM parameters from historical data
        returns = self.data['Returns'].dropna()
        mu = returns.mean()  # Drift (average daily return)
        sigma = returns.std()  # Volatility (daily return std dev)
        
        print(f"  • Historical drift (μ): {mu*252:.2%} annualized")
        print(f"  • Historical volatility (σ): {sigma*np.sqrt(252):.2%} annualized")
        
        # Starting price
        S0 = self.data['Close'].iloc[-1]
        
        # Time step (1 day)
        dt = 1
        
        # Initialize simulation matrix
        simulations = np.zeros((self.n_simulations, self.forecast_days))
        
        # Run simulations
        for i in range(self.n_simulations):
            prices = [S0]
            
            for t in range(self.forecast_days):
                # Generate random shock
                Z = np.random.standard_normal()
                
                # GBM formula
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * Z
                
                # Next price
                S_next = prices[-1] * np.exp(drift + diffusion)
                prices.append(S_next)
            
            simulations[i, :] = prices[1:]
        
        # Calculate statistics
        mean_path = simulations.mean(axis=0)
        median_path = np.median(simulations, axis=0)
        std_path = simulations.std(axis=0)
        
        # Confidence intervals (95%)
        ci_lower = np.percentile(simulations, 2.5, axis=0)
        ci_upper = np.percentile(simulations, 97.5, axis=0)
        
        print(f"  • Expected price in {self.forecast_days} days: ${mean_path[-1]:.2f}")
        print(f"  • 95% CI: [${ci_lower[-1]:.2f}, ${ci_upper[-1]:.2f}]")
        
        return simulations, mean_path, median_path, ci_lower, ci_upper
    
    def visualize_predictions(self, ml_forecast, mc_simulations, mc_mean, mc_ci_lower, mc_ci_upper):
        """
        Create interactive visualization using Plotly.
        
        Parameters
        ----------
        ml_forecast : pd.DataFrame
            ML model predictions
        mc_simulations : np.ndarray
            All Monte Carlo paths
        mc_mean : np.ndarray
            Average Monte Carlo path
        mc_ci_lower : np.ndarray
            Lower 95% confidence interval
        mc_ci_upper : np.ndarray
            Upper 95% confidence interval
        """
        # Historical data
        hist_dates = self.data.index[-60:]
        hist_prices = self.data['Close'].iloc[-60:]
        
        # Forecast dates
        forecast_dates = pd.date_range(self.data.index[-1] + timedelta(days=1), 
                                       periods=self.forecast_days, freq='D')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Machine Learning Forecast (Random Forest)',
                f'Monte Carlo Simulation ({self.n_simulations:,} paths)',
                'Monte Carlo with Confidence Interval',
                'Random Forest Feature Importance'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # Plot 1: ML Forecast
        fig.add_trace(
            go.Scatter(x=hist_dates, y=hist_prices, name='Historical',
                      mode='lines', line=dict(color='royalblue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=ml_forecast['Date'], y=ml_forecast['ML_Forecast'],
                      name='ML Forecast', mode='lines',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        fig.add_vline(x=self.data.index[-1], line_dash="dot", line_color="gray",
                     opacity=0.5, row=1, col=1)
        
        # Plot 2: Monte Carlo Paths (sample)
        fig.add_trace(
            go.Scatter(x=hist_dates, y=hist_prices, name='Historical',
                      mode='lines', line=dict(color='royalblue', width=2),
                      showlegend=False),
            row=1, col=2
        )
        
        # Add sample of MC paths
        sample_indices = np.random.choice(self.n_simulations, 50, replace=False)
        for idx in sample_indices:
            fig.add_trace(
                go.Scatter(x=forecast_dates, y=mc_simulations[idx],
                          mode='lines', line=dict(color='gray', width=0.5),
                          opacity=0.2, showlegend=False, hoverinfo='skip'),
                row=1, col=2
            )
        
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=mc_mean, name='Mean Path',
                      mode='lines', line=dict(color='red', width=2)),
            row=1, col=2
        )
        fig.add_vline(x=self.data.index[-1], line_dash="dot", line_color="gray",
                     opacity=0.5, row=1, col=2)
        
        # Plot 3: Monte Carlo with Confidence Intervals
        fig.add_trace(
            go.Scatter(x=hist_dates, y=hist_prices, name='Historical',
                      mode='lines', line=dict(color='royalblue', width=2),
                      showlegend=False),
            row=2, col=1
        )
        
        # Add confidence interval as filled area
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=mc_ci_upper,
                      mode='lines', line=dict(width=0),
                      showlegend=False, hoverinfo='skip'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=mc_ci_lower,
                      mode='lines', line=dict(width=0),
                      fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',
                      name='95% Confidence Interval'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=mc_mean, name='Expected Path',
                      mode='lines', line=dict(color='red', width=2),
                      showlegend=False),
            row=2, col=1
        )
        fig.add_vline(x=self.data.index[-1], line_dash="dot", line_color="gray",
                     opacity=0.5, row=2, col=1)
        
        # Plot 4: Feature Importance
        if self.feature_importance is not None:
            fig.add_trace(
                go.Bar(y=self.feature_importance['feature'],
                      x=self.feature_importance['importance'],
                      orientation='h',
                      marker=dict(color='steelblue'),
                      showlegend=False),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Importance Score", row=2, col=2)
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=2)
        
        fig.update_layout(
            title_text=f'Price Prediction Analysis - {self.ticker}',
            title_font_size=18,
            height=900,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
    
    def generate_report(self, ml_forecast, mc_mean, mc_ci_lower, mc_ci_upper):
        """
        Generate summary statistics comparing both methods.
        
        Parameters
        ----------
        ml_forecast : pd.DataFrame
            ML predictions
        mc_mean : np.ndarray
            Monte Carlo mean path
        mc_ci_lower : np.ndarray
            Lower confidence bound
        mc_ci_upper : np.ndarray
            Upper confidence bound
        
        Returns
        -------
        pd.DataFrame with comparative statistics
        """
        current_price = self.data['Close'].iloc[-1]
        
        report = pd.DataFrame({
            'Method': ['Current Price', 'ML Forecast', 'Monte Carlo Mean', 'MC 95% Lower', 'MC 95% Upper'],
            'Price': [
                current_price,
                ml_forecast['ML_Forecast'].iloc[-1],
                mc_mean[-1],
                mc_ci_lower[-1],
                mc_ci_upper[-1]
            ]
        })
        
        report['Return (%)'] = ((report['Price'] / current_price) - 1) * 100
        
        print(f"\n{'='*80}")
        print(f"FORECAST SUMMARY ({self.forecast_days} days ahead)")
        print(f"{'='*80}")
        print(report.to_string(index=False))
        print(f"{'='*80}\n")
        
        return report


def main():
    """
    Execute complete ML + Monte Carlo forecasting pipeline.
    
    WORKFLOW:
    1. Data Acquisition
    2. Feature Engineering
    3. ML Model Training
    4. ML Forecasting
    5. Monte Carlo Simulation
    6. Visualization & Reporting
    """
    print("="*80)
    print("MACHINE LEARNING PRICE FORECASTING WITH MONTE CARLO SIMULATION")
    print("="*80)
    
    # User inputs
    ticker = input("\nEnter Stock Ticker (default: AAPL): ").strip().upper() or "AAPL"
    forecast_days = input("Forecast horizon in days (default: 30): ").strip()
    forecast_days = int(forecast_days) if forecast_days else 30
    
    # Initialize predictor
    predictor = MLMonteCarloPricePredictor(
        ticker=ticker,
        forecast_days=forecast_days,
        n_simulations=1000
    )
    
    # Step 1: Load data
    predictor.load_data(days=730)
    
    # Step 2: Engineer features
    print("\n[FEATURES] Engineering predictive features...")
    predictor.engineer_features()
    print(f"  • Created {len(predictor.data.columns)} features")
    
    # Step 3: Train ML model
    metrics = predictor.train_ml_model(test_size=0.2)
    
    print("\n[FEATURES] Top 5 Most Important Features:")
    print(predictor.feature_importance.head().to_string(index=False))
    
    # Step 4: ML Forecast
    ml_forecast = predictor.ml_forecast()
    
    # Step 5: Monte Carlo Simulation
    mc_sims, mc_mean, mc_median, mc_lower, mc_upper = predictor.monte_carlo_simulation()
    
    # Step 6: Generate Report
    report = predictor.generate_report(ml_forecast, mc_mean, mc_lower, mc_upper)
    
    # Step 7: Visualize
    print("[VISUALIZATION] Generating plots...")
    predictor.visualize_predictions(ml_forecast, mc_sims, mc_mean, mc_lower, mc_upper)
    
    # Export option
    save = input("\nExport predictions to CSV? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Combine ML and MC results
        forecast_dates = pd.date_range(predictor.data.index[-1] + timedelta(days=1), 
                                       periods=forecast_days, freq='D')
        export_df = pd.DataFrame({
            'Date': forecast_dates,
            'ML_Forecast': ml_forecast['ML_Forecast'].values,
            'MC_Mean': mc_mean,
            'MC_Lower_95': mc_lower,
            'MC_Upper_95': mc_upper
        })
        
        export_df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
