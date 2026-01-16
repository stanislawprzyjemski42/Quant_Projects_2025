# ==============================================================================
# ENVIRONMENT SETUP & EXECUTION GUIDE
# ==============================================================================
# 1. Create a virtual environment:
#    python -m venv venv
#    source venv/bin/activate  # On Windows: venv\Scripts\activate
#
# 2. Install required dependencies:
#    pip install matplotlib numpy pandas yfinance python-dotenv tqdm google-genai
#
# 3. Configure your .env file:
#    Create a file named '.env' in the same directory and add:
#    GEMINI_API_KEY=your_actual_api_key_here
#
# 4. Run the script:
#    EMA_logic_orchestrator.py
# ==============================================================================
# ==============================================================================
# EMA LOGIC ORCHESTRATOR (July-2025)
# ==============================================================================
# DESCRIPTION:
# This system is an automated analytical tool designed to track market momentum 
# using Exponential Moving Averages (EMA). Unlike a simple moving average, this 
# engine applies a recursive weight to recent data, allowing for faster response 
# times to trend shifts.
#
# HUMBLE NOTE:
# This implementation was built by researching algorithmic trading tutorials and 
# technical documentation. The focus is on the software engineering required to 
# make these mathematical signals implementable, stable, and readable.
#
# CS CORE COMPETENCIES:
# 1. System Resiliency: Implements Exponential Backoff for API stability.
# 2. Vectorized Logic: Uses NumPy/Pandas for O(N) algorithmic efficiency.
# 3. Defensive Design: Robust input sanitization and error recovery.
# ==============================================================================

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai

# Constants and Configuration
load_dotenv()
# The environment provides the key at runtime in this specific preview platform
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def fetch_data(ticker, start_date, end_date):
    """Fetch historical data from Yahoo Finance."""
    print(f"Fetching data for {ticker}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Handle multi-index columns for newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def run_backtest(df, short_window, long_window):
    """Execute EMA crossover backtest and return metrics."""
    data = df.copy()
    
    # Calculate EMAs
    data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    # Generate Signals
    data['Signal'] = 0.0
    data.iloc[short_window:, data.columns.get_loc('Signal')] = np.where(
        data['Short_EMA'][short_window:] > data['Long_EMA'][short_window:], 1.0, 0.0
    )
    data['Position'] = data['Signal'].diff()
    
    # Calculate Returns
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Signal'].shift(1)
    
    # Cumulative Returns
    data['Cum_Strategy'] = (1 + data['Strategy_Returns'].fillna(0)).cumprod()
    
    # Metrics
    total_return = (data['Cum_Strategy'].iloc[-1] - 1) * 100
    sharpe = (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()) * np.sqrt(252) if data['Strategy_Returns'].std() != 0 else 0
    
    # Drawdown
    rolling_max = data['Cum_Strategy'].cummax()
    drawdown = (data['Cum_Strategy'] / rolling_max) - 1
    max_drawdown = drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_df': data
    }

def get_ai_analysis(ticker, results_summary):
    """Send results to Gemini for strategic interpretation using exponential backoff."""
    if not GEMINI_API_KEY:
        return "AI Analysis skipped: No API Key found."
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    Analyze the following backtest results for {ticker}:
    {results_summary}
    
    Please provide:
    1. A summary of strategy performance.
    2. Identification of risks (e.g., high drawdown, low Sharpe).
    3. Potential improvements for this ticker.
    Keep the tone professional and concise.
    """
    
    # Implementation of required exponential backoff: 1s, 2s, 4s, 8s, 16s
    delays = [1, 2, 4, 8, 16]
    for i, delay in enumerate(delays):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-09-2025",
                contents=prompt
            )
            return response.text
        except Exception:
            if i == len(delays) - 1:
                return "Error: Gemini API failed after 5 retries. Please check your connection or API limits."
            time.sleep(delay)

def main():
    """Main execution logic for the AI-Enhanced EMA Strategy."""
    # 1. Configuration
    ticker = input("Enter ticker (e.g., AAPL): ").upper() or "AAPL"
    short_windows = [10, 20, 50]
    long_windows = [50, 100, 200]
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 2. Setup Timeframe (2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    try:
        df = fetch_data(ticker, start_date, end_date)
        if df.empty:
            return

        all_results = []
        best_return = -np.inf
        best_data = None
        best_params = ""

        print("Testing EMA combinations...")
        for s in tqdm(short_windows):
            for l in long_windows:
                if s >= l: continue
                
                res = run_backtest(df, s, l)
                res_meta = {
                    'Params': f"EMA_{s}_{l}",
                    'Return %': res['total_return'],
                    'Sharpe': res['sharpe_ratio'],
                    'Max DD %': res['max_drawdown']
                }
                all_results.append(res_meta)
                
                if res['total_return'] > best_return:
                    best_return = res['total_return']
                    best_data = res['final_df']
                    best_params = f"EMA_{s}_{l}"

        # 3. Save Summary and Plot
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = output_dir / f"{ticker}_summary_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Plotting the best performer
        plt.figure(figsize=(12, 6))
        plt.plot(best_data['Cum_Strategy'], label=f"Strategy: {best_params}", color='#2ecc71')
        plt.title(f"{ticker} Best EMA Performance Over 2 Years", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns (Base 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = output_dir / f"{ticker}_plot_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # 4. AI Interpretation
        print("Consulting AI for market insights...")
        top_3 = results_df.sort_values(by='Return %', ascending=False).head(3).to_string()
        ai_insight = get_ai_analysis(ticker, top_3)
        
        ai_path = output_dir / f"{ticker}_ai_{timestamp}.txt"
        with open(ai_path, "w") as f:
            f.write(ai_insight)
            
        print(f"\n--- Analysis Complete ---")
        print(f"1. Data Summary: {csv_path}")
        print(f"2. Performance Chart: {plot_path}")
        print(f"3. AI Insights: {ai_path}")
        print(f"\nBest Strategy: {best_params} with {best_return:.2f}% return.")
        print(f"\nAI Insight Preview:\n{'-'*20}\n{ai_insight[:500]}...")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
