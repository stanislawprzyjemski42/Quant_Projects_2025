# StreamlinedQuantitativeTradingSystem

A modular quantitative research and signal-generation framework combining classical technical analysis, machine learning ensembles, and large language model interpretation. Designed to transform raw market data into structured, decision-oriented intelligence.

## SystemOverview

The system integrates three analytical layers:
1.Deterministic quantitative feature extraction  
2.Supervised machine learning ensemble prediction  
3.LLM-based interpretation and risk contextualization  

The objective is not prediction in isolation, but probabilistic decision support with transparent inputs.

## CoreFeatureEngine

For each ticker,the engine computes 20 normalized quantitative features grouped into four domains.

### MomentumAndTrend
-RelativeStrengthIndex(RSI)  
-MACDSignal  
-5D,10D,20Dpricechange  

### Volatility
-AnnualizedVolatility  
-20DRollingVolatility  
-RealizedVolatility  

### MovingAverages
-20D,50D,200DSMA  
-GoldenCrossDetection  
-DeathCrossDetection  

### StatisticalMoments
-ReturnSkewness  
-ReturnKurtosis  

All features are computed on adjusted close prices and aligned to prevent lookahead bias.

## MachineLearningEnsemble

The system trains three independent classifiers on two years of historical data to predict 5-day forward direction.

-RandomForestClassifier  
-LightGBMClassifier  
-GradientBoostingClassifier  

Each model outputs a probability estimate.The final MLConfidence score is the mean ensemble probability,not a hard vote.

Model training strictly respects chronological ordering.

## CompositeScoringModel

Each ticker receives a composite score from 0 to 100.

|Weight|Component|Description|
|---|---|---|
|40%|TechnicalStrength|Momentum,trend,and moving-average alignment|
|30%|RiskAdjustedReturns|Sharpe and Sortino ratios|
|20%|QualityScore|ROE,DebtEquity,PE|
|10%|MLConfidence|Ensemble bullish probability|

### RecommendationScale
-75+:STRONGBUY  
-65–74:BUY  
-50–64:HOLD  
-40–49:SELL  
-<40:STRONGSELL  

## RiskMetrics

Risk-adjusted performance is computed using standard formulations.

### SharpeRatio
$$Sharpe=\frac{R_p-R_f}{\sigma_p}$$

### SortinoRatio
$$Sortino=\frac{R_p-R_f}{\sigma_{down}}$$

## LLMInsightsLayer

The system integrates GoogleGemini2.0Flash to translate quantitative outputs into structured natural-language insights.

Generated reports include:
-Marketregimeinterpretation  
-Keydriversandrisks  
-Portfolioallocationconsiderations  

The LLM does not generate signals.It interprets existing quantitative evidence.

## Installation

### Requirements
-Python3.9+  
-GoogleGeminiAPIKey  

### Setup
```bash
git clone https://github.com/your-username/streamlined-quant-system.git
cd streamlined-quant-system
pip install yfinance pandas numpy scikit-learn lightgbm colorama python-dotenv google-genai
