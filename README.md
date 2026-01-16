# Quantitative Systems & Computational Logic (2025) 🚀

Welcome — this repository is a personal collection of some quantitative finance projects and experiments built between May and December 2025 made in January mainly to showcase my quant projects.  

My goal: learn by doing — implement math, build end-to-end Python systems, experiment with C++ numerical engines, and improve programming skills to analize the market better before I make investment decisons.  
This repo showcases my projects; it’s educational and experimental, not production trading software.

---

## Quick navigation ✨

- Overview
- Files & what they do
- Setup & requirements
- How to run
- Data & environment notes
- Disclaimers (please read) ⚠️
- Acknowledgements & references 
- Contact

---

## Short overview 🧭

This repository contains exploratory and educational implementations in Python and C++ focused on:
- quantitative feature engineering,
- lightweight ML experiments for price forecasts,
- end-to-end analysis pipelines,
- option pricing experiments in C++ for numerical accuracy and performance.

These are learning artifacts and experiments — not investment recommendations or production-ready trading systems.

---

## Files (what to look at) 📁

All files present in this repo and short descriptions:

- `_basic_quant_system.py` 🧭  
  Streamlined quantitative analysis system. Implements 20 essential features (returns, vol, RSI, MACD, MAs, Sharpe/Sortino, momentum), an ML ensemble (RandomForest, LightGBM, GradientBoosting), an interactive CLI, composite scoring and optional LLM/Gemini insights.

- `_machine_learning_price_forecasting.py` 🤖  
  ML-focused price forecasting experiments: feature pipelines, preprocessing and model experiments intended for backtesting/idea exploration.

- `_statistical_alpha_factor_engineering_system.py` 🧪  
  Tools for factor engineering, statistical alpha construction, cross-sectional tests and evaluation utilities.

- `_the_full_quantatative_program.py` 🧩  
  Larger integrated script that ties ingestion, analytics, ML, and reporting into a more feature-rich pipeline — useful to see how components compose.

- `_EMA_logic_orchestrator.py` ⚙️  
  Example orchestration around EMA-based signals and execution-flow demo.

- `_option_pricing_engine2.cpp` 🔧  
  C++ experiment implementing Black–Scholes formulas, Greeks and small harnesses to test numerical stability.

- `_cpp_option_pricer.cpp` 🔧  
  Additional C++ testing/pricing harness — used to explore performance and input validation.

- `README.md` — this file 📘

> Note: data/ and results/ may be present for sample CSVs or outputs (if included).

---

## Setup & requirements ✅

Recommended environment:
- Python 3.8+ (3.10+ recommended)
- pip, and use a virtual environment (venv)

Python packages commonly used:
- numpy
- pandas
- scikit-learn
- lightgbm
- yfinance
- colorama
- python-dotenv
- google-genai (optional — only if you want Gemini/LLM integration)

C++:
- g++ or clang++ supporting C++11 or later (for compiling the example C++ code)

Example install (use a venv):
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\\.venv\\Scripts\\Activate.ps1

pip install -U pip
pip install numpy pandas scikit-learn lightgbm yfinance colorama python-dotenv
# optionally:
# pip install google-genai
```

Notes:
- LightGBM may require build tools (CMake, a C++ compiler) on some platforms.
- The code attempts to be defensive but please test in your environment before running anything that touches real accounts or sensitive systems.

---

## How to run (examples) ▶️

Run the interactive basic quant system:
```bash
python _basic_quant_system.py
# or analyze tickers directly:
python _basic_quant_system.py AAPL MSFT GOOGL
```

Run the full integrated program:
```bash
python _the_full_quantatative_program.py
```

Run ML experiments:
```bash
python _machine_learning_price_forecasting.py
```

Compile and run C++ example:
```bash
# compile
g++ -std=c++11 -O2 _option_pricing_engine2.cpp -o option_pricer

# example run
./option_pricer --type call --S 100 --K 100 --r 0.01 --sigma 0.2 --T 0.5
```

Most scripts print usage or help when run with `-h` or without required arguments.

---

## Data & environment notes 🗂️

- By default Python scripts use `yfinance` to fetch historical data. You can adapt ingestion to local CSVs if preferred. In some projects FRED and AphaVantage API keys are used to get open source economic data to improve the AI systems market data contex.
- Optional LLM insights: set `GEMINI_API_KEY` (or relevant API key) in a `.env` file or environment variable to enable Gemini/LLM integration:
```
GEMINI_API_KEY=your_api_key_here
```
- Many scripts include validation checks, but please verify inputs and add tests before using any code for real decisions.

---

## Disclaimers — humble, clear, and important ⚠️

- Public use: Everyone is free to use this code for learning and experimentation. Use, adapt, and fork as you like, but please respect third-party licenses for dependencies and referenced material.
- Personal intent: I use these quant projects personally to make more data-driven decisions in my own investments. That is my personal practice and not an encouragement for others to invest or to rely on this code for trading.
- Not financial advice: this repository is for educational/demonstration purposes only. It is NOT investment advice.
- No guarantees: these systems do NOT guarantee returns or any specific performance. Past results and backtests are not predictive of future outcomes.
- Use at your own risk: I am not responsible for any losses. If you plan to use or adapt this code for live trading, do rigorous testing, risk-management checks, and consult qualified professionals.

Short legal clarity: It does not guarantee profits, returns, or accuracy — use responsibly.

---

## Acknowledgements & references 

I used many tutorials, community resources, and open-source examples while building these projects (notably Quantecon and other community tutorials and docs). I acknowledge that those materials were important in helping me create these projects. I accept the use of those resources as background and have adapted the ideas into original code here; nothing in this repo is intended to misrepresent others' work. This is an acknowledgement of influence rather than a request for thanks — please consider the original resources for license and reuse terms if you plan to copy their material directly.

---

## Style, testing & contribution notes 🛠️

- Design principles: separation of concerns, defensive programming, reproducibility, and readability.
- Tests: add unit/regression tests (pytest suggested) before trusting outputs for anything critical.
- Contributions: forks, PRs and issues are welcome. If you contribute, include tests and clear descriptions.

Contribution workflow (recommended):
1. Fork the repo
2. git checkout -b feat/meaningful-name
3. Add changes + tests
4. Open a PR with a clear description

---

## License & reuse ⚖️

This repo currently does not include a LICENSE file. Everyone can use this code for learning and experimentation, but please respect and follow the licenses of any third-party libraries or source materials referenced. If you want an explicit permission grant for reuse, consider adding a LICENSE (e.g., MIT) or contact the repository owner.

---

## Contact 📬

Owner: @stanislawprzyjemski42  
If you have suggestions, corrections, or collaboration ideas, please open an issue or a pull request. I appreciate constructive feedback.

---
