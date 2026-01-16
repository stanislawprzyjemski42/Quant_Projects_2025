# Quantitative Systems & Computational Logic (2025)

A collection of quantitative projects developed between May and December 2025. This repository demonstrates practical implementations of quantitative analysis systems in Python and computational pricing experiments in C++. The emphasis is on applying sound computer-science practices (modularity, defensive programming, documentation) to financial and numerical problems.

---

## Table of contents

- [Project overview](#project-overview)
- [Highlights](#highlights)
- [Technical approach](#technical-approach)
- [Key implementations](#key-implementations)
  - [Quantitative Analysis Systems (Python)](#quantitative-analysis-systems-python)
  - [Exploratory Pricing Tools (C++)](#exploratory-pricing-tools-c)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Python examples](#python-examples)
  - [C++ examples](#c-examples)
- [Repository layout](#repository-layout)
- [Design principles](#design-principles)
- [Testing & validation](#testing--validation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements & resources](#acknowledgements--resources)
- [Contact](#contact)

---

## Project overview

This repository contains two complementary tracks:

1. A system-level track implemented in Python focused on data ingestion, processing pipelines, and statistical modeling for quantitative workflows.
2. A logic-level track implemented in C++ exploring numerical methods and option pricing models (e.g., Black–Scholes), with an emphasis on low-level correctness and performance.

The goal is not to present production trading software, but to demonstrate how to translate financial mathematics and industry tutorials into robust, well-structured code with clear documentation and defensive programming.

---

## Highlights

- Modular Python systems that separate ingestion, transformation, modeling, and reporting.
- C++ engines used to experiment with numerical stability, input validation, and encapsulation of pricing logic.
- Clear documentation and examples so others can reproduce the numerical results and learn implementation patterns.
- Focus on reproducibility, readable code, and safety checks to avoid common runtime errors.

---

## Technical approach

- Python (System Layer): used for building end-to-end analysis tools, pipelines, and experimentation. The code favors readability, package isolation, and automated data validation steps.
- C++ (Logic Layer): used to study memory management, object-oriented design, and performant numerical implementations.
- Research-driven: implementations were informed by industry tutorials, academic references, and open-source examples — adapted into original implementations with tests and input sanitization.

---

## Key implementations

### Quantitative Analysis Systems (Python)
Files: `basic_quant_system.py`, `improved_quant.py`

- Purpose: Provide a modular framework for processing market or simulated data, running statistical models, and outputting results.
- Features:
  - Clear separation of data ingestion, preprocessing, model implementation, and reporting.
  - Defensive checks on incoming data (types, missing values, ranges).
  - Example model components: moving averages, momentum indicators, simple linear regressions for factor analysis.
- Example use-cases: backtesting small strategies, rapid prototyping of signals, batch processing of CSV time-series.

### Exploratory Pricing Tools (C++)
Files: `option_pricing_engine2.cpp`, `cpp-option-pricer.cpp`

- Purpose: Implement canonical option pricing formulas and small engines for experiments.
- Focus areas:
  - Black–Scholes closed-form pricing and Greeks.
  - Input validation: non-numeric inputs, negative or zero volatility/price handling, and informative error messages.
  - Encapsulation: pricing logic wrapped in classes to facilitate reuse and testing.
- Intended for educational experiments and performance comparisons.

---

## Getting started

### Prerequisites

- Python 3.8+ (recommended: 3.10 or later)
- Common Python packages: numpy, pandas, scipy (install via pip)
- C++ compiler supporting C++11 or later (g++, clang++)
- Build tools (optional): make or CMake for more complex builds

### Python examples

Install dependencies (recommended in a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows PowerShell
pip install -U pip
pip install numpy pandas scipy
```

Run a simple script:

```bash
python basic_quant_system.py
# or
python improved_quant.py --input data/sample_prices.csv --output results/report.csv
```

Each script prints help when run with `-h` or `--help`.

### C++ examples

Compile and run:

```bash
# compile
g++ -std=c++11 -O2 option_pricing_engine2.cpp -o option_pricer

# run with arguments (example)
./option_pricer --type call --S 100 --K 100 --r 0.01 --sigma 0.2 --T 0.5
```

The programs provide usage instructions when invoked without required arguments or with `--help`.

---

## Repository layout

- basic_quant_system.py — baseline Python system for ingestion + analysis
- improved_quant.py — refactored Python system with modular components
- option_pricing_engine2.cpp — C++ experiment implementing Black–Scholes and utilities
- cpp-option-pricer.cpp — additional C++ implementation and testing harness
- data/ — sample data (CSV) used for demonstrations (if present)
- results/ — example outputs produced by scripts (if present)
- README.md — this file
- LICENSE — repository license (if present)

---

## Design principles

- Separation of Concerns: data ingestion, processing, modeling, and reporting are split into separate modules/functions.
- Defensive Programming: validate inputs early, handle edge cases, and surface useful error messages rather than crashing.
- Reproducibility: deterministic behavior where possible, seed control for simulations, and clear dependency lists.
- Readability: prioritize clear variable names, docstrings/comments, and simple control flow.

---

## Testing & validation

- Numerical outputs are validated against known references (e.g., Black–Scholes analytical values).
- Unit and regression tests should be added for critical functions (recommended: pytest for Python).
- For the C++ code, compile-time warnings are treated seriously; address all warnings and add small runtime tests to check boundary conditions.

---

## Contributing

Contributions, corrections, and improvements are welcome. Recommended workflow:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/meaningful-name`.
3. Make changes and include tests where appropriate.
4. Open a pull request with a clear description of your changes and rationale.

Please follow the code style used in the repository and include tests for substantive changes.

---

## License

This repository does not include a LICENSE file by default. If you want to publish this project, consider a permissive license such as MIT or BSD. Contact the repository owner to confirm licensing preferences.

---

## Acknowledgements & resources

This work was informed by tutorials, open-source documentation, and reference materials in quantitative finance and numerical computing. Notable topics used as background include:

- Black–Scholes option pricing
- Numerical stability considerations in floating-point arithmetic
- Defensive programming patterns in C++ and Python

Please consult standard texts and the referenced online materials for deeper theoretical background.

---

## Contact

Repository owner: stanislawprzyjemski42

For questions, corrections, or collaboration requests, please open an issue or contact the owner via their GitHub profile.
