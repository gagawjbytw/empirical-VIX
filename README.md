# Empirical VIX Option Pricing

This project implements and compares two different approaches for VIX option pricing:
1. Legendre polynomial model
2. 3/2 stochastic volatility model

## Overview

The project provides a comprehensive implementation for calibrating and pricing VIX options using both the Legendre polynomial approach and the 3/2 stochastic volatility model. It includes functionality for:

- Historical VIX data analysis
- Model calibration
- Option pricing
- Error analysis and comparison
- Visualization of results

## Requirements

```
numpy
pandas
matplotlib
scipy
openpyxl  # for Excel file handling
```

## Project Structure

- `vix_option_pricing.py`: Main implementation file containing all models and processing logic
- `VIXdata_from_1990_01_02.xlsx`: Historical VIX data for model calibration
- Option contract data folders:
  - `20241030/`
  - `20241120/`
  - `20241218/`
  - `20241224/`

## Key Components

### 1. Legendre Model

The Legendre model implementation includes:
- Polynomial fitting of historical VIX data
- Legendre-Gauss quadrature for numerical integration
- Inner product computations
- Model calibration and pricing functions

### 2. 3/2 Stochastic Volatility Model

The 3/2 model implementation features:
- Goard & Mazur (2013) VIX call pricing formula
- Parameter calibration (α, β, κ)
- Numerical integration using modified Bessel functions

## Usage

1. Ensure all required dependencies are installed
2. Place your VIX option data in appropriate folders
3. Run the main script:

```bash
python vix_option_pricing.py
```

The script will:
- Process each option contract file
- Perform model calibration
- Calculate option prices
- Generate comparison metrics
- Create visualizations
- Save results in the `output/` directory

## Output

For each processed option contract, the program generates:
- Excel file with detailed results including:
  - Market prices
  - Model prices (both Legendre and 3/2)
  - Absolute and relative errors
  - Time series data
- Visualization plot comparing market prices with model prices

## Error Metrics

The program calculates and reports:
- Average absolute error
- Average relative error
- Filtered statistics for the 3/2 model (excluding NaN/Inf values)

## Notes

- The 3/2 model may produce NaN/Inf values for certain parameter combinations
- Model parameters are bounded within reasonable ranges during calibration
- The Legendre model typically provides more stable results compared to the 3/2 model

## References

- Goard & Mazur (2013) for the 3/2 stochastic volatility model
- Legendre polynomial approach for empirical VIX modeling 
