# 💼 DCF Valuation System

A comprehensive **Discounted Cash Flow (DCF) Valuation Tool** built with Streamlit for analyzing publicly traded companies.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

This application provides three integrated modules for complete equity valuation:

### 📊 WACC Calculator (Phase 1)
- **Beta Calculation**: OLS regression using 5 years of monthly returns
- **Cost of Equity**: CAPM-based calculation with confidence intervals
- **Cost of Debt**: Credit spread lookup based on firm rating
- **WACC Estimation**: Complete weighted average cost of capital
- **Visualization**: Regression plot showing stock vs. market returns

### 📈 Historical Analysis (Phase 2)
- **Financial Statement Analysis**: 4 years of historical data
- **Growth Metrics**: Revenue and EBIT growth rates
- **Profitability Margins**: Gross, EBIT, and EBITDA margins
- **Working Capital**: NWC analysis and trends
- **Reinvestment**: CapEx, D&A, and reinvestment rates
- **NOPAT**: Net operating profit after tax

### 🎯 DCF Valuation Model (Phase 3)
- **LTM-Based Projections**: Uses last twelve months as starting point
- **Customizable Assumptions**: Revenue growth, margins, reinvestment rates
- **Free Cash Flow**: Complete FCF projections (5-15 year horizon)
- **Terminal Value**: Gordon Growth Model calculation
- **Intrinsic Valuation**: Derives model-implied share price
- **Scenario Analysis**: Bull/Base/Bear cases using different WACC values
- **Price Comparison**: Visual comparison with historical stock prices

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dcf-valuation-system.git
   cd dcf-valuation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**

   Navigate to `http://localhost:8501`

## 📖 Usage

### WACC Calculator
1. Navigate to the **📊 WACC Calculator** tab
2. Enter a ticker symbol (e.g., MSFT, AAPL, GOOGL)
3. Set the risk-free rate (typically 10-year Treasury yield)
4. Adjust parameters as needed (EMRP, tax rate, credit rating)
5. Click **Calculate WACC**

### Historical Analysis
1. Navigate to the **📈 Historical Analysis** tab
2. Enter a ticker symbol
3. Set the effective tax rate
4. Click **Analyze Historical Data**
5. Review growth rates, margins, and reinvestment metrics

### DCF Valuation
1. Navigate to the **🎯 DCF Model** tab
2. Enter a ticker symbol
3. Adjust projection assumptions:
   - Revenue growth rates by period
   - EBIT margin
   - Reinvestment rate
   - Perpetual growth rate
4. Set WACC scenarios (Bull/Base/Bear)
5. Click **Run DCF Model**
6. Review intrinsic valuation and compare to current price

## 📊 Data Sources

- **Yahoo Finance API** (`yfinance`): Company financials, stock prices, market data
- **Damodaran Credit Spreads**: Rating-based spreads (updated January 2025)
- **User Input**: Risk-free rate (manual input)

## 🧮 Methodology

### WACC Calculation
```
WACC = (E/V × Re) + (D/V × Rd × (1-Tc))

Where:
- E/V = Equity weight
- D/V = Debt weight
- Re = Cost of equity (CAPM: Rf + β × EMRP)
- Rd = Cost of debt (Rf + Credit Spread)
- Tc = Corporate tax rate
```

### DCF Valuation
```
Firm Value = PV(FCF1...FCFn) + PV(Terminal Value)
Equity Value = Firm Value - Debt + Cash
Share Price = Equity Value / Shares Outstanding

Where:
- FCF = NOPAT × (1 - Reinvestment Rate)
- Terminal Value = FCFn × (1 + g) / (WACC - g)
```

## 📁 Project Structure

```
dcf-valuation-system/
├── app.py                              # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
├── CLAUDE.md                          # Development history
├── .gitignore                         # Git ignore rules
├── dcf1_wacc_jon_bergamo.py          # Original WACC code
├── dcf2_historical_analysis_jon_bergamo.py  # Original historical analysis
└── dcf3_dcf_model_jon_bergamo.py     # Original DCF model
```

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web framework
- **[yfinance](https://github.com/ranaroussi/yfinance)**: Financial data API
- **[pandas](https://pandas.pydata.org/)**: Data manipulation
- **[numpy](https://numpy.org/)**: Numerical computing
- **[matplotlib](https://matplotlib.org/)**: Plotting and visualization
- **[seaborn](https://seaborn.pydata.org/)**: Statistical visualization
- **[statsmodels](https://www.statsmodels.org/)**: Statistical models (OLS regression)

## ⚙️ Key Assumptions

- **Historical Period**: 5 years of monthly data for beta calculation
- **EMRP Default**: 5% (Equity Market Risk Premium)
- **Tax Rate Default**: 25% marginal tax rate
- **Scale Factor**: Values displayed in millions ($M)
- **Credit Spreads**: Based on US large non-financial firms

## ⚠️ Limitations

- Requires 5+ years of trading history for beta calculation
- Public companies only (requires market data)
- Yahoo Finance API rate limits may apply
- Credit spreads based on Damodaran's US data
- Risk-free rate must be manually updated

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Jon Bergamo**

## 🙏 Acknowledgments

- **Aswath Damodaran** for credit spread data
- **Yahoo Finance** for financial data API
- **Streamlit** for the amazing framework

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Disclaimer**: This tool is for educational and informational purposes only. It should not be considered as financial advice. Always conduct thorough research and consult with a qualified financial advisor before making investment decisions.
