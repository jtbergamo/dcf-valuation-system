# Financial Data Analytics - WACC Calculator Web App

## Project Overview

This project converts Python-based financial analysis tools (originally Jupyter notebooks) into an interactive web application using Streamlit. The goal is to create a comprehensive DCF (Discounted Cash Flow) valuation system accessible through a web interface.

## Original Files

The project started with three Python files containing financial analysis code:

### 1. `dcf1_wacc_jon_bergamo.py` - WACC Calculator
- Calculates Weighted Average Cost of Capital (WACC)
- Fetches company financial data from Yahoo Finance
- Computes equity and debt weights from market cap and total debt
- Calculates beta using OLS regression on 5 years of monthly returns
- Computes cost of equity using CAPM: `k_E = r_f + β * EMRP`
- Determines cost of debt using credit spread lookup tables
- Outputs WACC with 95% confidence intervals
- Originally used FRED API for risk-free rate data

### 2. `dcf2_historical_analysis_jon_bergamo.py` - Historical Financial Analysis
- Analyzes historical financial statements (4+ years)
- Computes growth rates for revenue and EBIT
- Calculates key margins: Gross Margin, EBIT Margin, EBITDA Margin
- Analyzes working capital (NWC) and changes
- Computes reinvestment metrics (CapEx, D&A, NWC changes)
- Calculates NOPAT and reinvestment rates
- **Status: Not yet implemented in web app**

### 3. `dcf3_dcf_model_jon_bergamo.py` - Complete DCF Valuation Model
- Uses Last Twelve Months (LTM) data as starting point
- Projects future financials based on assumptions
- Calculates terminal value using Gordon Growth Model
- Computes present value of cash flows
- Derives model-implied share price
- Includes sensitivity analysis across different WACC scenarios
- Visualizes results with historical price comparisons
- **Status: Not yet implemented in web app**

## Current Implementation: WACC Calculator MVP

### What We Built

A single-page Streamlit web application (`app.py`) that implements the WACC calculator functionality.

### Features Implemented

#### 1. Interactive Input Sidebar
- **Ticker Symbol**: Enter any publicly traded company (e.g., MSFT, AAPL, TSLA)
- **Risk-Free Rate**: Manual input (%) - removed FRED API dependency
- **Equity Market Risk Premium (EMRP)**: Default 5%, adjustable
- **Marginal Tax Rate**: Default 25%, adjustable
- **Firm Credit Rating**: Dropdown with all Moody's/S&P ratings
- **Market Index**: Default S&P 500 (^GSPC), customizable

#### 2. Company Overview Section
- Market capitalization
- Total debt
- Equity weight (w_E)
- Debt weight (w_D)

#### 3. Cost of Equity Calculation
- Beta calculation using OLS regression
- 5 years of monthly return data
- 95% confidence intervals for beta
- CAPM-based cost of equity
- Displays risk-free rate and EMRP used

#### 4. Cost of Debt Calculation
- Credit spread lookup based on firm rating
- Updated credit spread table (January 2025)
- Uses Damodaran's credit spread data

#### 5. WACC Calculation
- Complete WACC formula: `WACC = w_E * k_E + w_D * k_D * (1-t)`
- Point estimate with 95% confidence intervals
- Shows upper and lower bounds based on beta uncertainty
- Formatted summary table

#### 6. Visual Representation
- **Scatter plot** of monthly returns (stock vs. market index)
- **Best-fit regression line** showing beta calculation
- Displays R-squared value
- Clean, professional visualization using matplotlib/seaborn

### Technical Implementation

#### Dependencies
All listed in `requirements.txt`:
- `streamlit` - Web framework
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Enhanced visualizations
- `yfinance` - Yahoo Finance data API
- `statsmodels` - Statistical modeling (OLS regression)

#### Key Functions

**`get_credit_spread(rating, credit_spreads)`**
- Looks up credit spread from rating table
- Returns spread as decimal

**`get_stock_data(ticker_symbol, index_symbol)`**
- Downloads historical price data
- Cached with `@st.cache_data` for performance
- 5 years of monthly data

**`get_company_info(ticker_symbol)`**
- Fetches company fundamentals from Yahoo Finance
- Returns market cap, debt, company name
- Cached for performance

**`calculate_wacc(...)`**
- Main calculation engine
- Performs OLS regression for beta
- Computes all cost components
- Returns comprehensive results dictionary

## How to Run the App

### Initial Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the app
streamlit run app.py
```

### Access the App
- **Local URL**: http://localhost:8501
- **Network URL**: http://10.71.92.183:8501 (accessible from other devices on network)

### Current Status
The app is configured and running with:
- Streamlit credentials set up (email prompt disabled)
- Headless mode enabled for background operation
- Running on port 8501

## Project Structure

```
/Users/jonbergamo/Desktop/Fin Data Analytics/
├── app.py                              # Main Streamlit application (MVP)
├── requirements.txt                     # Python dependencies
├── claude.md                           # This documentation file
├── dcf1_wacc_jon_bergamo.py           # Original WACC calculator
├── dcf2_historical_analysis_jon_bergamo.py  # Original historical analysis
└── dcf3_dcf_model_jon_bergamo.py      # Original DCF model
```

## What Changed from Original Code

### Removed
- Google Colab specific code (`!pip install`, `userdata.get()`)
- FRED API integration for risk-free rate (now manual input)
- `display()` function calls (replaced with Streamlit equivalents)

### Added
- Streamlit UI components (sidebar, metrics, dataframes)
- Session state management for results
- Caching decorators for API calls
- Interactive input widgets
- Professional formatting and layout
- Error handling for API failures

### Modified
- Regression plot moved to end of results
- Enhanced visualization with R-squared display
- Formatted metrics with `st.metric()` cards
- Added confidence interval displays throughout

## Future Enhancements

### Phase 2: Historical Analysis Module (Planned)
- Add multi-page Streamlit app structure
- Implement historical financial statement analysis
- Show trends in growth rates and margins
- Interactive charts for historical metrics
- Timeline visualization

### Phase 3: DCF Valuation Model (Planned)
- Add projection assumptions interface
- Create editable projection tables with sliders
- Implement terminal value calculation
- Show firm value and share price derivation
- Comparison with current market price
- Sensitivity analysis dashboard
- Export results to Excel/CSV

### Potential Additional Features
- Save/load analysis sessions
- Compare multiple companies side-by-side
- Export professional PDF reports
- Add more visualization options
- Industry benchmarking
- Real-time data refresh
- Email alerts for target prices
- Integration with portfolio tracking

## Technical Notes

### Data Sources
- **Yahoo Finance** (`yfinance`): Company financials, stock prices, market data
- **Credit Spreads**: Damodaran's rating-based spreads (updated January 2025)
- **Risk-Free Rate**: Manual input (typically 10-year Treasury yield)

### Assumptions
- **Historical Period**: 5 years of monthly data for beta calculation
- **EMRP**: Default 5% (Equity Market Risk Premium)
- **Tax Rate**: Default 25% marginal tax rate
- **Scale Factor**: 1,000,000 (values shown in millions)

### Known Limitations
- Beta calculation requires 5 years of data (newer companies may not work)
- Yahoo Finance API rate limits may apply
- Credit spreads are based on US large non-financial firms
- Risk-free rate must be manually updated

## Development Workflow

### Session History (2025-12-02)

1. **Planning Phase**
   - Analyzed three Python files to understand functionality
   - Created implementation plan for MVP
   - Decided to start with WACC calculator only

2. **Implementation**
   - Created `app.py` with all WACC functionality
   - Removed Google Colab dependencies
   - Added Streamlit UI components
   - Moved regression plot to end per user request

3. **Deployment**
   - Created `requirements.txt`
   - Installed all dependencies via pip3
   - Configured Streamlit settings
   - Successfully launched app on localhost:8501

4. **Documentation**
   - Created this `claude.md` file for future reference

## Commands Reference

### Start the App
```bash
streamlit run app.py
```

### Start in Background/Headless
```bash
streamlit run app.py --server.headless=true
```

### Install/Update Dependencies
```bash
pip3 install -r requirements.txt
```

### Update pip (if needed)
```bash
python3 -m pip install --upgrade pip
```

### Find Streamlit Path
```bash
which streamlit
# or
/Users/jonbergamo/Library/Python/3.9/bin/streamlit
```

## Contact & Notes

- **Developer**: Jon Bergamo
- **Project Type**: Financial Analytics / DCF Valuation
- **Framework**: Streamlit
- **Data Source**: Yahoo Finance
- **Current Phase**: MVP (WACC Calculator only)
- **Last Updated**: 2025-12-02

---

## Quick Start for Next Session

1. Navigate to project directory:
   ```bash
   cd "/Users/jonbergamo/Desktop/Fin Data Analytics"
   ```

2. Start the app:
   ```bash
   streamlit run app.py
   ```

3. Open browser to: http://localhost:8501

4. To continue development:
   - Phase 2: Add Historical Analysis module
   - Phase 3: Add DCF Valuation module
   - Consider multi-page app structure when adding modules
