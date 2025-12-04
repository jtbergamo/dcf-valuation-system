import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
import time
from functools import wraps

# Page configuration
st.set_page_config(page_title="DCF Valuation System", page_icon="💼", layout="wide")

# Format setting
pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
sns.set_style("whitegrid")

# ==================== HELPER FUNCTIONS ====================

# Retry decorator for handling rate limits
def retry_with_backoff(max_retries=3, initial_delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    time.sleep(delay * attempt)  # Exponential backoff
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = delay * (attempt + 1)
                            st.warning(f"Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                    raise e
            return None
        return wrapper
    return decorator

# WACC Calculator Functions
credit_spreads = [
    {"GreaterThan": -100000, "LessThan": 0.199999, "Rating": "D2/D", "Spread": 19.00},
    {"GreaterThan": 0.2, "LessThan": 0.649999, "Rating": "C2/C", "Spread": 15.50},
    {"GreaterThan": 0.65, "LessThan": 0.799999, "Rating": "Ca2/CC", "Spread": 10.10},
    {"GreaterThan": 0.8, "LessThan": 1.249999, "Rating": "Caa/CCC", "Spread": 7.28},
    {"GreaterThan": 1.25, "LessThan": 1.499999, "Rating": "B3/B-", "Spread": 4.42},
    {"GreaterThan": 1.5, "LessThan": 1.749999, "Rating": "B2/B", "Spread": 3.00},
    {"GreaterThan": 1.75, "LessThan": 1.999999, "Rating": "B1/B+", "Spread": 2.61},
    {"GreaterThan": 2, "LessThan": 2.2499999, "Rating": "Ba2/BB", "Spread": 1.83},
    {"GreaterThan": 2.25, "LessThan": 2.49999, "Rating": "Ba1/BB+", "Spread": 1.55},
    {"GreaterThan": 2.5, "LessThan": 2.999999, "Rating": "Baa2/BBB", "Spread": 1.20},
    {"GreaterThan": 3, "LessThan": 4.249999, "Rating": "A3/A-", "Spread": 0.95},
    {"GreaterThan": 4.25, "LessThan": 5.499999, "Rating": "A2/A", "Spread": 0.85},
    {"GreaterThan": 5.5, "LessThan": 6.499999, "Rating": "A1/A+", "Spread": 0.77},
    {"GreaterThan": 6.5, "LessThan": 8.499999, "Rating": "Aa2/AA", "Spread": 0.60},
    {"GreaterThan": 8.5, "LessThan": 100000, "Rating": "Aaa/AAA", "Spread": 0.45}
]

def get_credit_spread(rating, credit_spreads):
    for entry in credit_spreads:
        if entry["Rating"] == rating:
            return entry["Spread"]/100
    st.warning(f"Warning: Rating '{rating}' not found in credit spread table.")
    return float('nan')

@st.cache_data(ttl=3600)  # Cache for 1 hour
@retry_with_backoff(max_retries=3, initial_delay=2)
def get_stock_data(ticker_symbol, index_symbol, period='5y', interval='1mo'):
    try:
        time.sleep(0.5)  # Small delay to avoid rapid-fire requests
        stock_data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        time.sleep(0.5)  # Delay between requests
        index_data = yf.download(index_symbol, period=period, interval=interval, progress=False)
        return stock_data, index_data
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "429" in error_msg or "too many" in error_msg.lower():
            st.error("⚠️ Yahoo Finance rate limit reached. Please wait a few minutes and try again.")
        else:
            st.error(f"Error downloading data: {e}")
        return None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
@retry_with_backoff(max_retries=3, initial_delay=2)
def get_company_info(ticker_symbol):
    try:
        time.sleep(0.5)  # Small delay to avoid rapid-fire requests
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return {
            'name': info.get('longName', ticker_symbol),
            'market_cap': info.get('marketCap', 0),
            'total_debt': info.get('totalDebt', 0)
        }
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "429" in error_msg or "too many" in error_msg.lower():
            st.error("⚠️ Yahoo Finance rate limit reached. Please wait a few minutes and try again, or try a different ticker.")
        else:
            st.error(f"Error getting company info: {e}")
        return None

def calculate_wacc(ticker_symbol, index_symbol, rf, emrp, firm_rating, marg_tax_rate, scale_factor):
    company_info = get_company_info(ticker_symbol)
    if company_info is None:
        return None

    company_name = company_info['name']
    market_cap = company_info['market_cap'] / scale_factor
    total_debt = company_info['total_debt'] / scale_factor

    w_E = market_cap / (market_cap + total_debt)
    w_D = total_debt / (market_cap + total_debt)

    stock_data, index_data = get_stock_data(ticker_symbol, index_symbol)
    if stock_data is None or index_data is None:
        return None

    stock_returns = stock_data['Close'].pct_change().dropna() - rf
    index_returns = index_data['Close'].pct_change().dropna() - rf

    X = sm.add_constant(index_returns)
    model = sm.OLS(stock_returns, X)
    results = model.fit()

    beta = results.params[index_symbol]
    conf_int = results.conf_int(alpha=0.05)
    beta_lower = conf_int.loc[index_symbol, 0]
    beta_upper = conf_int.loc[index_symbol, 1]

    cost_of_equity = rf + beta * emrp
    cost_of_equity_lower = rf + beta_lower * emrp
    cost_of_equity_upper = rf + beta_upper * emrp

    credit_spread = get_credit_spread(firm_rating, credit_spreads)
    cost_of_debt = rf + credit_spread

    wacc = w_E * cost_of_equity + w_D * cost_of_debt * (1 - marg_tax_rate)
    wacc_lower = w_E * cost_of_equity_lower + w_D * cost_of_debt * (1 - marg_tax_rate)
    wacc_upper = w_E * cost_of_equity_upper + w_D * cost_of_debt * (1 - marg_tax_rate)

    return {
        'company_name': company_name,
        'market_cap': market_cap,
        'total_debt': total_debt,
        'w_E': w_E,
        'w_D': w_D,
        'beta': beta,
        'beta_lower': beta_lower,
        'beta_upper': beta_upper,
        'cost_of_equity': cost_of_equity,
        'cost_of_equity_lower': cost_of_equity_lower,
        'cost_of_equity_upper': cost_of_equity_upper,
        'credit_spread': credit_spread,
        'cost_of_debt': cost_of_debt,
        'wacc': wacc,
        'wacc_lower': wacc_lower,
        'wacc_upper': wacc_upper,
        'stock_returns': stock_returns,
        'index_returns': index_returns,
        'regression_results': results
    }

# Historical Analysis Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
@retry_with_backoff(max_retries=3, initial_delay=2)
def get_financial_data(ticker_symbol):
    try:
        time.sleep(0.5)  # Small delay to avoid rapid-fire requests
        ticker = yf.Ticker(ticker_symbol)
        income_stmt = ticker.financials.T.sort_index().iloc[-4:]
        balance_sheet = ticker.balance_sheet.T.sort_index().iloc[-4:]
        cash_flow = ticker.cashflow.T.sort_index().iloc[-4:]
        info = ticker.info
        company_name = info.get('longName', ticker_symbol)
        return income_stmt, balance_sheet, cash_flow, company_name
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "429" in error_msg or "too many" in error_msg.lower():
            st.error("⚠️ Yahoo Finance rate limit reached. Please wait a few minutes and try again.")
        else:
            st.error(f"Error fetching data: {e}")
        return None, None, None, None

def calculate_historical_metrics(income_stmt, balance_sheet, cash_flow, eff_tax_rate, scale_factor):
    total_revenue = income_stmt['Total Revenue']
    gross_profit = income_stmt['Gross Profit']
    ebit = income_stmt.get('EBIT')
    ebitda = income_stmt.get('EBITDA')

    gross_margin = gross_profit / total_revenue
    ebit_margin = ebit / total_revenue
    ebitda_margin = ebitda / total_revenue

    revenue_growth = total_revenue.pct_change()
    ebit_growth = ebit.pct_change()

    if 'Tax Rate For Calcs' in income_stmt.columns:
        tax_rate_series = income_stmt['Tax Rate For Calcs']
    else:
        tax_rate_series = pd.Series([eff_tax_rate] * len(income_stmt), index=income_stmt.index)

    nopat = ebit * (1 - tax_rate_series)

    current_assets_adj = balance_sheet['Current Assets'] - balance_sheet['Cash Cash Equivalents And Short Term Investments']
    current_liabilities_adj = balance_sheet['Current Liabilities'] - balance_sheet['Current Debt And Capital Lease Obligation']
    nwc = current_assets_adj - current_liabilities_adj
    nwc_change = nwc.diff()

    capex = cash_flow['Capital Expenditure'] * -1
    da = cash_flow['Depreciation And Amortization']

    reinvestment = capex - da + nwc_change
    reinvestment_rate = reinvestment / nopat

    df_stats = pd.DataFrame({
        'Revenue Growth': revenue_growth,
        'EBIT Growth': ebit_growth,
        'Gross Margin': gross_margin,
        'EBIT Margin': ebit_margin,
        'EBITDA Margin': ebitda_margin,
        'Tax Rate': tax_rate_series,
        'Reinvestment Rate': reinvestment_rate
    })

    return {
        'df_stats': df_stats,
        'total_revenue': total_revenue / scale_factor,
        'gross_profit': gross_profit / scale_factor,
        'ebit': ebit / scale_factor,
        'ebitda': ebitda / scale_factor,
        'nopat': nopat / scale_factor,
        'nwc': nwc / scale_factor,
        'nwc_change': nwc_change / scale_factor,
        'capex': capex / scale_factor,
        'da': da / scale_factor,
        'reinvestment': reinvestment / scale_factor
    }

# ==================== MAIN APP ====================

# Title
st.title("💼 DCF Valuation System")
st.markdown("### Comprehensive Discounted Cash Flow Analysis for Public Companies")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 WACC Calculator", "📈 Historical Analysis", "🎯 DCF Model"])

# ==================== HOME TAB ====================
with tab1:
    st.markdown("""
    Welcome to the **DCF Valuation System** - a complete toolkit for performing professional-grade
    discounted cash flow analysis on publicly traded companies.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 WACC Calculator")
        st.markdown("""
        **Phase 1: Cost of Capital**

        - Beta calculation (OLS regression)
        - CAPM cost of equity
        - Credit spread-based cost of debt
        - 95% confidence intervals
        - Regression visualization

        **Status:** ✅ Complete
        """)

    with col2:
        st.markdown("### 📈 Historical Analysis")
        st.markdown("""
        **Phase 2: Historical Performance**

        - Revenue & EBIT growth rates
        - Margin analysis
        - Working capital trends
        - Reinvestment metrics
        - NOPAT & reinvestment rates

        **Status:** ✅ Complete
        """)

    with col3:
        st.markdown("### 🎯 DCF Model")
        st.markdown("""
        **Phase 3: Valuation Model**

        - LTM-based projections
        - Terminal value calculation
        - Firm value & share price
        - Sensitivity analysis
        - Price comparison

        **Status:** ✅ Complete
        """)

    st.markdown("---")
    st.markdown("## 🚀 How to Use")
    st.markdown("""
    1. **Select a tab** above (WACC Calculator or Historical Analysis)
    2. **Enter your ticker symbol** in the sidebar
    3. **Adjust parameters** as needed
    4. **Click the calculate/analyze button**
    5. **Review results** with detailed metrics and visualizations
    """)

    st.markdown("---")
    st.markdown("## 📊 Data Sources & Methodology")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Data Sources:**
        - Yahoo Finance API (yfinance)
        - Damodaran Credit Spreads (Jan 2025)
        - Manual risk-free rate input

        **Key Assumptions:**
        - 5 years monthly data for beta
        - EMRP default: 5%
        - Tax rate default: 25%
        """)

    with col2:
        st.markdown("""
        **Methodology:**
        - CAPM: k_E = r_f + β × EMRP
        - Credit spreads for cost of debt
        - OLS regression for beta
        - Gordon Growth for terminal value

        **Output Format:**
        - Figures in millions ($M)
        - 2 decimal places
        - 95% confidence intervals
        """)

# ==================== WACC CALCULATOR TAB ====================
with tab2:
    st.header("📊 WACC Calculator")
    st.markdown("Calculate the Weighted Average Cost of Capital for any publicly traded company")

    # Sidebar inputs for WACC
    with st.sidebar:
        st.header("WACC Input Parameters")

        ticker_symbol_wacc = st.text_input("Ticker Symbol (WACC)", value="MSFT", key="wacc_ticker").upper()
        rf = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1, key="wacc_rf") / 100
        emrp = st.number_input("Equity Market Risk Premium (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key="wacc_emrp") / 100
        marg_tax_rate = st.number_input("Marginal Tax Rate (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, key="wacc_tax") / 100

        ratings_list = [entry["Rating"] for entry in credit_spreads]
        firm_rating = st.selectbox("Firm Credit Rating", options=ratings_list, index=ratings_list.index("Aaa/AAA"), key="wacc_rating")

        index_symbol = st.text_input("Market Index Symbol", value="^GSPC", key="wacc_index")
        index_name = st.text_input("Market Index Name", value="S&P 500", key="wacc_index_name")

        scale_factor = 1000000
        scale_name = '$M'

        if st.button("Calculate WACC", type="primary", key="wacc_calc_btn"):
            with st.spinner("Calculating WACC..."):
                results = calculate_wacc(ticker_symbol_wacc, index_symbol, rf, emrp, firm_rating, marg_tax_rate, scale_factor)

                if results:
                    st.session_state['stored_wacc_results'] = results
                    st.session_state['stored_wacc_ticker'] = ticker_symbol_wacc
                    st.session_state['stored_wacc_index_name'] = index_name
                    st.session_state['stored_wacc_scale_name'] = scale_name
                    st.session_state['stored_wacc_rf'] = rf
                    st.session_state['stored_wacc_emrp'] = emrp

    # Display WACC results
    if 'stored_wacc_results' in st.session_state:
        results = st.session_state['stored_wacc_results']
        ticker_symbol = st.session_state['stored_wacc_ticker']
        index_name = st.session_state['stored_wacc_index_name']
        scale_name = st.session_state['stored_wacc_scale_name']
        rf = st.session_state['stored_wacc_rf']
        emrp = st.session_state['stored_wacc_emrp']

        st.success(f"✅ WACC calculated for {results['company_name']} ({ticker_symbol})")

        st.subheader("1. Company Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market Cap", f"${results['market_cap']:,.2f}{scale_name}")
        with col2:
            st.metric("Total Debt", f"${results['total_debt']:,.2f}{scale_name}")
        with col3:
            st.metric("Equity Weight", f"{results['w_E']:.2%}")
        with col4:
            st.metric("Debt Weight", f"{results['w_D']:.2%}")

        st.subheader("2. Cost of Equity (CAPM)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Beta (β)", f"{results['beta']:.3f}")
            st.caption(f"95% CI: [{results['beta_lower']:.3f}, {results['beta_upper']:.3f}]")
        with col2:
            st.metric("Risk-Free Rate", f"{rf:.2%}")
            st.metric("EMRP", f"{emrp:.2%}")
        with col3:
            st.metric("Cost of Equity", f"{results['cost_of_equity']:.2%}")
            st.caption(f"95% CI: [{results['cost_of_equity_lower']:.2%}, {results['cost_of_equity_upper']:.2%}]")

        st.subheader("3. Cost of Debt")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Credit Rating", firm_rating)
        with col2:
            st.metric("Credit Spread", f"{results['credit_spread']:.2%}")
        with col3:
            st.metric("Cost of Debt", f"{results['cost_of_debt']:.2%}")

        st.subheader("4. Weighted Average Cost of Capital (WACC)")
        st.latex(r"WACC = w_E \times k_E + w_D \times k_D \times (1-t)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("WACC (Lower CI)", f"{results['wacc_lower']:.2%}",
                     delta=f"{(results['wacc_lower'] - results['wacc'])*100:.2f} bps",
                     delta_color="inverse")
        with col2:
            st.metric("WACC (Estimate)", f"{results['wacc']:.2%}")
        with col3:
            st.metric("WACC (Upper CI)", f"{results['wacc_upper']:.2%}",
                     delta=f"{(results['wacc_upper'] - results['wacc'])*100:.2f} bps",
                     delta_color="normal")

        wacc_df = pd.DataFrame({
            'WACC': [results['wacc_lower']*100, results['wacc']*100, results['wacc_upper']*100]
        }, index=['Lower CI (95%)', 'Estimate', 'Upper CI (95%)'])

        st.dataframe(wacc_df.style.format("{:.2f}%"), use_container_width=True)

        with st.expander("View Detailed Regression Statistics"):
            st.text(results['regression_results'].summary())

        st.subheader("5. Beta Regression Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(results['index_returns'], results['stock_returns'],
                   alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5,
                   label='Monthly Returns')

        x_line = np.linspace(results['index_returns'].min(), results['index_returns'].max(), 100)
        y_line = results['regression_results'].params['const'] + results['regression_results'].params[index_symbol] * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Best Fit Line (β = {results["beta"]:.3f})')

        ax.set_xlabel(f'{index_name} Excess Returns', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{ticker_symbol} Excess Returns', fontsize=11, fontweight='bold')
        ax.set_title(f'Security Market Line: {ticker_symbol} vs {index_name}\n(5 Years Monthly Data)',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        r_squared = results['regression_results'].rsquared
        ax.text(0.95, 0.05, f'R² = {r_squared:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        st.pyplot(fig)
    else:
        st.info("👈 Enter your parameters in the sidebar and click 'Calculate WACC' to begin")

# ==================== HISTORICAL ANALYSIS TAB ====================
with tab3:
    st.header("📈 Historical Financial Analysis")
    st.markdown("Analyze historical financial performance to understand company fundamentals")

    # Sidebar inputs for Historical Analysis
    with st.sidebar:
        st.header("Historical Analysis Parameters")

        ticker_symbol_hist = st.text_input("Ticker Symbol (Historical)", value="MSFT", key="hist_ticker").upper()
        eff_tax_rate = st.number_input("Effective Tax Rate (%)", min_value=0.0, max_value=100.0, value=19.0, step=1.0, key="hist_tax") / 100

        scale_factor = 1000000
        scale_name = 'M'

        if st.button("Analyze Historical Data", type="primary", key="hist_analyze_btn"):
            with st.spinner("Fetching and analyzing data..."):
                income_stmt, balance_sheet, cash_flow, company_name = get_financial_data(ticker_symbol_hist)

                if income_stmt is not None:
                    results = calculate_historical_metrics(income_stmt, balance_sheet, cash_flow, eff_tax_rate, scale_factor)

                    st.session_state['stored_hist_results'] = results
                    st.session_state['stored_hist_company'] = company_name
                    st.session_state['stored_hist_ticker'] = ticker_symbol_hist
                    st.session_state['stored_hist_scale'] = scale_name

    # Display Historical Analysis results
    if 'stored_hist_results' in st.session_state:
        results = st.session_state['stored_hist_results']
        company_name = st.session_state['stored_hist_company']
        ticker_symbol = st.session_state['stored_hist_ticker']
        scale_name = st.session_state['stored_hist_scale']

        st.success(f"✅ Historical analysis complete for {company_name} ({ticker_symbol})")

        st.subheader("1. Revenue & EBIT Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Total Revenue ($M)**")
            revenue_df = pd.DataFrame({'Revenue': results['total_revenue']})
            st.dataframe(revenue_df.style.format("{:,.2f}"), use_container_width=True)

        with col2:
            st.markdown("**EBIT ($M)**")
            ebit_df = pd.DataFrame({'EBIT': results['ebit']})
            st.dataframe(ebit_df.style.format("{:,.2f}"), use_container_width=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        results['total_revenue'].plot(kind='bar', ax=ax1, color='steelblue', edgecolor='navy')
        ax1.set_title('Total Revenue Trend', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Revenue (${scale_name})', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        results['ebit'].plot(kind='bar', ax=ax2, color='coral', edgecolor='darkred')
        ax2.set_title('EBIT Trend', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'EBIT (${scale_name})', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("2. Growth Rates")
        growth_df = results['df_stats'][['Revenue Growth', 'EBIT Growth']].copy() * 100
        st.dataframe(growth_df.style.format("{:.2f}%"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(growth_df.index))
        width = 0.35

        ax.bar(x - width/2, growth_df['Revenue Growth'], width, label='Revenue Growth', color='steelblue', edgecolor='navy')
        ax.bar(x + width/2, growth_df['EBIT Growth'], width, label='EBIT Growth', color='coral', edgecolor='darkred')

        ax.set_title('Historical Growth Rates', fontsize=13, fontweight='bold')
        ax.set_ylabel('Growth Rate (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(growth_df.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("3. Margin Analysis")
        margin_df = results['df_stats'][['Gross Margin', 'EBIT Margin', 'EBITDA Margin']].copy() * 100
        st.dataframe(margin_df.style.format("{:.2f}%"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        margin_df.plot(kind='line', ax=ax, marker='o', linewidth=2.5)
        ax.set_title('Historical Margin Trends', fontsize=13, fontweight='bold')
        ax.set_ylabel('Margin (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("4. Complete Summary Statistics")
        summary_df = results['df_stats'].copy()
        summary_df[['Revenue Growth', 'EBIT Growth', 'Gross Margin', 'EBIT Margin',
                     'EBITDA Margin', 'Tax Rate', 'Reinvestment Rate']] *= 100

        st.dataframe(summary_df.style.format("{:.2f}%"), use_container_width=True)

        with st.expander("📊 Key Insights & Interpretation"):
            st.markdown("""
            **Growth Rates:** Positive growth indicates business expansion

            **Margins:** Higher margins show better profitability and pricing power

            **Reinvestment Rate:** High rate (>50%) suggests growth phase; low rate (<30%) indicates maturity

            **For DCF:** Use average growth rates, margin trends, and reinvestment rates for future projections
            """)
    else:
        st.info("👈 Enter a ticker symbol in the sidebar and click 'Analyze Historical Data' to begin")

# ==================== DCF MODEL TAB ====================
with tab4:
    st.header("🎯 DCF Valuation Model")
    st.markdown("Project future cash flows and derive intrinsic share price using discounted cash flow analysis")

    # Sidebar inputs for DCF Model
    with st.sidebar:
        st.header("DCF Model Parameters")

        ticker_symbol_dcf = st.text_input("Ticker Symbol (DCF)", value="MSFT", key="dcf_ticker").upper()

        st.subheader("Projection Assumptions")
        time_horizon = st.slider("Projection Period (Years)", min_value=5, max_value=15, value=10, key="dcf_horizon")

        # Growth rates
        st.markdown("**Revenue Growth Rates (%)**")
        growth_years_1_3 = st.number_input("Years 1-3", min_value=0.0, max_value=50.0, value=15.0, step=1.0, key="dcf_growth_1_3") / 100
        growth_years_4_6 = st.number_input("Years 4-6", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="dcf_growth_4_6") / 100
        growth_years_7_10 = st.number_input("Years 7-10", min_value=0.0, max_value=50.0, value=6.0, step=1.0, key="dcf_growth_7_10") / 100

        # Margins
        ebit_margin_pct = st.number_input("EBIT Margin (%)", min_value=0.0, max_value=100.0, value=46.0, step=1.0, key="dcf_ebit_margin") / 100
        eff_tax_rate_dcf = st.number_input("Effective Tax Rate (%)", min_value=0.0, max_value=100.0, value=19.0, step=1.0, key="dcf_eff_tax") / 100

        # Reinvestment
        reinv_rate_pct = st.number_input("Reinvestment Rate (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, key="dcf_reinv") / 100

        # Terminal value
        ss_growth_pct = st.number_input("Perpetual Growth Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key="dcf_ss_growth") / 100

        # WACC inputs
        st.subheader("WACC Scenarios")
        wacc_bear = st.number_input("Bear Case WACC (%)", min_value=0.0, max_value=50.0, value=10.54, step=0.1, key="dcf_wacc_bear") / 100
        wacc_base = st.number_input("Base Case WACC (%)", min_value=0.0, max_value=50.0, value=9.25, step=0.1, key="dcf_wacc_base") / 100
        wacc_bull = st.number_input("Bull Case WACC (%)", min_value=0.0, max_value=50.0, value=7.96, step=0.1, key="dcf_wacc_bull") / 100

        scale_factor = 1000000
        scale_name = 'M'

        if st.button("Run DCF Model", type="primary", key="dcf_run_btn"):
            with st.spinner("Running DCF valuation..."):
                try:
                    # Get company data with delay to avoid rate limiting
                    time.sleep(0.5)
                    ticker = yf.Ticker(ticker_symbol_dcf)
                    shares_outstanding = ticker.info.get('sharesOutstanding', 0)
                    total_debt = ticker.info.get('totalDebt', 0)
                    total_cash = ticker.info.get('totalCash', 0)
                    company_name = ticker.info.get('longName', ticker_symbol_dcf)

                    # Get LTM data
                    ltm_data = ticker.quarterly_income_stmt.T.sort_index().iloc[-4:]
                    ltm_revenue = ltm_data['Total Revenue'].sum()
                    most_recent_date = ltm_data.index[-1]

                    # Create growth pattern
                    growth_pattern = []
                    for i in range(time_horizon):
                        if i < 3:
                            growth_pattern.append(growth_years_1_3)
                        elif i < 6:
                            growth_pattern.append(growth_years_4_6)
                        else:
                            growth_pattern.append(growth_years_7_10)

                    # Create date range
                    freq_dict = {1:'YE-JAN',2:'YE-FEB',3:'YE-MAR',4:'YE-APR',5:'YE-MAY',6:'YE-JUN',
                                7:'YE-JUL',8:'YE-AUG',9:'YE-SEP',10:'YE-OCT',11:'YE-NOV',12:'YE-DEC'}
                    f = freq_dict[most_recent_date.month]
                    new_dates = pd.date_range(start=most_recent_date, periods=time_horizon+1, freq=f)

                    # Create projections dataframe
                    projections = pd.DataFrame(index=new_dates[1:])
                    projections['Revenue Growth'] = growth_pattern
                    projections['EBIT Margin'] = [ebit_margin_pct] * time_horizon
                    projections['Reinvestment Rate'] = [reinv_rate_pct] * time_horizon

                    # Calculate dollar projections
                    projections['Revenue'] = ltm_revenue * (1 + projections['Revenue Growth']).cumprod()
                    projections['EBIT'] = projections['Revenue'] * projections['EBIT Margin']
                    projections['NOPAT'] = projections['EBIT'] * (1 - eff_tax_rate_dcf)
                    projections['FCF'] = projections['NOPAT'] * (1 - projections['Reinvestment Rate'])

                    # Calculate valuations for all WACC scenarios
                    wacc_scenarios = {'Bear': wacc_bear, 'Base': wacc_base, 'Bull': wacc_bull}
                    results_dict = {}

                    for scenario, wacc in wacc_scenarios.items():
                        time_periods = np.arange(1, len(projections) + 1)
                        pv_fcfs = projections['FCF'] / ((1 + wacc) ** time_periods)

                        terminal_value = projections['FCF'].iloc[-1] * ((1 + ss_growth_pct) / (wacc - ss_growth_pct))
                        pv_terminal = terminal_value / ((1 + wacc) ** time_horizon)

                        firm_value = pv_fcfs.sum() + pv_terminal
                        equity_value = firm_value - total_debt + total_cash
                        share_price = equity_value / shares_outstanding

                        results_dict[scenario] = {
                            'wacc': wacc,
                            'pv_fcfs': pv_fcfs,
                            'terminal_value': terminal_value,
                            'pv_terminal': pv_terminal,
                            'firm_value': firm_value,
                            'equity_value': equity_value,
                            'share_price': share_price
                        }

                    # Get historical stock prices
                    stock_prices = ticker.history(period="1y")['Close']

                    # Store everything in session state with stored_ prefix
                    st.session_state['stored_dcf_results'] = results_dict
                    st.session_state['stored_dcf_projections'] = projections
                    st.session_state['stored_dcf_company'] = company_name
                    st.session_state['stored_dcf_ticker'] = ticker_symbol_dcf
                    st.session_state['stored_dcf_shares'] = shares_outstanding
                    st.session_state['stored_dcf_debt'] = total_debt
                    st.session_state['stored_dcf_cash'] = total_cash
                    st.session_state['stored_dcf_ltm_rev'] = ltm_revenue
                    st.session_state['stored_dcf_scale'] = scale_name
                    st.session_state['stored_dcf_ss_growth'] = ss_growth_pct
                    st.session_state['stored_dcf_stock_prices'] = stock_prices

                except Exception as e:
                    error_msg = str(e)
                    if "rate" in error_msg.lower() or "429" in error_msg or "too many" in error_msg.lower():
                        st.error("⚠️ Yahoo Finance rate limit reached. Please wait a few minutes and try again.")
                    else:
                        st.error(f"Error running DCF model: {e}")

    # Display DCF results
    if 'stored_dcf_results' in st.session_state:
        results_dict = st.session_state['stored_dcf_results']
        projections = st.session_state['stored_dcf_projections']
        company_name = st.session_state['stored_dcf_company']
        ticker_symbol = st.session_state['stored_dcf_ticker']
        shares_outstanding = st.session_state['stored_dcf_shares']
        total_debt = st.session_state['stored_dcf_debt']
        total_cash = st.session_state['stored_dcf_cash']
        ltm_revenue = st.session_state['stored_dcf_ltm_rev']
        scale_name = st.session_state['stored_dcf_scale']
        ss_growth_pct = st.session_state['stored_dcf_ss_growth']
        stock_prices = st.session_state['stored_dcf_stock_prices']

        st.success(f"✅ DCF valuation complete for {company_name} ({ticker_symbol})")

        # Section 1: Key Inputs Summary
        st.subheader("1. Model Inputs Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("LTM Revenue", f"${ltm_revenue/scale_factor/1000:,.2f}M")
        with col2:
            st.metric("Shares Outstanding", f"{shares_outstanding/1000000:,.2f}M")
        with col3:
            st.metric("Total Debt", f"${total_debt/scale_factor/1000:,.2f}M")
        with col4:
            st.metric("Total Cash", f"${total_cash/scale_factor/1000:,.2f}M")

        # Section 2: Projections
        st.subheader("2. Financial Projections")

        # Show ratios
        st.markdown("**Projection Assumptions (Ratios)**")
        ratios_df = projections[['Revenue Growth', 'EBIT Margin', 'Reinvestment Rate']].copy() * 100
        st.dataframe(ratios_df.style.format("{:.2f}%"), use_container_width=True)

        # Show dollar values
        st.markdown("**Projected Values ($M)**")
        values_df = projections[['Revenue', 'EBIT', 'NOPAT', 'FCF']].copy() / scale_factor
        st.dataframe(values_df.style.format("{:,.2f}"), use_container_width=True)

        # Projections chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        (projections['Revenue']/scale_factor).plot(kind='bar', ax=ax1, color='steelblue', edgecolor='navy')
        ax1.set_title('Revenue Projection', fontweight='bold')
        ax1.set_ylabel(f'Revenue (${scale_name})', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        (projections['EBIT']/scale_factor).plot(kind='bar', ax=ax2, color='coral', edgecolor='darkred')
        ax2.set_title('EBIT Projection', fontweight='bold')
        ax2.set_ylabel(f'EBIT (${scale_name})', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        (projections['NOPAT']/scale_factor).plot(kind='bar', ax=ax3, color='green', edgecolor='darkgreen')
        ax3.set_title('NOPAT Projection', fontweight='bold')
        ax3.set_ylabel(f'NOPAT (${scale_name})', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        (projections['FCF']/scale_factor).plot(kind='bar', ax=ax4, color='purple', edgecolor='indigo')
        ax4.set_title('Free Cash Flow Projection', fontweight='bold')
        ax4.set_ylabel(f'FCF (${scale_name})', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Section 3: Valuation Results
        st.subheader("3. Valuation Results by Scenario")

        valuation_data = []
        for scenario in ['Bull', 'Base', 'Bear']:
            res = results_dict[scenario]
            valuation_data.append({
                'Scenario': scenario,
                'WACC': f"{res['wacc']:.2%}",
                'Firm Value ($M)': f"{res['firm_value']/scale_factor/1000:,.2f}",
                'Equity Value ($M)': f"{res['equity_value']/scale_factor/1000:,.2f}",
                'Share Price': f"${res['share_price']:.2f}"
            })

        valuation_df = pd.DataFrame(valuation_data)
        st.dataframe(valuation_df, use_container_width=True)

        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bull Case Price", f"${results_dict['Bull']['share_price']:.2f}",
                     delta=f"+{((results_dict['Bull']['share_price']/results_dict['Base']['share_price'])-1)*100:.1f}%")
        with col2:
            st.metric("Base Case Price", f"${results_dict['Base']['share_price']:.2f}")
        with col3:
            st.metric("Bear Case Price", f"${results_dict['Bear']['share_price']:.2f}",
                     delta=f"{((results_dict['Bear']['share_price']/results_dict['Base']['share_price'])-1)*100:.1f}%",
                     delta_color="inverse")

        # Section 4: Historical Price Comparison
        st.subheader("4. Model Price vs Historical Stock Price")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical prices
        ax.plot(stock_prices.index, stock_prices.values, label='Historical Price', linewidth=2, color='blue')

        # Plot DCF estimates as horizontal lines
        ax.axhline(y=results_dict['Base']['share_price'], color='green', linestyle='--', linewidth=2, label=f"Base Case: ${results_dict['Base']['share_price']:.2f}")
        ax.axhline(y=results_dict['Bull']['share_price'], color='darkgreen', linestyle=':', linewidth=2, label=f"Bull Case: ${results_dict['Bull']['share_price']:.2f}")
        ax.axhline(y=results_dict['Bear']['share_price'], color='red', linestyle=':', linewidth=2, label=f"Bear Case: ${results_dict['Bear']['share_price']:.2f}")

        # Shade the confidence interval
        ax.fill_between(stock_prices.index, results_dict['Bear']['share_price'], results_dict['Bull']['share_price'],
                        color='gray', alpha=0.2, label='DCF Price Range')

        ax.set_title(f'{ticker_symbol} - DCF Model Price vs Historical Price (1 Year)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Section 5: Detailed Breakdown
        with st.expander("📊 Detailed Valuation Breakdown (Base Case)"):
            base_res = results_dict['Base']

            st.markdown("**Present Value of Free Cash Flows**")
            pv_fcf_df = pd.DataFrame({
                'Year': range(1, len(projections) + 1),
                'FCF ($M)': (projections['FCF'] / scale_factor).values,
                'PV of FCF ($M)': (base_res['pv_fcfs'] / scale_factor).values
            })
            st.dataframe(pv_fcf_df.style.format({'FCF ($M)': '{:,.2f}', 'PV of FCF ($M)': '{:,.2f}'}), use_container_width=True)

            st.markdown("**Terminal Value Calculation**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Terminal Value", f"${base_res['terminal_value']/scale_factor/1000:,.2f}M")
                st.caption(f"Using Gordon Growth Model at {ss_growth_pct:.1%} perpetual growth")
            with col2:
                st.metric("PV of Terminal Value", f"${base_res['pv_terminal']/scale_factor/1000:,.2f}M")
                st.caption(f"Discounted at {base_res['wacc']:.2%} WACC")

            st.markdown("**Firm Value to Equity Value**")
            breakdown_df = pd.DataFrame({
                'Component': ['PV of FCFs', 'PV of Terminal Value', 'Firm Value', 'Less: Total Debt', 'Plus: Total Cash', 'Equity Value'],
                'Value ($M)': [
                    base_res['pv_fcfs'].sum()/scale_factor/1000,
                    base_res['pv_terminal']/scale_factor/1000,
                    base_res['firm_value']/scale_factor/1000,
                    -total_debt/scale_factor/1000,
                    total_cash/scale_factor/1000,
                    base_res['equity_value']/scale_factor/1000
                ]
            })
            st.dataframe(breakdown_df.style.format({'Value ($M)': '{:,.2f}'}), use_container_width=True)

            st.markdown(f"**Implied Share Price:** ${base_res['share_price']:.2f} (Equity Value / {shares_outstanding/1000000:,.2f}M shares)")

    else:
        st.info("👈 Enter your assumptions in the sidebar and click 'Run DCF Model' to begin")

        st.markdown("""
        ### About the DCF Model:

        This module performs a complete **Discounted Cash Flow (DCF) valuation** to derive an intrinsic share price:

        **Process:**
        1. **LTM Data**: Uses Last Twelve Months revenue as the starting point
        2. **Projections**: Projects Revenue, EBIT, NOPAT, and FCF based on your assumptions
        3. **Discounting**: Discounts all future cash flows to present value using WACC
        4. **Terminal Value**: Calculates perpetual value using Gordon Growth Model
        5. **Valuation**: Derives firm value, equity value, and intrinsic share price
        6. **Scenarios**: Provides Bull/Base/Bear cases using different WACC assumptions

        **Key Assumptions:**
        - **Revenue Growth**: How fast the company will grow
        - **EBIT Margin**: Operating profitability
        - **Reinvestment Rate**: Capital reinvestment needs
        - **Perpetual Growth**: Long-term sustainable growth rate
        - **WACC**: Discount rate (cost of capital)

        **Output:**
        - Intrinsic share price range (Bull/Base/Bear scenarios)
        - Complete financial projections
        - Historical price comparison
        - Detailed valuation breakdown
        """)

