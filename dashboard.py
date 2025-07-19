import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def detect_column_mapping(df):
    """Automatically detect column mappings based on common naming patterns"""
    columns = df.columns.str.lower()
    mapping = {}
    
    # Date patterns
    date_patterns = ['date', 'time', 'timestamp', 'deal time']
    for pattern in date_patterns:
        matches = [col for col in df.columns if pattern in col.lower()]
        if matches:
            mapping['date'] = matches[0]
            break
    
    # Grade/Product patterns
    grade_patterns = ['grade', 'product', 'commodity', 'instrument']
    for pattern in grade_patterns:
        matches = [col for col in df.columns if pattern in col.lower()]
        if matches:
            mapping['grade'] = matches[0]
            break
    
    # Delivery period patterns
    period_patterns = ['del. period', 'delivery period', 'period']
    for pattern in period_patterns:
        matches = [col for col in df.columns if pattern in col.lower()]
        if matches:
            mapping['delivery_period'] = matches[0]
            break
    
    # Year patterns
    year_patterns = ['del. year', 'delivery year', 'year']
    for pattern in year_patterns:
        matches = [col for col in df.columns if pattern in col.lower()]
        if matches:
            mapping['delivery_year'] = matches[0]
            break
    
    # Price patterns
    price_patterns = ['price', 'rate', 'cost']
    for pattern in price_patterns:
        matches = [col for col in df.columns if pattern in col.lower() and 'unit' not in col.lower()]
        if matches:
            mapping['price'] = matches[0]
            break
    
    # Volume patterns
    volume_patterns = ['volume', 'quantity', 'amount', 'mw', 'size']
    for pattern in volume_patterns:
        matches = [col for col in df.columns if pattern in col.lower() and 'unit' not in col.lower()]
        if matches:
            mapping['volume'] = matches[0]
            break
    
    return mapping



def load_and_process_data(uploaded_file):
    """Load and process uploaded data file"""
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Read Excel file, skip first 2 rows
            df = pd.read_excel(uploaded_file, skiprows=2)
        elif uploaded_file.name.endswith('.csv'):
            # Read CSV file, skip first 2 rows
            df = pd.read_csv(uploaded_file, skiprows=2)
        else:
            st.error("Unsupported file format. Please upload .xlsx, .xls, or .csv files.")
            return None, None
        
        # Remove completely empty columns and rows
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Auto-detect column mapping
        column_mapping = detect_column_mapping(df)
        
        return df, column_mapping
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def calculate_price_volatility(df, price_col):
    """Calculate price volatility"""
    if price_col in df.columns:
        prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
        if len(prices) > 1:
            return prices.std()
    return 0


def calculate_risk_metrics(df, price_col, volume_col=None, confidence_level=0.95):
    """Calculate comprehensive risk metrics"""
    try:
        # Convert prices to numeric and remove NaN values
        prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
        
        if len(prices) < 10:
            return {
                'var_95': np.nan,
                'cvar_95': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'max_drawdown': np.nan,
                'sharpe_ratio': np.nan
            }
        
        # Calculate price returns
        price_returns = prices.pct_change().dropna()
        
        # VaR calculation (95% confidence level)
        var_95 = np.percentile(price_returns, (1 - confidence_level) * 100)
        
        # CVaR calculation (Conditional VaR - expected loss beyond VaR)
        cvar_95 = price_returns[price_returns <= var_95].mean() if len(price_returns[price_returns <= var_95]) > 0 else var_95
        
        # Skewness and Kurtosis
        skewness = stats.skew(price_returns)
        kurtosis = stats.kurtosis(price_returns, fisher=True)  # Excess kurtosis
        
        # Maximum Drawdown
        cumulative_returns = (1 + price_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = price_returns.mean() / price_returns.std() * np.sqrt(252) if price_returns.std() != 0 else 0
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")
        return {
            'var_95': np.nan,
            'cvar_95': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'max_drawdown': np.nan,
            'sharpe_ratio': np.nan
        }

def create_risk_distribution_chart(df, price_col):
    """Create price returns distribution with VaR and CVaR visualization"""
    try:
        prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
        
        if len(prices) < 10:
            return None
            
        # Calculate returns
        returns = prices.pct_change().dropna() * 100  # Convert to percentage
        
        # Calculate risk metrics
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()
        
        # Create histogram
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Price Returns',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add VaR line
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"VaR 95%: {var_95:.2f}%",
            annotation_position="top left"
        )
        
        # Add CVaR line
        fig.add_vline(
            x=cvar_95,
            line_dash="dot",
            line_color="darkred",
            line_width=2,
            annotation_text=f"CVaR 95%: {cvar_95:.2f}%",
            annotation_position="bottom left"
        )
        
        # Shade the tail area (CVaR region)
        tail_returns = returns[returns <= var_95]
        if len(tail_returns) > 0:
            fig.add_trace(go.Histogram(
                x=tail_returns,
                nbinsx=50,
                name='CVaR Region',
                marker_color='red',
                opacity=0.5
            ))
        
        fig.update_layout(
            title='Price Returns Distribution with Risk Metrics',
            xaxis_title='Daily Returns (%)',
            yaxis_title='Frequency',
            showlegend=True,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating risk distribution chart: {str(e)}")
        return None
def generate_price_forecast(df, date_col, price_col, periods=30):
    """Simple moving average forecast"""
    try:
        # Prepare data
        forecast_df = df.copy()
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors='coerce')
        forecast_df[price_col] = pd.to_numeric(forecast_df[price_col], errors='coerce')
        
        # Remove NaN values and sort by date
        forecast_df = forecast_df.dropna(subset=[date_col, price_col])
        forecast_df = forecast_df.sort_values(date_col)
        
        if len(forecast_df) < 10:
            return None, "Insufficient data for forecasting"
        
        # Calculate moving averages
        forecast_df['MA_7'] = forecast_df[price_col].rolling(window=7, min_periods=1).mean()
        forecast_df['MA_30'] = forecast_df[price_col].rolling(window=30, min_periods=1).mean()
        
        # Simple forecast based on trend
        recent_trend = forecast_df[price_col].tail(10).mean() - forecast_df[price_col].tail(20).head(10).mean()
        last_price = forecast_df[price_col].iloc[-1]
        forecast_price = last_price + recent_trend
        
        return forecast_price, None
        
    except Exception as e:
        return None, str(e)

def create_daily_volume_chart(df, date_col, volume_col, grade_filter=None):
    """Create daily trading volume chart with trend line"""
    try:
        chart_df = df.copy()
        if grade_filter:
            chart_df = chart_df[chart_df[st.session_state.column_mapping['grade']] == grade_filter]
        
        chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors='coerce')
        chart_df[volume_col] = pd.to_numeric(chart_df[volume_col], errors='coerce')
        
        # Group by date and sum volumes
        daily_volume = chart_df.groupby(chart_df[date_col].dt.date)[volume_col].sum().reset_index()
        daily_volume.columns = ['Date', 'Total_Volume']
        daily_volume['Date'] = pd.to_datetime(daily_volume['Date'])
        
        # Calculate moving average trend line
        daily_volume = daily_volume.sort_values('Date')
        daily_volume['Volume_Trend'] = daily_volume['Total_Volume'].rolling(window=7, min_periods=1, center=True).mean()
        
        # Create figure
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=daily_volume['Date'],
            y=daily_volume['Total_Volume'],
            name='Daily Volume',
            marker_color='#B19CD9',
            opacity=0.8
        ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=daily_volume['Date'],
            y=daily_volume['Volume_Trend'],
            mode='lines',
            name='Volume Trend',
            line=dict(color='#FF8C00', width=3, dash='dash'),
            yaxis='y'
        ))
        
        fig.update_layout(
            title='Daily Trading Volume',
            xaxis_title="Date",
            yaxis_title="Volume (MW)",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating volume chart: {str(e)}")
        return None

def create_price_chart(df, date_col, price_col, grade_filter=None):
    """Create average price per day chart with trend line and price range"""
    try:
        chart_df = df.copy()
        if grade_filter:
            chart_df = chart_df[chart_df[st.session_state.column_mapping['grade']] == grade_filter]
        
        chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors='coerce')
        chart_df[price_col] = pd.to_numeric(chart_df[price_col], errors='coerce')
        
        # Group by date and calculate statistics
        daily_stats = chart_df.groupby(chart_df[date_col].dt.date)[price_col].agg([
            'mean', 'min', 'max', 'std'
        ]).reset_index()
        daily_stats.columns = ['Date', 'Avg_Price', 'Min_Price', 'Max_Price', 'Std_Price']
        daily_stats['Date'] = pd.to_datetime(daily_stats['Date'])
        daily_stats = daily_stats.sort_values('Date')
        
        # Calculate trend line (linear regression)
        from scipy import stats
        if len(daily_stats) > 1:
            x_numeric = np.arange(len(daily_stats))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, daily_stats['Avg_Price'])
            daily_stats['Trend_Line'] = slope * x_numeric + intercept
        else:
            daily_stats['Trend_Line'] = daily_stats['Avg_Price']
        
        # Create figure
        fig = go.Figure()
        
        # Add price range (fill between min and max)
        fig.add_trace(go.Scatter(
            x=daily_stats['Date'].tolist() + daily_stats['Date'].tolist()[::-1],
            y=daily_stats['Max_Price'].tolist() + daily_stats['Min_Price'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Price Range',
            showlegend=True
        ))
        
        # Add average price line
        fig.add_trace(go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['Avg_Price'],
            mode='lines',
            name='Average Price',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['Trend_Line'],
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Average Price Over Time',
            xaxis_title="Date",
            yaxis_title="Price (£/MWh)",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return None

def create_price_distribution_chart(df, price_col, grade_filter=None):
    """Create price distribution histogram with mean price line"""
    try:
        chart_df = df.copy()
        if grade_filter:
            chart_df = chart_df[chart_df[st.session_state.column_mapping['grade']] == grade_filter]
        
        chart_df[price_col] = pd.to_numeric(chart_df[price_col], errors='coerce')
        prices = chart_df[price_col].dropna()
        
        # Calculate mean price
        mean_price = prices.mean()
        
        fig = px.histogram(x=prices, nbins=30,
                          title='Price Distribution',
                          labels={'x': 'Price (GBP/MWh)', 'y': 'Frequency'})
        
        # Add mean price line
        fig.add_vline(x=mean_price, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: £{mean_price:.2f}", 
                     annotation_position="top right")
        
        fig.update_layout(
            xaxis_title="Price (GBP/MWh)",
            yaxis_title="Frequency",
            showlegend=False,
            height=550
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating distribution chart: {str(e)}")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">⚡ Energy Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    
    # Sidebar
    st.sidebar.header("🔧 Data & Filters")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your energy trading data",
        type=['xlsx', 'xls', 'csv'],
        help="Upload Excel or CSV files. First 2 rows will be skipped automatically."
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading and processing data..."):
            data, column_mapping = load_and_process_data(uploaded_file)
        
        if data is not None:
            st.session_state.data = data
            st.session_state.column_mapping = column_mapping
            
            st.sidebar.success(f"✅ Loaded {len(data)} records")
            
            # Show detected structure
            with st.sidebar.expander("🔍 Detected Data Structure", expanded=False):
                st.write("**Auto-detected columns:**")
                for key, value in column_mapping.items():
                    st.write(f"• {key.replace('_', ' ').title()}: `{value}`")
                
                if len(column_mapping) < 6:
                    st.warning("⚠️ Some columns need manual mapping below")
            
            # Manual column mapping
            st.sidebar.subheader("📊 Column Mapping")
            
            available_columns = [''] + list(data.columns)
            
            # Essential column mappings
            date_col = st.sidebar.selectbox(
                "Date Column", 
                available_columns,
                index=available_columns.index(column_mapping.get('date', '')) if column_mapping.get('date') in available_columns else 0
            )
            
            grade_col = st.sidebar.selectbox(
                "Grade/Product Column",
                available_columns,
                index=available_columns.index(column_mapping.get('grade', '')) if column_mapping.get('grade') in available_columns else 0
            )
            
            year_col = st.sidebar.selectbox(
                "Delivery Year Column",
                available_columns,
                index=available_columns.index(column_mapping.get('delivery_year', '')) if column_mapping.get('delivery_year') in available_columns else 0
            )
            
            price_col = st.sidebar.selectbox(
                "Price Column",
                available_columns,
                index=available_columns.index(column_mapping.get('price', '')) if column_mapping.get('price') in available_columns else 0
            )
            
            volume_col = st.sidebar.selectbox(
                "Volume Column",
                available_columns,
                index=available_columns.index(column_mapping.get('volume', '')) if column_mapping.get('volume') in available_columns else 0
            )
            
            period_col = st.sidebar.selectbox(
                "Delivery Period Column",
                available_columns,
                index=available_columns.index(column_mapping.get('delivery_period', '')) if column_mapping.get('delivery_period') in available_columns else 0
            )
            
            # Update mapping
            st.session_state.column_mapping.update({
                'date': date_col,
                'grade': grade_col,
                'delivery_year': year_col,
                'price': price_col,
                'volume': volume_col,
                'delivery_period': period_col
            })
            
            # Validation
            required_cols = [date_col, grade_col, year_col, price_col, volume_col]
            if not all(required_cols):
                st.sidebar.error("❌ Please map all required columns")
                st.error("Please complete the column mapping in the sidebar to proceed.")
                return
            
            # Filters
            st.sidebar.subheader("🎯 Filters")
            
            # Year filter
            available_years = sorted(data[year_col].dropna().unique())
            selected_years = st.sidebar.multiselect(
                "Delivery Years",
                available_years,
                default=available_years,
                help="Filter data by delivery year"
            )
            
            # Grade filter
            available_grades = sorted(data[grade_col].dropna().unique())
            selected_grade = st.sidebar.selectbox(
                "Select Grade/Product",
                ['All'] + list(available_grades),
                help="Analyze specific product or all products"
            )
            
            # Delivery period filter
            if period_col and period_col in data.columns:
                available_periods = sorted(data[period_col].dropna().unique())
                selected_periods = st.sidebar.multiselect(
                    "Delivery Periods",
                    available_periods,
                    default=available_periods,
                    help="Filter data by delivery period"
                )
            else:
                selected_periods = None
            
            # Filter data
            filtered_data = data[data[year_col].isin(selected_years)]
            if selected_grade != 'All':
                filtered_data = filtered_data[filtered_data[grade_col] == selected_grade]
            if selected_periods and period_col:
                filtered_data = filtered_data[filtered_data[period_col].isin(selected_periods)]
            
            if filtered_data.empty:
                st.warning("No data matches the selected filters. Please adjust your selection.")
                return
            
            # Main dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            # Key metrics
            total_deals = len(filtered_data)
            avg_price = filtered_data[price_col].astype(float).mean()
            total_volume = filtered_data[volume_col].astype(float).sum()
            price_volatility = calculate_price_volatility(filtered_data, price_col)
            
            with col1:
                st.metric("Total Deals", f"{total_deals:,}")
            
            with col2:
                st.metric("Average Price", f"£{avg_price:.2f}/MWh")
            
            with col3:
                st.metric("Total Volume", f"{total_volume:,.0f} MW")
            
            with col4:
                st.metric("Price Volatility", f"£{price_volatility:.2f}")
            
            # Risk Metrics Section
            st.subheader("⚠️ Risk Metrics")
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(filtered_data, price_col, volume_col)
            
            # Display risk metrics
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                var_value = risk_metrics['var_95']
                var_display = f"{var_value:.2%}" if not np.isnan(var_value) else "N/A"
                st.metric("VaR (95%)", var_display, help="Value at Risk: Maximum expected loss at 95% confidence level")
            
            with risk_col2:
                cvar_value = risk_metrics['cvar_95']
                cvar_display = f"{cvar_value:.2%}" if not np.isnan(cvar_value) else "N/A"
                st.metric("CVaR (95%)", cvar_display, help="Conditional VaR: Expected loss beyond VaR threshold")
            
            with risk_col3:
                skew_value = risk_metrics['skewness']
                skew_display = f"{skew_value:.3f}" if not np.isnan(skew_value) else "N/A"
                skew_interpretation = "Left-skewed" if skew_value < -0.5 else "Right-skewed" if skew_value > 0.5 else "Symmetric"
                st.metric("Price Skew", skew_display, delta=skew_interpretation, help="Measure of price distribution asymmetry")
            
            with risk_col4:
                kurt_value = risk_metrics['kurtosis']
                kurt_display = f"{kurt_value:.3f}" if not np.isnan(kurt_value) else "N/A"
                kurt_interpretation = "Fat tails" if kurt_value > 0 else "Thin tails"
                st.metric("Price Kurtosis", kurt_display, delta=kurt_interpretation, help="Measure of tail risk (excess kurtosis)")
            
            # Additional risk metrics in expandable section
            with st.expander("📊 Additional Risk Metrics"):
                add_risk_col1, add_risk_col2 = st.columns(2)
                
                with add_risk_col1:
                    max_dd_value = risk_metrics['max_drawdown']
                    max_dd_display = f"{max_dd_value:.2%}" if not np.isnan(max_dd_value) else "N/A"
                    st.metric("Maximum Drawdown", max_dd_display, help="Largest peak-to-trough decline")
                
                with add_risk_col2:
                    sharpe_value = risk_metrics['sharpe_ratio']
                    sharpe_display = f"{sharpe_value:.3f}" if not np.isnan(sharpe_value) else "N/A"
                    st.metric("Sharpe Ratio", sharpe_display, help="Risk-adjusted return measure")
            
            # Risk interpretation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="risk-box">
                    <h4>🎯 Risk Interpretation</h4>
                    <p><strong>VaR Analysis:</strong> {"High risk - significant potential losses" if not np.isnan(var_value) and var_value < -0.05 else "Moderate risk levels" if not np.isnan(var_value) else "Insufficient data"}</p>
                    <p><strong>Distribution Shape:</strong> {"Negative skew indicates higher probability of extreme losses" if not np.isnan(skew_value) and skew_value < -0.5 else "Positive skew indicates higher probability of extreme gains" if not np.isnan(skew_value) and skew_value > 0.5 else "Relatively symmetric distribution"}</p>
                    <p><strong>Tail Risk:</strong> {"High tail risk - expect more extreme events" if not np.isnan(kurt_value) and kurt_value > 1 else "Normal tail risk" if not np.isnan(kurt_value) else "Unable to assess"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Risk distribution chart
                risk_chart = create_risk_distribution_chart(filtered_data, price_col)
                if risk_chart:
                    st.plotly_chart(risk_chart, use_container_width=True)
            
            # Charts section
            st.subheader("📈 Trading Analysis")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                volume_chart = create_daily_volume_chart(
                    filtered_data, date_col, volume_col, 
                    selected_grade if selected_grade != 'All' else None
                )
                if volume_chart:
                    st.plotly_chart(volume_chart, use_container_width=True)
            
            with chart_col2:
                price_chart = create_price_chart(
                    filtered_data, date_col, price_col,
                    selected_grade if selected_grade != 'All' else None
                )
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
            
            # Price distribution
            st.subheader("💹 Price Distribution")
            distribution_chart = create_price_distribution_chart(
                filtered_data, price_col,
                selected_grade if selected_grade != 'All' else None
            )
            if distribution_chart:
                st.plotly_chart(distribution_chart, use_container_width=True)
            
            # Data insights
            st.subheader("🔍 Data Insights")
            
            try:
                insights_data = filtered_data.copy()
                insights_data[date_col] = pd.to_datetime(insights_data[date_col], errors='coerce')
                insights_data[price_col] = pd.to_numeric(insights_data[price_col], errors='coerce')
                
                # Calculate insights
                price_data = insights_data.dropna(subset=[date_col, price_col])
                
                if len(price_data) > 0:
                    # Highest price date
                    highest_price_idx = price_data[price_col].idxmax()
                    highest_price = price_data.loc[highest_price_idx, price_col]
                    highest_price_date = price_data.loc[highest_price_idx, date_col]
                    
                    # Lowest price date
                    lowest_price_idx = price_data[price_col].idxmin()
                    lowest_price = price_data.loc[lowest_price_idx, price_col]
                    lowest_price_date = price_data.loc[lowest_price_idx, date_col]
                    
                    # Recent trend
                    if len(price_data) >= 10:
                        recent_avg = price_data[price_col].tail(5).mean()
                        previous_avg = price_data[price_col].tail(10).head(5).mean()
                        trend_direction = "📈 Increasing" if recent_avg > previous_avg else "📉 Decreasing"
                    else:
                        trend_direction = "📊 Insufficient data for trend analysis"
                    
                    # Price forecast
                    forecast_price, forecast_error = generate_price_forecast(
                        insights_data, date_col, price_col
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>📊 Price Analysis</h4>
                            <p><strong>Highest Price:</strong> £{highest_price:.2f}/MWh on {highest_price_date.strftime('%Y-%m-%d')}</p>
                            <p><strong>Lowest Price:</strong> £{lowest_price:.2f}/MWh on {lowest_price_date.strftime('%Y-%m-%d')}</p>
                            <p><strong>Price Range:</strong> £{highest_price - lowest_price:.2f}/MWh</p>
                            <p><strong>Recent Trend:</strong> {trend_direction}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>🔮 Market Forecast</h4>
                            <p><strong>Total Deals:</strong> {len(filtered_data):,}</p>
                            <p><strong>Active Products:</strong> {filtered_data[grade_col].nunique()}</p>
                            <p><strong>Date Range:</strong> {len(pd.to_datetime(filtered_data[date_col]).dt.date.unique())} days</p>
                            {"<p><strong>Forecast Price:</strong> £{:.2f}/MWh</p>".format(forecast_price) if forecast_price else "<p><strong>Forecast:</strong> Insufficient data</p>"}
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
            
            # Raw data view
            with st.expander("📋 View Filtered Data"):
                st.dataframe(filtered_data, use_container_width=True, height=400)
                
                # Download filtered data
                csv_buffer = io.StringIO()
                filtered_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"filtered_energy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to the Energy Trading Dashboard!**
        
        📤 **Upload your data file** (Excel or CSV) using the sidebar to get started.
        
        ✨ **Features:**
        - 🔄 **Automatic data detection** for common column types
        - 📊 **Interactive charts** for volume and price analysis  
        - 🎯 **Smart filtering** by delivery year and product grade
        - 📈 **Price volatility** and trend analysis
        - 🔮 **Price forecasting** based on historical trends
        - 💡 **Data insights** with key statistics and patterns
        
        📋 **Expected data structure:**
        - First 2 rows will be skipped automatically
        - Columns should include: Date, Grade/Product, Delivery Year, Price, Volume
        - The system will try to auto-detect these columns based on common naming patterns
        """)
        
        # Sample data format
        st.subheader("📝 Expected Data Format")
        sample_data = pd.DataFrame({
            'Date': ['2025-07-18', '2025-07-18', '2025-07-19'],
            'Grade': ['UK OTC base load', 'UK OTC base load', 'UK spark spread'],
            'Del. Year': [2025, 2026, 2025],
            'Volume': [15, 10, 5],
            'Price': [64.06, 75.25, 82.50],
            'Del. Period': ['summer Gregorian', 'winter Gregorian', 'quarter 4']
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()