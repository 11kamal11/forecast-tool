import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import chardet
import warnings
import traceback
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Data Analysis & Forecasting Tool", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(ttl=3600)
def load_data(file):
    try:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        file.seek(0)
        delimiters = [',', ';', '\t', '|']
        for delimiter in delimiters:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, on_bad_lines='warn', low_memory=False)
                if len(df.columns) > 1:
                    st.info(f"Loaded CSV with delimiter '{delimiter}' and encoding '{encoding}'")
                    return df
            except:
                continue
        file.seek(0)
        df = pd.read_csv(file, encoding=encoding, engine='python', low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}\n{traceback.format_exc()}")
        return None

def is_date_column(series, sample_size=100):
    """Check if a column contains date-like values."""
    sample = series.head(sample_size).astype(str).str.strip()
    date_formats = ['%Y-%m', '%Y%m', '%Y/%m', '%m/%Y', '%b %Y', '%Y']
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
            if parsed.notna().sum() >= len(sample) * 0.8:
                return True
        except:
            continue
    try:
        parsed = pd.to_datetime(sample, errors='coerce')
        if parsed.notna().sum() >= len(sample) * 0.8:
            return True
    except:
        return False
    # Check for YYYYMM pattern (e.g., 202209)
    if sample.str.match(r'^\d{6}$').sum() >= len(sample) * 0.8:
        return True
    return False

def preprocess_time_series(df, date_col, target_col, granularity):
    try:
        st.write(f"Preprocessing time series: Date Column = {date_col}, Target = {target_col}, Granularity = {granularity}")
        st.write(f"Raw {date_col} sample (first 5):\n{df[date_col].head()}")
        st.write(f"Unique {date_col} values (first 10):\n{df[date_col].unique()[:10]}")
        st.write(f"Total unique {date_col} values: {len(df[date_col].unique())}")
        
        # Handle YYYYMM format explicitly
        if df[date_col].astype(str).str.match(r'^\d{6}$').any():
            try:
                df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce')
                st.info("Parsed dates as YYYYMM format (e.g., 202209 → 2022-09-01)")
            except:
                pass
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Try additional formats if needed
        if df[date_col].isna().all():
            for fmt in ['%Y-%m', '%Y%m', '%Y/%m', '%m/%Y', '%b %Y', '%Y']:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                    if not df[date_col].isna().all():
                        st.info(f"Parsed dates using format: {fmt}")
                        break
                except:
                    continue
        
        if df[date_col].isna().any():
            st.warning(f"{df[date_col].isna().sum()} invalid dates found. Filling with forward/backward fill.")
            df[date_col] = df[date_col].fillna(method='ffill').fillna(method='bfill')
        
        df = df.sort_values(date_col)
        df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        df['y'] = df['y'].fillna(df['y'].mean())
        st.write(f"After initial preprocessing (rows: {len(df)}):\n{df.head()}")
        
        if len(df['ds'].unique()) < 2:
            st.error(f"Only {len(df['ds'].unique())} unique date(s) found. Need at least 2 for forecasting.")
            return None
        
        # Warn about granularity mismatch
        if granularity == 'Weekly' and len(df['ds'].unique()) < 7:
            st.warning("Weekly granularity may not be suitable for monthly or yearly data. Consider 'Monthly' or 'Yearly'.")
        
        df.set_index('ds', inplace=True)
        resampling_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Half-Yearly': '6M', 'Yearly': 'Y'}
        if granularity in resampling_map:
            df = df.resample(resampling_map[granularity]).mean()
            df = df.reset_index()
        st.write(f"After resampling ({granularity}, rows: {len(df)}):\n{df.head()}")
        
        if len(df) < 2:
            st.error(f"After resampling, only {len(df)} row(s) remain. Need at least 2 non-NaN rows.")
            return None
        
        if df['y'].isna().all():
            st.error("All target values are NaN after resampling.")
            return None
        if df['y'].isna().any():
            st.warning(f"{df['y'].isna().sum()} NaN values in target after resampling. Filling with mean.")
            df['y'] = df['y'].fillna(df['y'].mean())
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing time series: {str(e)}\n{traceback.format_exc()}")
        return None

def create_qq_plot(data, col):
    sorted_data = np.sort(data[col].dropna())
    if len(sorted_data) == 0:
        return None
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
    slope = np.std(sorted_data)
    intercept = np.mean(sorted_data)
    line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Data Points', marker=dict(color='teal')))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Normal Line', line=dict(dash='dash', color='red')))
    fig.update_layout(
        title=f'Q-Q Plot for {col}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        showlegend=True
    )
    return fig

def generate_conclusion(df, num_col, cat_col):
    summary_stats = df[num_col].describe()
    if '50%' not in summary_stats:
        return f"**Analysis Conclusion**: Insufficient data for {num_col} to compute detailed statistics."
    max_category = df.loc[df[num_col].idxmax(), cat_col] if cat_col in df.columns and not df[num_col].empty else "N/A"
    skew = abs(summary_stats['mean'] - summary_stats['50%']) > summary_stats['std'] / 2
    conclusion = f"""
    **Analysis Conclusion**:
    - **Average {num_col}**: {summary_stats['mean']:.2f} (Std: {summary_stats['std']:.2f})
    - **Range**: {summary_stats['min']:.2f} to {summary_stats['max']:.2f}
    - **Top Category**: '{max_category}' has the highest {num_col}
    - **Distribution**: {'Skewed' if skew else 'Relatively normal'} based on mean-median difference
    - **Insight**: {f'High variability in {num_col} suggests diverse data points.' if summary_stats['std'] > summary_stats['mean'] / 2 else f'Low variability in {num_col} indicates consistent data.'}
    """
    return conclusion

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return rmse, mae, mape

def analysis_mode(df):
    st.header("Data Analysis")
    st.subheader("Full Data Table")
    page_size = 10
    total_rows = len(df)
    max_pages = (total_rows + page_size - 1) // page_size
    col1, col2 = st.columns([2, 3])
    with col1:
        current_page = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
    with col2:
        st.write(f"Showing page {current_page} of {max_pages}")
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, hide_index=False)

    if st.checkbox("Show Full Dataset"):
        st.dataframe(df, use_container_width=True, hide_index=False, height=400)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Data Types")
    st.dataframe(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes}), use_container_width=True, hide_index=True)

    st.header("Visualizations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols or not numeric_cols:
        st.warning("CSV must have at least one categorical and one numerical column.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            cat_col = st.selectbox("Categorical Column", categorical_cols)
        with col2:
            num_col = st.selectbox("Numerical Column", numeric_cols)
        bins = st.slider("Histogram Bins", 5, 50, 10, step=5)

        st.subheader("Pie Chart")
        value_counts = df.groupby(cat_col)[num_col].sum().reset_index()
        fig1 = px.pie(value_counts, names=cat_col, values=num_col, title=f'Distribution of {num_col} by {cat_col}')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Histogram")
        fig2 = px.histogram(df, x=num_col, nbins=bins, title=f'Distribution of {num_col}')
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Bar Chart")
        fig3 = px.bar(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Scatter Plot")
        if len(numeric_cols) >= 2:
            num_col2 = st.selectbox("Second Numerical Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_num")
            fig4 = px.scatter(df, x=num_col, y=num_col2, color=cat_col, title=f'{num_col} vs {num_col2}')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Scatter plot requires at least two numerical columns.")

        st.subheader("Box Plot")
        fig5 = px.box(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Q-Q Plot")
        fig6 = create_qq_plot(df, num_col)
        if fig6:
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning(f"No valid data for Q-Q plot of {num_col}.")

        st.header("Conclusion")
        conclusion = generate_conclusion(df, num_col, cat_col)
        st.markdown(conclusion)

def forecasting_mode(df):
    st.header("Data Forecasting")
    st.subheader("Time Series Forecasting")
    with st.sidebar:
        st.subheader("Forecasting Configuration")
        date_cols = [col for col in df.columns if is_date_column(df[col])]
        if not date_cols:
            st.error("No date-like columns found. Ensure your date column (e.g., '_YearMonth' with format YYYYMM) exists.")
            return
        default_date = '_YearMonth' if '_YearMonth' in date_cols else date_cols[0]
        date_col = st.selectbox("Date Column", date_cols, index=date_cols.index(default_date), help="Select a date column (e.g., '_YearMonth' in YYYYMM format)")
        target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        default_target = 'MonthlyRate' if 'MonthlyRate' in target_cols else target_cols[0]
        target_col = st.selectbox("Target Column", target_cols, index=target_cols.index(default_target), help="Select a numerical column (e.g., 'MonthlyRate')")
        granularity = st.selectbox("Forecast Granularity", ["Weekly", "Monthly", "Quarterly", "Half-Yearly", "Yearly"], help="Choose the time period for forecasting")
        model_type = st.selectbox("Model", ["Prophet", "ARIMA"], help="Select forecasting model")
        forecast_horizon = st.number_input("Forecast Horizon (units)", min_value=1, max_value=365, value=30, help="Number of periods to forecast")
        if model_type == "ARIMA":
            p = st.slider("AR Order (p)", 0, 5, 5)
            d = st.slider("Differencing (d)", 0, 2, 1)
            q = st.slider("MA Order (q)", 0, 5, 0)

    if st.button("Generate Forecast"):
        with st.spinner("Training model..."):
            try:
                processed_df = preprocess_time_series(df, date_col, target_col, granularity)
                if processed_df is None:
                    return
                train_size = int(len(processed_df) * 0.8)
                if train_size < 1:
                    st.error("Training set empty after split. Need more data.")
                    return
                train = processed_df[:train_size]
                test = processed_df[train_size:]

                if model_type == "Prophet":
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                    model.fit(train)
                    freq_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Half-Yearly': '6M', 'Yearly': 'Y'}
                    future = model.make_future_dataframe(periods=forecast_horizon, freq=freq_map[granularity])
                    forecast = model.predict(future)
                    y_pred = forecast['yhat'].iloc[train_size:train_size+len(test)]
                    rmse, mae, mape = calculate_metrics(test['y'], y_pred)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                    fig.update_layout(title=f"Prophet Forecast (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)",
                                      xaxis_title="Date", yaxis_title=target_col, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
                    st.download_button("Download Forecast", forecast_df.to_csv(index=False).encode('utf-8'),
                                      file_name="forecast.csv", mime="text/csv")

                elif model_type == "ARIMA":
                    model = ARIMA(train['y'], order=(p, d, q))
                    model_fit = model.fit()
                    y_pred = model_fit.forecast(steps=len(test))
                    forecast = model_fit.forecast(steps=len(test)+forecast_horizon)
                    rmse, mae, mape = calculate_metrics(test['y'], y_pred)

                    freq_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Half-Yearly': '6M', 'Yearly': 'Y'}
                    dates = pd.date_range(start=train['ds'].iloc[-1], periods=len(test)+forecast_horizon+1, freq=freq_map[granularity])[1:]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))
                    fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines', name='Forecast'))
                    fig.update_layout(title=f"ARIMA Forecast (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)",
                                      xaxis_title="Date", yaxis_title=target_col, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    forecast_df = pd.DataFrame({'Date': dates, 'Forecast': forecast})
                    st.download_button("Download Forecast", forecast_df.to_csv(index=False).encode('utf-8'),
                                      file_name="forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}\n{traceback.format_exc()}")

def main():
    st.title("Data Analysis & Forecasting Tool")
    st.markdown("Analyze or forecast your CSV data with customizable options.")

    with st.sidebar:
        st.header("Mode Selection")
        mode = st.radio("Select Mode", ["Analysis", "Forecasting"])

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns!")
            if mode == "Analysis":
                analysis_mode(df)
            else:
                forecasting_mode(df)

if __name__ == "__main__":
    main()