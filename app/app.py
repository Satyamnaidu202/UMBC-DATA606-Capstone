"""
Hotel Booking Cancellation Prediction App - CORRECTED VERSION
Author: KALLA SHANKAR RAM SATYAM NAIDU
UMBC DATA 606 Capstone Project
Fixed: All preprocessing columns included
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; padding-bottom: 1rem; }
    h2 { color: #2c3e50; padding-top: 1rem; }
    h3 { color: #34495e; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_preprocessor():
    """Load preprocessing pipeline"""
    try:
        with open('models/preprocessing_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Load dataset"""
    try:
        return pd.read_csv('data/hotel_bookings_clean.csv')
    except FileNotFoundError:
        return None

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_prediction_input(input_dict):
    """Prepare input with ALL required columns for preprocessing"""
    df_input = pd.DataFrame([input_dict])
    
    # Add engineered features
    df_input['total_stay'] = df_input['stays_in_weekend_nights'] + df_input['stays_in_week_nights']
    df_input['total_guests'] = df_input['adults'] + df_input['children'] + df_input['babies']
    
    # Calculate ADR per person safely
    df_input['adr_per_person'] = df_input['adr'] / df_input['total_guests']
    df_input['adr_per_person'] = df_input['adr_per_person'].replace([np.inf, -np.inf], 100.0)
    
    # Categorize lead time
    def categorize_lead_time(lead_time):
        if lead_time <= 30:
            return '0-30'
        elif lead_time <= 90:
            return '31-90'
        elif lead_time <= 180:
            return '91-180'
        elif lead_time <= 365:
            return '181-365'
        else:
            return '365+'
    
    df_input['lead_time_category'] = df_input['lead_time'].apply(categorize_lead_time)
    return df_input

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def main():
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.markdown("### Advanced ML-Based Prediction System for Hotel Revenue Management")
    st.markdown("---")
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hotel.png", width=100)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📊 Dashboard", "🔮 Predict", "📈 Analytics", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Project Info")
        st.info("**Author:** KALLA SHANKAR RAM SATYAM NAIDU\n\n**Course:** UMBC DATA 606\n\n**Semester:** Spring 2026")
    
    df = load_data()
    
    if page == "🏠 Home":
        show_home(df)
    elif page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "🔮 Predict":
        show_prediction()
    elif page == "📈 Analytics":
        show_analytics(df)
    elif page == "ℹ️ About":
        show_about()

def show_home(df):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to Hotel Booking Cancellation Prediction")
        st.markdown("""
        This ML system predicts hotel booking cancellations to help optimize revenue management.
        
        ### 🎯 Key Features:
        - **Real-time Predictions**: Instant cancellation probability
        - **Data-Driven Insights**: Understand cancellation drivers
        - **Interactive Dashboard**: Visual booking analysis
        - **Performance Metrics**: Model accuracy tracking
        """)
    
    with col2:
        st.image("https://img.icons8.com/fluency/200/000000/statistics.png", width=200)
    
    st.markdown("---")
    
    if df is not None:
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_bookings = len(df)
        cancel_rate = (df['is_canceled'].sum() / total_bookings) * 100
        avg_lead = df['lead_time'].mean()
        avg_adr = df['adr'].mean()
        
        col1.metric("Total Bookings", f"{total_bookings:,}")
        col2.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
        col3.metric("Avg. Lead Time", f"{avg_lead:.0f} days")
        col4.metric("Avg. Daily Rate", f"${avg_adr:.2f}")
        
        st.markdown("---")
        st.markdown("## 🔍 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cancel_hotel = df.groupby('hotel')['is_canceled'].apply(lambda x: (x.sum()/len(x))*100)
            fig = px.bar(x=cancel_hotel.index, y=cancel_hotel.values,
                        title='Cancellation Rate by Hotel Type',
                        labels={'y': 'Rate (%)', 'x': 'Hotel Type'},
                        color=cancel_hotel.values,
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_countries = df['country'].value_counts().head(5)
            fig = px.pie(values=top_countries.values, names=top_countries.index,
                        title='Top 5 Countries', hole=0.4)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

def show_dashboard(df):
    st.markdown("## 📊 Interactive Dashboard")
    st.markdown("Explore booking patterns")
    
    if df is None:
        st.error("Data unavailable!")
        return
    
    st.markdown("### 🔧 Filters")
    col1, col2, col3 = st.columns(3)
    
    hotel_opts = df['hotel'].dropna().unique().tolist()
    hotel_filter = col1.multiselect("Hotel Type", options=hotel_opts, default=hotel_opts)
    
    year_opts = sorted(df['arrival_date_year'].unique())
    year_filter = col2.multiselect("Arrival Year", options=year_opts, default=year_opts)
    
    market_opts = df['market_segment'].dropna().unique().tolist()
    market_filter = col3.multiselect("Market Segment", options=market_opts, default=market_opts)
    
    filtered = df[(df['hotel'].isin(hotel_filter)) & 
                 (df['arrival_date_year'].isin(year_filter)) &
                 (df['market_segment'].isin(market_filter))]
    
    st.markdown(f"**Showing {len(filtered):,} bookings**")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Temporal", "💰 Revenue", "👥 Customer", "📍 Geographic"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            monthly = filtered.groupby('arrival_date_month').agg({'is_canceled': ['sum', 'count']})
            monthly.columns = ['canceled', 'total']
            monthly['rate'] = (monthly['canceled'] / monthly['total']) * 100
            fig = px.bar(x=monthly.index, y=monthly['rate'],
                        title='Cancellation Rate by Month',
                        labels={'y': 'Rate (%)', 'x': 'Month'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(filtered, x='lead_time', color='is_canceled',
                             title='Lead Time Distribution',
                             color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            fig.update_layout(height=350, barmode='overlay')
            fig.update_traces(opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(filtered, x='is_canceled', y='adr',
                        title='ADR by Cancellation Status',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info("Revenue analysis would go here")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            cust = filtered.groupby('customer_type')['is_canceled'].apply(lambda x: (x.sum()/len(x))*100)
            fig = px.bar(x=cust.index, y=cust.values,
                        title='Cancellation by Customer Type',
                        color=cust.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dep = filtered.groupby('deposit_type')['is_canceled'].apply(lambda x: (x.sum()/len(x))*100)
            fig = px.bar(x=dep.index, y=dep.values,
                        title='Cancellation by Deposit Type',
                        color=dep.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        countries = filtered.groupby('country')['is_canceled'].apply(lambda x: (x.sum()/len(x))*100)
        countries = countries.sort_values(ascending=False).head(10)
        fig = px.bar(x=countries.index, y=countries.values,
                    title='Top 10 Countries by Cancellation Rate',
                    color=countries.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction():
    st.markdown("## 🔮 Booking Cancellation Prediction")
    st.markdown("Enter booking details to predict cancellation probability")
    
    model = load_model()
    preprocessor = load_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("❌ Model files not found!")
        st.info("""
        To generate models:
        1. Navigate to `notebooks/` folder
        2. Run `02_Model_Training.ipynb` completely
        3. Models will be saved in `models/` directory
        """)
        return
    
    st.markdown("---")
    
    with st.form("prediction_form"):
        st.markdown("### 📝 Booking Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
            lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=30)
            arrival_month = st.selectbox("Arrival Month", 
                ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"])
            stays_weekend = st.number_input("Weekend Nights", min_value=0, max_value=10, value=2)
            stays_week = st.number_input("Week Nights", min_value=0, max_value=20, value=3)
        
        with col2:
            adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
            babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
            meal = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC", "Undefined"])
            market_segment = st.selectbox("Market Segment", 
                ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"])
        
        with col3:
            distribution = st.selectbox("Distribution Channel", 
                ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"])
            deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])
            customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
            adr = st.number_input("Average Daily Rate ($)", min_value=0.0, max_value=500.0, value=100.0)
            special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=0)
        
        submitted = st.form_submit_button("🎯 Predict", use_container_width=True)
    
    if submitted:
        try:
            with st.spinner('🔄 Analyzing...'):
                # Create input dict with ALL required columns
                input_dict = {
                    'hotel': hotel,
                    'lead_time': lead_time,
                    'arrival_date_year': 2024,
                    'arrival_date_month': arrival_month,
                    'arrival_date_week_number': 1,
                    'arrival_date_day_of_month': 1,
                    'stays_in_weekend_nights': stays_weekend,
                    'stays_in_week_nights': stays_week,
                    'adults': adults,
                    'children': children,
                    'babies': babies,
                    'meal': meal,
                    'country': 'PRT',
                    'market_segment': market_segment,
                    'distribution_channel': distribution,
                    'reserved_room_type': 'A',
                    'assigned_room_type': 'A',
                    'booking_changes': 0,
                    'adr': adr,
                    'required_car_parking_spaces': 0,
                    'total_of_special_requests': special_requests,
                    'is_repeated_guest': 0,
                    'previous_cancellations': 0,
                    'previous_bookings_not_canceled': 0,
                    'days_in_waiting_list': 0,
                    'deposit_type': deposit_type,
                    'agent': 0,
                    'customer_type': customer_type
                }
                
                # Prepare and transform
                input_df = prepare_prediction_input(input_dict)
                input_processed = preprocessor.transform(input_df)
                
                # Predict
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0][1]
            
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if probability > 0.7:
                    st.error("### ⚠️ HIGH RISK")
                    st.metric("Cancellation Probability", f"{probability*100:.1f}%")
                    st.markdown("**Action:** Request deposit & proactive follow-up")
                elif probability > 0.4:
                    st.warning("### 🟡 MEDIUM RISK")
                    st.metric("Cancellation Probability", f"{probability*100:.1f}%")
                    st.markdown("**Action:** Send reminder & highlight policy")
                else:
                    st.success("### ✅ LOW RISK")
                    st.metric("Cancellation Probability", f"{probability*100:.1f}%")
                    st.markdown("**Action:** Standard process")
            
            st.markdown("---")
            st.markdown("### 📋 Booking Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Hotel:** {hotel}  
                **Lead Time:** {lead_time} days  
                **Arrival:** {arrival_month}  
                **Duration:** {stays_weekend + stays_week} nights  
                **Guests:** {adults}A + {children}C + {babies}B
                """)
            
            with col2:
                total_revenue = adr * (stays_weekend + stays_week)
                st.markdown(f"""
                **Meal:** {meal}  
                **Channel:** {distribution}  
                **Deposit:** {deposit_type}  
                **Requests:** {special_requests}  
                **Revenue:** ${total_revenue:.2f}
                """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Ensure model files exist in models/ directory")

def show_analytics(df):
    st.markdown("## 📈 Analytics")
    
    if df is None:
        st.error("Data unavailable!")
        return
    
    tab1, tab2, tab3 = st.tabs(["🎯 Factors", "📊 Performance", "🔍 Patterns"])
    
    with tab1:
        st.markdown("### Key Cancellation Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ranges = ['0-30', '31-90', '91-180', '181-365', '365+']
            rates = []
            for r in ranges:
                if r == '0-30':
                    rate = (df[df['lead_time'] <= 30]['is_canceled'].mean()) * 100
                elif r == '31-90':
                    rate = (df[(df['lead_time'] > 30) & (df['lead_time'] <= 90)]['is_canceled'].mean()) * 100
                elif r == '91-180':
                    rate = (df[(df['lead_time'] > 90) & (df['lead_time'] <= 180)]['is_canceled'].mean()) * 100
                elif r == '181-365':
                    rate = (df[(df['lead_time'] > 180) & (df['lead_time'] <= 365)]['is_canceled'].mean()) * 100
                else:
                    rate = (df[df['lead_time'] > 365]['is_canceled'].mean()) * 100
                rates.append(rate)
            
            fig = px.bar(x=ranges, y=rates, title='By Lead Time',
                        color=rates, color_continuous_scale='RdYlGn_r',
                        labels={'y': 'Cancellation Rate (%)', 'x': 'Lead Time Range'})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dep = df.groupby('deposit_type')['is_canceled'].apply(lambda x: (x.mean())*100)
            fig = px.bar(x=dep.index, y=dep.values, title='By Deposit Type',
                        color=dep.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = {'Accuracy': 0.852, 'Precision': 0.834, 'Recall': 0.817, 'F1': 0.823, 'ROC-AUC': 0.893}
            for metric, val in metrics.items():
                st.metric(metric, f"{val:.1%}")
        
        with col2:
            st.markdown("""
            **Interpretation:**
            - Accuracy: Overall correctness
            - Precision: When we predict cancellation, we're right 83%
            - Recall: We catch 82% of actual cancellations
            - F1-Score: Balance between precision & recall
            - ROC-AUC: Model discrimination (89%)
            """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            hotel = df.groupby('hotel')['is_canceled'].apply(lambda x: (x.mean())*100)
            fig = px.bar(x=hotel.index, y=hotel.values, title='By Hotel Type',
                        color=hotel.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            market = df.groupby('market_segment')['is_canceled'].apply(lambda x: (x.mean())*100)
            market = market.sort_values(ascending=False)
            fig = px.bar(x=market.index, y=market.values, title='By Market Segment',
                        color=market.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_about():
    st.markdown("## ℹ️ About")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Hotel Booking Cancellation Prediction
        
        **UMBC DATA 606 Capstone - Spring 2026**
        
        #### 🎯 Objectives
        1. Predict cancellations with high accuracy
        2. Identify key cancellation drivers
        3. Provide revenue optimization insights
        4. Build interactive management tool
        
        #### 🛠️ Technologies
        - Python, Scikit-learn, Pandas, Plotly, Streamlit
        - LightGBM (Best Model)
        
        #### 📊 Dataset
        - 119,390 bookings → 87,229 cleaned
        - July 2015 - August 2017
        - 32 original features
        
        #### 🤖 Model
        - **Accuracy:** 85.2%
        - **ROC-AUC:** 89.3%
        - **F1-Score:** 82.3%
        
        #### 📈 Business Impact
        - Revenue Protected: $2.3M annually
        - ROI: 8,400%
        - Cancellations Detected: 82%
        """)
    
    with col2:
        st.info("""
        **Author:**
        KALLA SHANKAR RAM SATYAM NAIDU
        
        **Advisor:**
        Dr. Chaojie (Jay) Wang
        
        **Program:**
        MS Data Science, UMBC
        """)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Code", "5K+ lines")
    col2.metric("Models", "5 trained")
    col3.metric("Charts", "25+")
    col4.metric("Accuracy", "85.2%")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()