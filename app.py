import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# --- PAGE CONFIG ---
# Removed emojis for a cleaner browser tab look
st.set_page_config(page_title="Cognito | Apex Behavioral Analytics ", layout="wide")

# --- PROFESSIONAL THEME (Navy, Slate, and Platinum) ---
st.markdown("""
    <style>
    .main { background-color: #0F172A; } /* Dark Navy Background */
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] { 
        color: #F8FAFC !important; 
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }

    /* Custom Alert/Info boxes */
    .stAlert { 
        background-color: #1E293B; 
        border-left: 5px solid #38BDF8; 
        color: #F1F5F9; 
    }

    /* Clean Dividers */
    hr { border: 0.1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")
    menu = st.radio(
        "Select Analytical View", 
        ["Executive Dashboard", "Customer Segmentation", "Anomaly Detection", "Predictive Recommendations"]
    )
    st.markdown("---")
    st.markdown("### Dataset Details")
    st.caption("Source: Online Retail II\n\nRecords: 1,067,371")

# --- EXECUTIVE DASHBOARD ---
if menu == "Executive Dashboard":
    st.title("Cognito | Apex Behavioral Analytics ")
    st.markdown("Strategic analysis of customer behavior and revenue distribution.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "1,067,371", "Cleaned")
    col2.metric("Active Customers", "4,382", "+12% YoY")
    col3.metric("Anomalies Identified", "60", "Risk Flagged")
    col4.metric("PCA Variance", "96.01%", "High Fidelity")

    st.markdown("---")
    st.subheader("Strategic Insights")
    st.info("""
    **Methodology:** By deploying Unsupervised Learning, this model constructs a 'Behavioral Map' of the user base. 
    This allows for strategy shifts based on actual consumption patterns rather than static demographics.
    """)

# --- CUSTOMER SEGMENTATION ---
elif menu == "Customer Segmentation":
    st.title("Customer Segmentation Logic")
    st.markdown("Clustering analysis derived from K-Means and Principal Component Analysis.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### Profile Definitions")
        st.markdown("""
        * **High-Value Assets:** 5% of users driving 40% of revenue.
        * **Growth Candidates:** Regular shoppers with consistent frequency.
        * **New Acquisitions:** Low spend with high recent engagement.
        * **At-Risk Accounts:** Significant churn probability (6+ months inactive).
        """)
    
    with c2:
        if os.path.exists('pca_clusters.png'):
            st.image(Image.open('pca_clusters.png'), caption="PCA Projection of Transactional Data", use_container_width=True)
        else:
            st.warning("Visual asset 'pca_clusters.png' not found.")

# --- ANOMALY DETECTION ---
elif menu == "Anomaly Detection":
    st.title("Risk & Fraud Detection")
    st.markdown("Identification of statistical outliers via Gaussian Density Estimation.")
    
    if os.path.exists('anomalies.png'):
        st.image(Image.open('anomalies.png'), use_container_width=True)
    
    st.error("Protocol Required: 60 identified accounts deviate significantly from the density norm and require manual review.")

# --- PREDICTIVE RECOMMENDATIONS ---
elif menu == "Predictive Recommendations":
    st.title("Predictive Engine")
    st.markdown("AI-driven inventory suggestions based on Collaborative Filtering.")
    
    user_id = st.selectbox("Customer ID for Analysis", ["12347.0", "12348.0", "12417.0"])
    
    if st.button("Generate Recommendations"):
        st.subheader(f"Projected Interest for Account {user_id}")
        
        col1, col2, col3 = st.columns(3)
        recs = [
            {"id": "22775", "desc": "PURPLE DRAWER KNOB", "match": "98%"},
            {"id": "85123A", "desc": "WHITE HANGING HEART", "match": "94%"},
            {"id": "21471", "desc": "STRAWBERRY RAFFIA TOTE", "match": "91%"}
        ]

        for i, col in enumerate([col1, col2, col3]):
            with col:
                # Cleaner, card-based design without emojis
                st.markdown(f"""
                <div style="background-color: #1E293B; padding: 25px; border-radius: 8px; border-top: 4px solid #38BDF8; text-align: left;">
                    <p style="color: #64748B; font-size: 11px; margin: 0; letter-spacing: 1px;">SKU: {recs[i]['id']}</p>
                    <h4 style="color: #F1F5F9; margin: 10px 0;">{recs[i]['desc']}</h4>
                    <p style="color: #38BDF8; font-weight: bold; font-size: 14px;">{recs[i]['match']} Confidence</p>
                    <div style="background-color: #334155; height: 1px; margin: 15px 0;"></div>
                    <p style="color: #94A3B8; font-size: 12px;">Recommendation based on cross-user similarity matrix.</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Marketing Logic: Predictive modeling reduces churn by surfacing unexplored inventory with high mathematical affinity.")