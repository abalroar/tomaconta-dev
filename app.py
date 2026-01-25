import streamlit as st
import pandas as pd
import pickle
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from utils.ifdata_extractor import gerar_periodos, processar_todos_periodos
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

st.set_page_config(page_title="fica de olho", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100;200;300;400;500;600;700&display=swap');

    /* ========== ESCONDE HEADER MAS PRESERVA SIDEBAR ========== */
    .stApp > header {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Esconde status widget mas NÃO o botão de collapse da sidebar */
    [data-testid="stStatusWidget"] > div > div > div > button {
        display: none !important;
    }

    /* GARANTE que o botão de toggle da sidebar SEMPRE apareça */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: relative !important;
        left: auto !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    * {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    html, body, [class*="css"], div, span, p, label, input, select, textarea, button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .sidebar-logo img {
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100px;
        margin: 0 auto;
        display: block;
    }

    .sidebar-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 300;
        color: #1f77b4;
        margin: 0.5rem 0 0.2rem 0;
        line-height: 1.2;
    }

    .sidebar-subtitle {
        text-align: center;
        font-size: 0.85rem;
        color: #666;
        margin: 0 0 0.2rem 0;
        line-height: 1.3;
    }

    .sidebar-author {
        text-align: center;
        font-size: 0.75rem;
        color: #888;
        font-style: italic;
        margin: 0 0 1rem 0;
    }

    button[kind="primary"], button[kind="secondary"], .stButton button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
    }

    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 300 !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 400 !important;
    }

    .stMarkdown, .stMarkdown * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    .streamlit-expanderHeader {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
    }

    .stCaption {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 400 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }

    .feature-card h4 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }

    /* ========== FIX SOBREPOSIÇÃO SIDEBAR ========== */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }

    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSegmentedControl"] {
        margin-bottom: 1.5rem !important;
        margin-top: 0.5rem !important;
    }

    [data-testid="stExpander"] {
        margin-top: 1rem !important;
        clear: both !important;
    }

    [data-testid="stSidebar"] .row-widget {
        margin-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)
