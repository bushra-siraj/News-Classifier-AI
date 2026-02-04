import streamlit as st
import joblib
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
from streamlit_lottie import st_lottie

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="BBC News AI Dashboard", layout="wide", page_icon="📰")

# --- 2. ASSET LOADING (Models & Animations) ---
@st.cache_resource
def load_assets():
    # Load ML components
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('bbc-text.csv').dropna()
    return model, tfidf, le, df

# 1. Improved loader with error handling
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5) # Add a timeout so it doesn't hang
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None 
    
model, tfidf, le, df = load_assets()
lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_news = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

# --- 3. SIDEBAR ---
with st.sidebar:
   if lottie_ai:
    st_lottie(lottie_ai, height=200)
    st.title("Settings & Info")
    st.success("**Top Model:** Logistic Regression")
    st.metric(label="Best Accuracy", value="95.51%")
    st.divider()
    st.write("Created for: Data Science Project")

   else:
    st.title("Settings & Info")
    st.success("**Top Model:** Logistic Regression")
    st.metric(label="Best Accuracy", value="95.51%")
    st.divider()
    st.write("Created for: Data Science Exam 2026")

# --- 4. MAIN INTERFACE ---
st.title("📰 News Classification & Insights")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Data Insights", "🧠 AI Predictor", "☁️ Word Explorer"])

# --- TAB 1: DATA INSIGHTS (Using saved images) ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Univariate Analysis")
        st.image('news_category_distribution.png', caption="Total Count per Category")
        st.image('word_count_distribution.png', caption="Frequency of Article Lengths")
        
    with col2:
        st.subheader("Bivariate Analysis")
        st.image('avg_word_count_per_category.png', caption="Average Length per Category")
        st.image('boxplot_word_count_tech_sport.png', caption="Comparison: Tech vs Sport")
    
    st.divider()
    st.subheader("Overall Data Trends")
    st.image('violin_plot_word_count_category.png', use_container_width=True)

# --- TAB 2: AI PREDICTOR ---
with tab2:
    col_text, col_anim = st.columns([2, 1])
    
    with col_anim:
        if lottie_news:
            st_lottie(lottie_news, height=200)
    
    with col_text:
        st.header("Live News Classifier")
        user_input = st.text_area("Paste a news snippet here:", height=150, placeholder="Apple announces new iPhone...")

    if st.button("Analyze and Classify"):
        if user_input:
            # Transformation and Prediction
            vec = tfidf.transform([user_input])
            pred_num = model.predict(vec)
            category = le.inverse_transform(pred_num)[0]
            
            st.markdown(f"### The AI thinks this is: **{category.upper()}**")
            st.balloons()
        else:
            st.warning("Please enter text before classifying.")

# --- TAB 3: WORD EXPLORER (Word Clouds) ---
with tab3:
    st.header("Category Vocabulary")
    selected_cat = st.selectbox("Choose a category to see its signature words:", df['category'].unique())
    
    if selected_cat:
        # Create WordCloud
        category_text = " ".join(df[df['category'] == selected_cat]['text'])
        wc = WordCloud(width=1000, height=500, background_color='black', colormap='Set2').generate(category_text)
        
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)