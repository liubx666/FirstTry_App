import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 加载模型和向量器
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 清洗函数
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# Streamlit 页面
st.title("🕵️ 假新闻识别器 Fake News Detector")
news = st.text_area("请输入新闻内容")

if st.button("判断真假"):
    if news.strip() == "":
        st.warning("请输入内容")
    else:
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        prob = model.predict_proba(vec)[0][1]
        label = "❌ 假新闻" if prob >= 0.5 else "✅ 真实新闻"
        st.markdown(f"### {label}")
        st.markdown(f"**可信度评分：{round(prob * 100, 2)}%**")
