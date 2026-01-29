import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ---------------- Page Config ----------------
st.set_page_config(page_title="End-to-End ML App", layout="wide")
st.title("End-to-End ML Classification Application")

# ---------------- Sidebar: Model Selection ----------------
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Algorithm",
    ["Decision Tree", "Random Forest", "KNN", "Naive Bayes"]
)

# ---------------- Sidebar: Hyperparameters ----------------
if model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("No. of Trees", 50, 300, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_name == "KNN":
    k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)
    distance = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan"])

# Naive Bayes has no major hyperparameters

# ---------------- Step 1: Upload Dataset ----------------
st.header("Step 1: Data Ingestion")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")
    st.dataframe(df.head())

# ---------------- Step 2: EDA ----------------
if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    st.write("Shape:", df.shape)
    st.write("Missing Values:\n", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- Step 3: Data Cleaning ----------------
if df is not None:
    st.header("Step 3: Data Cleaning")
    strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Drop Rows"])
    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.success("Data cleaning completed")

# ---------------- Step 4: Model Training ----------------
if df is not None:
    st.header("Step 4: Train Model")
    target = st.selectbox("Select Target Column", df_clean.columns)

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = X.select_dtypes(include=np.number)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # -------- Model Selection --------
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=k, metric=distance)

    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"{model_name} Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
