import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.markdown("""
<div style="
background:linear-gradient(90deg,#0ea5e9,#2563eb,#7c3aed);
padding:25px;
border-radius:20px;
color:white;
text-align:center;
margin-bottom:20px;">
<h1>📊 Diabetes Outlier & Analysis Dashboard</h1>
<p>Explore outliers, analyze patterns, interact with charts, and evaluate ML models</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("healthcarediabetes.csv")

    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

    return df

df = load_data()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div style="
background:linear-gradient(90deg,#2563eb,#0ea5e9);
padding:25px;
border-radius:20px;
color:white;
text-align:center;
margin-bottom:20px;">
<h1>📊 Diabetes Outlier & Analysis Dashboard</h1>
<p>Explore outliers, analyze patterns, interact with charts, and evaluate ML models</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Controls")
col_choice = st.sidebar.selectbox("Select a column", df.columns[:-1])
show_missing = st.sidebar.checkbox("Show missing rows")

# ---------------------------------------------------------
# NAVIGATION MENU
# ---------------------------------------------------------
st.markdown("### 📌 Navigation")

nav = st.columns(6)
page = None

with nav[0]:
    if st.button("Home"):
        page = "home"

with nav[1]:
    if st.button("Cleaning"):
        page = "cleaning"

with nav[2]:
    if st.button("Outliers"):
        page = "outliers"

with nav[3]:
    if st.button("Analysis"):
        page = "analysis"

with nav[4]:
    if st.button("Interactive"):
        page = "interactive"

with nav[5]:
    if st.button("Model"):
        page = "model"

if page is None:
    page = "home"

st.markdown("---")

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "home":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    if show_missing:
        st.subheader("Rows with Missing Values")
        st.dataframe(df[df.isna().any(axis=1)])

# ---------------------------------------------------------
# CLEANING PAGE
# ---------------------------------------------------------
elif page == "cleaning":
    st.subheader("Data Cleaning Overview")

    st.write("""
    - Replaced invalid zeros with NaN  
    - Cleaned columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI  
    - Missing values kept for analysis, dropped only for ML  
    """)

    st.write("Missing Values Count")
    st.dataframe(df.isna().sum())

# ---------------------------------------------------------
# OUTLIERS PAGE
# ---------------------------------------------------------
elif page == "outliers":

    # Histogram
    st.subheader(f"Histogram of {col_choice}")
    fig, ax = plt.subplots()
    sns.histplot(df[col_choice].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # IQR
    st.subheader(f"IQR Outliers for {col_choice}")
    Q1 = df[col_choice].quantile(0.25)
    Q3 = df[col_choice].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers_iqr = df[(df[col_choice] < lower) | (df[col_choice] > upper)]

    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df[col_choice], ax=ax2)
    st.pyplot(fig2)

    st.write(f"**IQR Range:** {lower:.2f} to {upper:.2f}")
    st.dataframe(outliers_iqr[[col_choice]].dropna().head(20))

    st.markdown("---")

    # Z-score
    st.subheader(f"Z-score Outliers for {col_choice}")
    df_clean = df.dropna(subset=[col_choice])
    z_scores = zscore(df_clean[col_choice])
    z_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.5)

    outliers_z = df_clean[np.abs(z_scores) > z_threshold]

    fig3, ax3 = plt.subplots()
    ax3.scatter(df_clean.index, df_clean[col_choice], alpha=0.6)
    ax3.scatter(outliers_z.index, outliers_z[col_choice], color="red")
    st.pyplot(fig3)

    st.write(f"**Z-score Outliers:** {outliers_z.shape[0]}")
    st.dataframe(outliers_z[[col_choice]].head(20))

# ---------------------------------------------------------
# ANALYSIS PAGE
# ---------------------------------------------------------
elif page == "analysis":

    # Outcome distribution
    st.subheader("Outcome Distribution")
    fig4, ax4 = plt.subplots()
    sns.countplot(x="Outcome", data=df, ax=ax4)
    st.pyplot(fig4)

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

    st.markdown("---")

    # Distribution by outcome
    st.subheader(f"{col_choice} Distribution by Outcome")
    fig6, ax6 = plt.subplots()
    sns.kdeplot(data=df, x=col_choice, hue="Outcome", fill=True, ax=ax6)
    st.pyplot(fig6)

    st.markdown("---")

    # BMI categories
    st.subheader("BMI Category Distribution")
    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,100],
                                labels=["Underweight","Normal","Overweight","Obese"])
    fig7, ax7 = plt.subplots()
    sns.countplot(x="BMI_Category", data=df, ax=ax7)
    st.pyplot(fig7)

    st.markdown("---")

    # Age groups
    st.subheader("Age Group Distribution")
    df["Age_Group"] = pd.cut(df["Age"], bins=[20,30,40,50,60,80],
                             labels=["20-30","30-40","40-50","50-60","60+"])
    fig8, ax8 = plt.subplots()
    sns.countplot(x="Age_Group", hue="Outcome", data=df, ax=ax8)
    st.pyplot(fig8)

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance (Random Forest)")
    df_fi = df.dropna()
    numeric_cols = df_fi.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("Outcome")

    X_fi = df_fi[numeric_cols]
    y_fi = df_fi["Outcome"]

    model_fi = RandomForestClassifier()
    model_fi.fit(X_fi, y_fi)

    importances = pd.Series(model_fi.feature_importances_, index=X_fi.columns)
    fig9, ax9 = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax9)
    st.pyplot(fig9)

# ---------------------------------------------------------
# INTERACTIVE PAGE
# ---------------------------------------------------------
elif page == "interactive":

    st.subheader("Interactive Scatter Plot")
    fig10 = px.scatter(df, x="Glucose", y="BMI", color="Outcome")
    st.plotly_chart(fig10)

    st.markdown("---")

    st.subheader("Interactive Histogram")
    fig11 = px.histogram(df, x=col_choice, nbins=30, color="Outcome")
    st.plotly_chart(fig11)

    st.markdown("---")

    st.subheader("Download Cleaned Data")
    st.download_button("Download CSV", df.to_csv().encode(), "cleaned_data.csv")

# ---------------------------------------------------------
# MODEL PAGE
# ---------------------------------------------------------
elif page == "model":

    st.subheader("Model Performance (Random Forest)")

    df_model = df.dropna()
    numeric_cols_model = df_model.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_model.remove("Outcome")

    X = df_model[numeric_cols_model]
    y = df_model["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig12, ax12 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax12)
    st.pyplot(fig12)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.caption("Dashboard built with Streamlit using the Pima Indians Diabetes dataset.")
