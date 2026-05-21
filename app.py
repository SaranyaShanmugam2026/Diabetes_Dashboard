import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Streamlit page setup
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv("healthcarediabetes.csv")

    # Replace invalid zeros with NaN
    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

    return df

df = load_data()

st.title("📊 Diabetes Dataset – Outlier Exploration Dashboard")
st.write("This interactive dashboard helps explore outliers using Histogram, Boxplot (IQR), and Z‑score methods.")

# Sidebar controls
st.sidebar.header("Controls")
col_choice = st.sidebar.selectbox("Select a column to explore", df.columns[:-1])
show_missing = st.sidebar.checkbox("Show rows with missing values")

# Show missing values
if show_missing:
    st.subheader("Rows with Missing Values")
    st.dataframe(df[df.isna().any(axis=1)])

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

st.markdown("---")

# Histogram
st.subheader(f"Histogram of {col_choice}")
fig, ax = plt.subplots()
sns.histplot(df[col_choice].dropna(), kde=True, ax=ax)
ax.set_xlabel(col_choice)
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.markdown("---")

# Boxplot + IQR
st.subheader(f"Boxplot and IQR Outliers for {col_choice}")

Q1 = df[col_choice].quantile(0.25)
Q3 = df[col_choice].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df[col_choice] < lower_bound) | (df[col_choice] > upper_bound)]

c1, c2 = st.columns(2)

with c1:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df[col_choice], ax=ax2)
    ax2.set_xlabel(col_choice)
    st.pyplot(fig2)

with c2:
    st.write(f"**IQR Range:** {lower_bound:.2f} to {upper_bound:.2f}")
    st.write(f"**Number of IQR Outliers:** {outliers_iqr[col_choice].notna().sum()}")
    st.dataframe(outliers_iqr[[col_choice]].dropna().head(20))

st.markdown("---")

# Z-score outliers
st.subheader(f"Z-score Outliers for {col_choice}")

df_clean = df.dropna(subset=[col_choice])
z_scores = zscore(df_clean[col_choice])
z_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.5)

outliers_z_mask = np.abs(z_scores) > z_threshold
outliers_z = df_clean[outliers_z_mask]

fig3, ax3 = plt.subplots()
ax3.scatter(df_clean.index, df_clean[col_choice], label="Normal", alpha=0.6)
ax3.scatter(outliers_z.index, outliers_z[col_choice], color="red", label="Outliers")
ax3.set_xlabel("Index")
ax3.set_ylabel(col_choice)
ax3.set_title(f"{col_choice} with Z-score Outliers (|Z| > {z_threshold})")
ax3.legend()
st.pyplot(fig3)

st.write(f"**Number of Z-score Outliers:** {outliers_z.shape[0]}")
st.dataframe(outliers_z[[col_choice]].head(20))

st.markdown("---")

# Outcome distribution
st.subheader("Outcome Distribution")
fig4, ax4 = plt.subplots()
sns.countplot(x="Outcome", data=df, ax=ax4)
ax4.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
ax4.set_ylabel("Count")
st.pyplot(fig4)

# 1. Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("---")

# 2. Distribution by Outcome
st.subheader(f"{col_choice} Distribution by Outcome")
fig, ax = plt.subplots()
sns.kdeplot(data=df, x=col_choice, hue="Outcome", fill=True, ax=ax)
st.pyplot(fig)

st.markdown("---")

# 3. BMI Category Analysis
st.subheader("BMI Category Distribution")
df["BMI_Category"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)
fig, ax = plt.subplots()
sns.countplot(x="BMI_Category", data=df, ax=ax)
st.pyplot(fig)

st.markdown("---")

# 4. Age Group Analysis
st.subheader("Age Group Distribution")
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[20,30,40,50,60,80],
    labels=["20-30","30-40","40-50","50-60","60+"]
)
fig, ax = plt.subplots()
sns.countplot(x="Age_Group", hue="Outcome", data=df, ax=ax)
st.pyplot(fig)

st.markdown("---")

# 5. Feature Importance (Random Forest)
st.subheader("Feature Importance (Random Forest)")
df_clean = df.dropna()
X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
importances.sort_values().plot(kind="barh", ax=ax)
st.pyplot(fig)



st.caption("Dashboard built with Streamlit using the Pima Indians Diabetes dataset.")
