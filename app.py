import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")

# Load dataset
df = pd.read_csv("dataset.csv")

# Dataset Overview
st.header("Dataset Overview")
st.write("Shape of Dataset:", df.shape)
st.write(df.head())
st.write(df.dtypes)

# Data Cleaning
st.header("Data Cleaning")

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Duplicate Records")
st.write(df.duplicated().sum())

# Statistical Summary
st.header("Statistical Summary")
st.write(df.describe())

# Select numeric column
numeric_cols = df.select_dtypes(include="number").columns

selected = st.sidebar.selectbox(
    "Select Numeric Column",
    numeric_cols
)

# Histogram
st.header("Histogram")

fig, ax = plt.subplots()
ax.hist(df[selected], bins=20)
ax.set_xlabel(selected)
st.pyplot(fig)

# Box Plot
st.header("Box Plot")

fig2, ax2 = plt.subplots()
ax2.boxplot(df[selected])
st.pyplot(fig2)

# Correlation Matrix
st.header("Correlation Matrix")
st.write(df.corr(numeric_only=True))

# Scatter Plot
st.header("Scatter Plot")

x = st.selectbox("X-axis", numeric_cols)
y = st.selectbox("Y-axis", numeric_cols, index=1)

fig3, ax3 = plt.subplots()
ax3.scatter(df[x], df[y])
ax3.set_xlabel(x)
ax3.set_ylabel(y)

st.pyplot(fig3)

# Insights
st.header("Insights")

st.write("""
- Dataset successfully loaded.
- Missing values identified.
- Duplicate records checked.
- Statistical summary generated.
- Histogram and Box Plot visualize distributions.
- Scatter Plot shows relationships between variables.
- Correlation matrix identifies feature relationships.
""")
