import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")

# Load dataset (safe loading)
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# ---------------------------
# DATA CLEANING (REAL)
# ---------------------------
df = df.drop_duplicates()

# Fill missing values (numeric only)
df.fillna(df.select_dtypes(include="number").mean(), inplace=True)

# ---------------------------
# DATASET OVERVIEW
# ---------------------------
st.header("Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())
st.write(df.dtypes)

# ---------------------------
# DATA CLEANING REPORT
# ---------------------------
st.header("Data Cleaning")

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Duplicate Records (after cleaning)")
st.write(df.duplicated().sum())

# ---------------------------
# STATISTICS
# ---------------------------
st.header("Statistical Summary")
st.write(df.describe())

# ---------------------------
# SIDEBAR FILTERS (IMPORTANT)
# ---------------------------
st.sidebar.header("Filters")

numeric_cols = df.select_dtypes(include="number").columns

selected_col = st.sidebar.selectbox("Select Numeric Column", numeric_cols)

min_val, max_val = float(df[selected_col].min()), float(df[selected_col].max())

range_val = st.sidebar.slider(
    "Filter Range",
    min_val,
    max_val,
    (min_val, max_val)
)

df = df[(df[selected_col] >= range_val[0]) & (df[selected_col] <= range_val[1])]

# ---------------------------
# HISTOGRAM
# ---------------------------
st.header("Histogram")

fig, ax = plt.subplots()
ax.hist(df[selected_col], bins=20)
ax.set_xlabel(selected_col)
st.pyplot(fig)

# ---------------------------
# BOX PLOT
# ---------------------------
st.header("Box Plot")

fig2, ax2 = plt.subplots()
ax2.boxplot(df[selected_col])
st.pyplot(fig2)

# ---------------------------
# CORRELATION HEATMAP (IMPORTANT)
# ---------------------------
st.header("Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# ---------------------------
# SCATTER PLOT
# ---------------------------
st.header("Scatter Plot")

x = st.selectbox("X-axis", numeric_cols)
y = st.selectbox("Y-axis", numeric_cols)

fig4, ax4 = plt.subplots()
ax4.scatter(df[x], df[y])
ax4.set_xlabel(x)
ax4.set_ylabel(y)
st.pyplot(fig4)

# ---------------------------
# INSIGHTS
# ---------------------------
st.header("Insights")

st.write("""
- Dataset loaded successfully.
- Duplicate records removed.
- Missing values handled using mean imputation.
- Interactive filters applied.
- Distribution visualized using histogram and box plot.
- Correlation heatmap shows relationships between features.
- Scatter plot shows variable relationships.
""")
