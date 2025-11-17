import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IBM HR Dashboard", layout="wide")
# ===================== STYLES: NEON ANIMATED + GLASS CARDS =====================
st.markdown(
    """
    <style>
    /* Animated neon gradient background (slow smooth) */
    .stApp {
      background: linear-gradient(120deg, #021B79, #0575E6, #00C9FF, #00FFB3);
      background-size: 800% 800%;
      animation: gradientMove 18s ease infinite;
      color: #eef6ff;
    }


@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.top-filter-container {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex; 
    gap: 20px;
    padding: 15px;
    border-radius: 14px;
    background: rgba(0,0,0,0.4);
    backdrop-filter: blur(6px);
    margin-bottom: 20px;
}

.kpi-card {
    border-radius:15px;
    padding:20px;
    text-align:center;
    margin-bottom:10px;
    backdrop-filter: blur(6px);
    transition: transform 0.2s;
}
.kpi-card:hover {
    transform: scale(1.05);
}

.chart-container {
    background-color: rgba(255,255,255,0.1);
    padding:10px;
    border-radius:12px;
}

.chart-explanation {
    font-family: 'Lato', sans-serif;
    font-size: 14px;
    color: #00CED1;
    line-height: 1.4;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style="
    text-align: center;
    margin-top: -60px;
    color: #00CED1;
    font-family: 'Oswald', sans-serif;
    font-size: 52px;
    font-weight: bold;
    text-shadow: 2px 2px 6px #000000;
">
📊 IBM HR Analytics Dashboard
</h1>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.columns = df.columns.str.strip()

# ---------------- TOP BAR FILTERS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    dept = st.selectbox("🏢 Department", ["All"] + sorted(df["Department"].unique()))
with col2:
    role = st.selectbox("💼 Job Role", ["All"] + sorted(df["JobRole"].unique()))
with col3:
    gender = st.selectbox("👤 Gender", ["All"] + sorted(df["Gender"].unique()))

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- APPLY FILTERS ----------------
filtered_df = df.copy()
if dept != "All":
    filtered_df = filtered_df[filtered_df["Department"] == dept]
if role != "All":
    filtered_df = filtered_df[filtered_df["JobRole"] == role]
if gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender]

# ---------------- KPI CARDS ----------------
def kpi_card(title, value, color="#00CED1"):
    st.markdown(f"""
    <div class='kpi-card' style='background: linear-gradient(135deg, {color}80, {color}40);'>
        <h3 style='font-family:Oswald,sans-serif; font-size:20px; color:white;'>{title}</h3>
        <h2 style='font-family:Oswald,sans-serif; font-size:32px; color:white;'>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

# First row
col1, col2 = st.columns(2)
with col1:
    kpi_card("Total Employees", len(filtered_df), color="#0077B6")
with col2:
    kpi_card("Attrition Count", filtered_df.Attrition.value_counts().get("Yes",0), color="#FF4C4C")

# Second row
col3, col4 = st.columns(2)
with col3:
    kpi_card("Average Age", f"{filtered_df.Age.mean():.1f}", color="#00BFA6")
with col4:
    kpi_card("Avg Years at Company", f"{filtered_df.YearsAtCompany.mean():.1f}", color="#FF8C00")

# ---------------- CHARTS (2 PER ROW) ----------------
chart_pairs = [
    ("Attrition Count", lambda df: px.histogram(
        df, x="Attrition", color="Attrition",
        color_discrete_map={"Yes":"red","No":"green"}, text_auto=True)),
    
    ("Department-wise Attrition", lambda df: px.bar(
        df.groupby(["Department","Attrition"]).size().reset_index(name="Count"),
        x="Department", y="Count", color="Attrition",
        text="Count", barmode="stack", color_discrete_map={"Yes":"red","No":"green"})),
    
    ("Age Distribution", lambda df: px.histogram(
        df, x="Age", nbins=15, color_discrete_sequence=["#00bfff"])),
    
    ("Correlation Heatmap", lambda df: px.imshow(
        df.select_dtypes(include=["int64","float64"]).corr(), text_auto=True, color_continuous_scale="RdBu_r")),
    
    ("Age vs Years at Company (Scatter)", lambda df: px.scatter(
        df, x="Age", y="YearsAtCompany", color="Attrition",
        color_discrete_map={"Yes":"red","No":"green"}, hover_data=["JobRole","Department"])),
    
    ("Monthly Income Distribution (Boxplot)", lambda df: px.box(
        df, y="MonthlyIncome", points="all", color_discrete_sequence=["#ff6347"]))
]

explanations = {
    "Attrition Count": "Shows the total number of employees who left (Yes) vs stayed (No).",
    "Department-wise Attrition": "Stacked bar chart displaying attrition counts per department.",
    "Age Distribution": "Histogram showing the distribution of employee ages.",
    "Correlation Heatmap": "Shows correlation between numeric features in the dataset.",
    "Age vs Years at Company (Scatter)": "Scatter plot of Age vs YearsAtCompany colored by Attrition.",
    "Monthly Income Distribution (Boxplot)": "Boxplot showing spread and outliers in Monthly Income."
}

for i in range(0, len(chart_pairs), 2):
    colA, colB = st.columns(2)
    for j, col in enumerate([colA, colB]):
        if i+j < len(chart_pairs):
            title, func = chart_pairs[i+j]
            col.subheader(title)
            fig = func(filtered_df)
            col.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            col.plotly_chart(fig, use_container_width=True)
            col.markdown(f"<div class='chart-explanation'>{explanations[title]}</div>", unsafe_allow_html=True)
            col.markdown("</div>", unsafe_allow_html=True)
