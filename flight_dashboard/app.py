import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
model = joblib.load("model/delay_model.pkl")


st.set_page_config(
    page_title="Flight Delay Dashboard",
    layout="wide"
)

st.title("✈️ Flight Delay Analytics Dashboard")
st.markdown(
    "This dashboard displays flight delay analysis and delay predictions"
    "based on historical flight data."
)
st.divider()


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("D:\\project\\flight_dashboard\\data\\flights_sample_10000.csv")

df = load_data()

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("🔍 Filter Data")

airline = st.sidebar.multiselect(
    "Airline",
    options=df["AIRLINE_CODE"].unique(),
    default=df["AIRLINE_CODE"].unique()
)

origin = st.sidebar.multiselect(
    "Origin",
    options=df["ORIGIN"].unique(),
    default=df["ORIGIN"].unique()
)

dest = st.sidebar.multiselect(
    "Destination",
    options=df["DEST"].unique(),
    default=df["DEST"].unique()
)

day = st.sidebar.multiselect(
    "Day",
    options=sorted(df["DAY"].unique()),
    default=sorted(df["DAY"].unique())
)

filtered_df = df[
    (df["AIRLINE_CODE"].isin(airline)) &
    (df["ORIGIN"].isin(origin)) &
    (df["DEST"].isin(dest)) &
    (df["DAY"].isin(day))
]



# ---------------- OVERVIEW ----------------
st.subheader("📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Flights", len(filtered_df))
col2.metric("Delayed Flights", filtered_df["DELAYED"].sum())
col3.metric(
    "Delay Percentage",
    f"{(filtered_df['DELAYED'].mean()*100):.2f}%"
)


# ---------------- VISUALIZATION ----------------
st.subheader("📈 Delay Analysis")

fig, ax = plt.subplots()
filtered_df.groupby("DEP_HOUR")["DELAYED"].mean().plot(ax=ax)
ax.set_ylabel("Delay Rate")
ax.set_xlabel("Departure Hour")
st.pyplot(fig)

st.subheader("📊 Delay Rate by Airline")

fig, ax = plt.subplots()
filtered_df.groupby("AIRLINE_CODE")["DELAYED"].mean().sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Delay Rate")
st.pyplot(fig)

st.subheader("📈 Delay Rate by Departure Hour")

fig, ax = plt.subplots()
filtered_df.groupby("DEP_HOUR")["DELAYED"].mean().plot(ax=ax)
ax.set_xlabel("Departure Hour")
ax.set_ylabel("Delay Rate")
st.pyplot(fig)

# ---------------- PREDICTION ----------------
st.divider()
st.header("🤖 Flight Delay Prediction")

with st.form("prediction_form"):
    airline_input = st.selectbox("Airline", df["AIRLINE_CODE"].unique())
    origin_input = st.selectbox("Origin", df["ORIGIN"].unique())
    dest_input = st.selectbox("Destination", df["DEST"].unique())
    day_input = st.selectbox("Day", sorted(df["DAY"].unique()))
    hour_input = st.slider("Departure Hour", 0, 23, 12)
    distance_input = st.number_input("Distance (miles)", min_value=0)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "AIRLINE_CODE": airline_input,
        "ORIGIN": origin_input,
        "DEST": dest_input,
        "DAY": day_input,
        "DEP_HOUR": hour_input,
        "DISTANCE": distance_input
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Flight is likely to be DELAYED")
    else:
        st.success("✅ Flight is likely to be ON TIME")
