import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import gdown

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Neck-In Simulator",
    layout="wide"
)

# =========================
# LOAD MODEL FROM GOOGLE DRIVE
# =========================

MODEL_FILE = "neckin_cut_level_model_v5.pkl"

if not os.path.exists(MODEL_FILE):

    gdown.download(
        "https://drive.google.com/uc?id=1uxErFQwDyYjB26f690-r5pCiQEqTtP9Q",
        MODEL_FILE,
        quiet=False
    )

pipeline = joblib.load(MODEL_FILE)

# =========================
# BOND AREA TABLE
# =========================

bond_area = {
    "SOFT CD-ROD": 10.0,
    "CD ROD": 10.0,
    "OE": 18.0,
    "SB2+": 13.9,
    "SOFT CIRCLE": 25.0,
    "SOFT CIRCLES": 25.0,
    "HONEYCOMB": 16.2,
    "SOFT DOT": 12.0,
    "SOFT DOTS": 12.0
}

# =========================
# FUNCTIONS
# =========================

def expand_cut_setup(text):

    widths = []

    for item in text.split(","):

        qty, width = item.strip().split("x")

        widths.extend(
            [int(width)] * int(qty)
        )

    return widths


def simulate_recipe(
    technology,
    basisweight,
    calander,
    coating,
    width_rewinder,
    cut_setup
):

    widths = expand_cut_setup(cut_setup)

    total_cuts = len(widths)

    center = (total_cuts + 1) / 2

    rows = []

    for pos, width in enumerate(widths, start=1):

        rows.append({

            "ItemNumber": 0,

            "Technology": technology,

            "BasisWeight": basisweight,

            "Calander": calander,

            "BondArea": bond_area[calander],

            "Coating": coating,

            "Width_Rewinder": width_rewinder,

            "CutWidth": width,

            "CutWidthPct":
                width / width_rewinder,

            "Position": pos,

            "PositionPct":
                pos / total_cuts,

            "TotalCuts": total_cuts,

            "DistanceFromCenter":
                pos - center

        })

    pred_df = pd.DataFrame(rows)

    pred_df["PredictedNeckIn"] = (
        pipeline.predict(pred_df)
    )

    pred_df["NeckInPerSide"] = (
        pred_df["PredictedNeckIn"] / 2
    )

    pred_df["FinishedWidth"] = (
        pred_df["CutWidth"]
        +
        pred_df["PredictedNeckIn"]
    )

    return pred_df

# =========================
# HEADER
# =========================
def get_confidence(

    technology,
    basisweight,
    calander,
    coating,
    width_rewinder

):

    sample = pd.DataFrame([{

        "Technology":
        tech_encoder.transform(
            [technology]
        )[0],

        "Basis-Weight":
        basisweight,

        "Calander":
        cal_encoder.transform(
            [calander]
        )[0],

        "Coating":
        coat_encoder.transform(
            [coating]
        )[0],

        "Width_Rewinder":
        width_rewinder

    }])

    distance, idx = (
        nn.kneighbors(sample)
    )

    distance = distance[0][0]

    score = max(
        0,
        min(
            100,
            round(
                100 - distance,
                0
            )
        )
    )

    nearest_product = (

        historical_recipes
        .iloc[idx[0][0]]

        ["Item Number"]

    )

    return (
        score,
        nearest_product
    )

def confidence_label(score):

    if score >= 90:
        return "HIGH"

    elif score >= 70:
        return "MEDIUM"

    else:
        return "LOW"
        
st.title("📏 Neck-In Simulator")

st.caption(
    "Predict neck-in for every cut position."
)

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    technology = st.selectbox(
        "Technology",
        options=["", "SB", "SMS", "BSB"]
    )

    basisweight = st.text_input(
        "Basis Weight (gsm)",
        placeholder="Enter basis weight"
    )

    calander = st.selectbox(
        "Calander",
        options=[""] + list(bond_area.keys())
    )

with col2:

    coating = st.selectbox(
        "Coating",
        options=["", "Philic", "Phobic"]
    )

    width_rewinder = st.text_input(
        "Width Rewinder (mm)",
        placeholder="Enter rewinder width"
    )

cut_setup = st.text_input(
    "Cut Setup",
    placeholder="Example: 12x194,7x250,2x280"
)

# =========================
# PREDICT
# =========================

if st.button("Predict Neck-In"):

    try:

        if not technology:
            st.error("Please select a Technology.")
            st.stop()

        if not calander:
            st.error("Please select a Calander.")
            st.stop()

        if not coating:
            st.error("Please select a Coating.")
            st.stop()

        if not basisweight:
            st.error("Please enter Basis Weight.")
            st.stop()

        if not width_rewinder:
            st.error("Please enter Width Rewinder.")
            st.stop()

        if not cut_setup:
            st.error("Please enter a Cut Setup.")
            st.stop()

        basisweight = float(basisweight)
        width_rewinder = float(width_rewinder)

        result = simulate_recipe(
            technology,
            basisweight,
            calander,
            coating,
            width_rewinder,
            cut_setup
        )

        avg_neck = (
            result["PredictedNeckIn"]
            .mean()
        )

        trim = (
            width_rewinder
            -
            result["FinishedWidth"]
            .sum()
        )
        score, nearest_product = get_confidence(

            technology,
            basisweight,
            calander,
            coating,
            width_rewinder

        )

        confidence = confidence_label(
            score
        )

               # =====================
        # METRICS
        # =====================

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Average Neck-In",
                f"{avg_neck:.2f} mm"
            )

        with m2:

            st.metric(
                "Expected Trim",
                f"{trim:.1f} mm"
            )

        with m3:

            st.metric(
                "Confidence",
                confidence
            )

        st.info(
            f"""
Most Similar Historical Product:
{nearest_product}

Similarity Score:
{score:.0f}%
"""
        )


        # =====================
        # PROFILE CHART
        # =====================

        fig = px.line(
            result,
            x="Position",
            y="PredictedNeckIn",
            markers=True,
            title="Predicted Neck-In Profile"
        )

        fig.update_layout(
            xaxis_title="Position",
            yaxis_title="Neck-In (mm)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================
        # TABLE
        # =====================

        display_df = result[
            [
                "Position",
                "CutWidth",
                "PredictedNeckIn",
                "NeckInPerSide",
                "FinishedWidth"
            ]
        ].copy()

        display_df.columns = [
            "Position",
            "Cut Width (mm)",
            "Neck-In (mm)",
            "Per Side (mm)",
            "Finished Width (mm)"
        ]

        display_df = display_df.round(2)

        st.subheader(
            "Machine Setup Table"
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )

        # =====================
        # DOWNLOAD CSV
        # =====================

        csv = (
            display_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="Download Setup CSV",
            data=csv,
            file_name="neck_in_setup.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )
