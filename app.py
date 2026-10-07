import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Neck-In Simulator",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

pipeline = joblib.load(
    "neckin_cut_level_model_v2_compressed.pkl"
)

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

    widths = expand_cut_setup(
        cut_setup
    )

    total_cuts = len(widths)

    center = (
        total_cuts + 1
    ) / 2

    rows = []

    for pos, width in enumerate(
        widths,
        start=1
    ):

        rows.append({

            "ItemNumber": 0,

            "Technology": technology,

            "BasisWeight": basisweight,

            "Calander": calander,

            "BondArea": bond_area[calander],

            "Coating": coating,

            "Width_Rewinder":
                width_rewinder,

            "CutWidth": width,

            "CutWidthPct":
                width / width_rewinder,

            "Position": pos,

            "PositionPct":
                pos / total_cuts,

            "TotalCuts":
                total_cuts,

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

st.title("📏 Neck-In Simulator")

st.caption(
    "Predict neck-in profile for each cut position."
)

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    technology = st.selectbox(
        "Technology",
        ["SB", "SMS", "BSB"]
    )

    basisweight = st.number_input(
        "Basis Weight (gsm)",
        value=18.0,
        step=0.5
    )

    calander = st.selectbox(
        "Calander",
        list(bond_area.keys())
    )

with col2:

    coating = st.selectbox(
        "Coating",
        [
            "Philic",
            "Phobic"
        ]
    )

    width_rewinder = st.number_input(
        "Width Rewinder (mm)",
        value=5238
    )

cut_setup = st.text_input(
    "Cut Setup",
    "10x170,9x330"
)

# =========================
# PREDICT
# =========================

if st.button("Predict Neck-In"):

    try:

        result = simulate_recipe(
            technology,
            basisweight,
            calander,
            coating,
            width_rewinder,
            cut_setup
        )

        avg_neck = (
            result[
                "PredictedNeckIn"
            ].mean()
        )

        trim = (
            width_rewinder
            -
            result[
                "FinishedWidth"
            ].sum()
        )

        # =====================
        # METRICS
        # =====================

        m1, m2 = st.columns(2)

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

        # =====================
        # CHART
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
        # RESULT TABLE
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