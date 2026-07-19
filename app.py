import sqlite3
from datetime import datetime
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from ai_helper import generate_ai_report


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL APP
    ========================= */

    .stApp {
        background:
        radial-gradient(
            circle at top right,
            #dbeafe,
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #eef2f3,
            #d9e7ff
        );
    }


    /* =========================
       HEADER
    ========================= */

    .main-header {
        background:
        linear-gradient(
            135deg,
            #0f2027,
            #203a43,
            #2c5364
        );

        padding: 35px;

        border-radius: 24px;

        color: white;

        text-align: center;

        box-shadow:
        0px 12px 35px
        rgba(0, 0, 0, 0.25);

        margin-bottom: 30px;
    }

    .main-header h1 {
        font-size: 44px;
        margin: 0;
        padding: 0;
    }

    .main-header p {
        font-size: 18px;
        margin-top: 10px;
        opacity: 0.9;
    }


    /* =========================
       UPLOAD CARD
    ========================= */

    .upload-card {
        background: rgba(
            255,
            255,
            255,
            0.80
        );

        padding: 30px;

        border-radius: 22px;

        text-align: center;

        border: 2px dashed #64748b;

        margin-bottom: 20px;

        box-shadow:
        0px 8px 25px
        rgba(0, 0, 0, 0.10);
    }

    .upload-card h2 {
        color: #203a43;
        margin-bottom: 10px;
    }

    .upload-card p {
        color: #475569;
        font-size: 16px;
    }


    /* =========================
       RESULT CARD
    ========================= */

    .result-card {
        background: white;

        padding: 25px;

        border-radius: 20px;

        box-shadow:
        0px 8px 22px
        rgba(0, 0, 0, 0.12);

        margin-bottom: 18px;

        border-left: 6px solid #2c5364;
    }


    /* =========================
       FOOTER
    ========================= */

    .footer {
        text-align: center;

        color: #64748b;

        margin-top: 45px;

        padding: 20px;

        font-size: 14px;
    }


    /* =========================
       SIDEBAR
    ========================= */

    section[data-testid="stSidebar"] {

        background:
        linear-gradient(
            180deg,
            #0f2027,
            #203a43
        );
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    section[data-testid="stSidebar"] button {

        background-color:
        #2c5364 !important;

        color: white !important;

        border:
        1px solid
        #5b7c8d !important;
    }

    section[data-testid="stSidebar"] button:hover {

        background-color:
        #1f4037 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "New Inspection"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "image" not in st.session_state:
    st.session_state.image = None

if "heatmap" not in st.session_state:
    st.session_state.heatmap = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


# =========================================================
# DATABASE
# =========================================================

DB_NAME = "inspection_history.db"


def init_db():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction TEXT,

            confidence REAL,

            severity TEXT,

            risk_score INTEGER,

            repair_priority TEXT,

            public_safety_risk TEXT,

            recommended_action TEXT,

            status TEXT,

            inspection_date TEXT

        )
        """
    )

    conn.commit()

    conn.close()


init_db()


# =========================================================
# SAVE INSPECTION
# =========================================================

def save_inspection(
    prediction,
    confidence,
    ai_result
):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO inspections (

            prediction,
            confidence,
            severity,
            risk_score,
            repair_priority,
            public_safety_risk,
            recommended_action,
            status,
            inspection_date

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (

            prediction,

            confidence,

            ai_result.get(
                "severity",
                "Unknown"
            ),

            int(
                ai_result.get(
                    "risk_score",
                    0
                )
            ),

            ai_result.get(
                "repair_priority",
                "Routine"
            ),

            ai_result.get(
                "public_safety_risk",
                ""
            ),

            ai_result.get(
                "recommended_action",
                ""
            ),

            "New",

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        )
    )

    conn.commit()

    conn.close()


# =========================================================
# LOAD CNN MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    return tf.keras.models.load_model(
        "visionguard_model_v1.keras"
    )


try:

    model = load_ai_model()

except Exception as e:

    st.error(
        "❌ Model loading failed."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# LOAD CLASS NAMES
# =========================================================

try:

    with open(
        "class_names.json",
        "r"
    ) as f:

        class_indices = json.load(f)


    class_names = {

        int(value): key

        for key, value in class_indices.items()

    }

except Exception as e:

    st.error(
        "❌ class_names.json could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam_heatmap(
    img_array,
    model
):

    try:

        base_model = model.get_layer(
            "mobilenetv2_1.00_224"
        )

        last_conv_layer = base_model.get_layer(
            "Conv_1"
        )

        grad_model = tf.keras.models.Model(

            inputs=base_model.input,

            outputs=[
                last_conv_layer.output,
                base_model.output
            ]

        )

        with tf.GradientTape() as tape:

            conv_outputs, base_output = grad_model(
                img_array
            )

            x = model.get_layer(
                "global_average_pooling2d_1"
            )(base_output)

            x = model.get_layer(
                "dense_2"
            )(x)

            x = model.get_layer(
                "dropout_1"
            )(
                x,
                training=False
            )

            predictions = model.get_layer(
                "dense_3"
            )(x)

            pred_index = tf.argmax(
                predictions[0]
            )

            class_channel = predictions[
                :,
                pred_index
            ]

        grads = tape.gradient(
            class_channel,
            conv_outputs
        )

        if grads is None:

            return None

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[
            ...,
            tf.newaxis
        ]

        heatmap = tf.squeeze(
            heatmap
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        if max_value == 0:

            return None

        heatmap /= (
            max_value
            + 1e-8
        )

        return heatmap.numpy()

    except Exception as e:

        print(
            "Grad-CAM Error:",
            e
        )

        return None


# =========================================================
# PREDICTION
# =========================================================

def predict(
    image
):

    img = image.convert(
        "RGB"
    ).resize(
        (
            224,
            224
        )
    )

    arr = np.array(
        img,
        dtype=np.float32
    )

    arr = arr / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    pred = model.predict(
        arr,
        verbose=0
    )

    idx = int(
        np.argmax(
            pred[0]
        )
    )

    confidence = float(
        pred[0][idx]
    )

    probabilities = {

        class_names[i]:

        float(
            pred[0][i]
        )

        for i in range(
            len(
                pred[0]
            )
        )

    }

    heatmap = make_gradcam_heatmap(
        arr,
        model
    )

    return (

        class_names[idx],

        confidence,

        heatmap,

        probabilities

    )


# =========================================================
# HEATMAP OVERLAY
# =========================================================

def create_heatmap_overlay(
    image,
    heatmap
):

    if heatmap is None:

        return None

    heatmap_resized = cv2.resize(

        heatmap,

        (
            image.width,
            image.height
        )

    )

    heatmap_uint8 = np.uint8(

        255
        *
        heatmap_resized

    )

    heatmap_color = cv2.applyColorMap(

        heatmap_uint8,

        cv2.COLORMAP_JET

    )

    original = np.array(

        image.convert(
            "RGB"
        )

    )

    original = cv2.cvtColor(

        original,

        cv2.COLOR_RGB2BGR

    )

    overlay = cv2.addWeighted(

        original,

        0.6,

        heatmap_color,

        0.4,

        0

    )

    overlay = cv2.cvtColor(

        overlay,

        cv2.COLOR_BGR2RGB

    )

    return overlay


# =========================================================
# RESET INSPECTION
# =========================================================

def reset_inspection():

    st.session_state.prediction = None

    st.session_state.confidence = None

    st.session_state.image = None

    st.session_state.heatmap = None

    st.session_state.probabilities = None

    st.session_state.ai_result = None

    st.session_state.analysis_complete = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "🚧 VisionGuard AI"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 🧠 AI Infrastructure Intelligence"
    )

    st.write(
        "Detect infrastructure damage, "
        "understand risk, and prioritize action."
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 🛣️ Supported Classes"
    )

    st.write(
        "🕳️ Pothole Issues"
    )

    st.write(
        "🚧 Damaged Road Issues"
    )

    st.write(
        "🚸 Broken Road Sign Issues"
    )

    st.markdown(
        "---"
    )

    if st.button(
        "🏠 New Inspection",
        use_container_width=True
    ):

        st.session_state.page = "New Inspection"

        reset_inspection()

        st.rerun()


    if st.button(
        "📋 Inspection History",
        use_container_width=True
    ):

        st.session_state.page = "Inspection History"

        st.rerun()


    if st.button(
        "📊 Smart Dashboard",
        use_container_width=True
    ):

        st.session_state.page = "Smart Dashboard"

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🚧 VisionGuard AI</h1>
        <p>
            Smart Infrastructure Inspection
            &amp; Risk Intelligence Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SMART DASHBOARD
# =========================================================

if st.session_state.page == "Smart Dashboard":

    st.markdown(
        "## 📊 Infrastructure Intelligence Dashboard"
    )

    conn = sqlite3.connect(
        DB_NAME
    )

    df = pd.read_sql_query(

        """
        SELECT *
        FROM inspections
        ORDER BY id DESC
        """,

        conn

    )

    conn.close()


    if df.empty:

        st.info(
            "No inspection data available yet."
        )

    else:

        total = len(
            df
        )

        critical = len(

            df[
                df[
                    "severity"
                ]
                .astype(
                    str
                )
                .str
                .lower()
                == "critical"
            ]

        )

        high_priority = len(

            df[
                df[
                    "repair_priority"
                ]
                .astype(
                    str
                )
                .str
                .lower()
                .str
                .contains(
                    "immediate|urgent",
                    na=False
                )
            ]

        )

        avg_risk = round(

            df[
                "risk_score"
            ]
            .mean(),

            1

        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        c1.metric(
            "🔍 Total Inspections",
            total
        )

        c2.metric(
            "🔴 Critical Issues",
            critical
        )

        c3.metric(
            "⚠️ Average Risk",
            f"{avg_risk}/100"
        )

        c4.metric(
            "🚨 High Priority",
            high_priority
        )


        st.markdown(
            "---"
        )


        col1, col2 = st.columns(
            2
        )


        with col1:

            st.markdown(
                "### 🛣️ Damage Distribution"
            )

            damage_counts = (

                df[
                    "prediction"
                ]

                .value_counts()

                .reset_index()

            )

            damage_counts.columns = [

                "Damage Type",

                "Count"

            ]

            fig = px.pie(

                damage_counts,

                names="Damage Type",

                values="Count",

                hole=0.45

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )


        with col2:

            st.markdown(
                "### ⚠️ Severity Distribution"
            )

            severity_counts = (

                df[
                    "severity"
                ]

                .value_counts()

                .reset_index()

            )

            severity_counts.columns = [

                "Severity",

                "Count"

            ]

            fig = px.bar(

                severity_counts,

                x="Severity",

                y="Count",

                text="Count"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )


        st.markdown(
            "---"
        )


        st.markdown(
            "### 📈 Risk Score Timeline"
        )


        risk_df = df.copy()


        risk_df[
            "Inspection"
        ] = [

            f"Inspection #{x}"

            for x in risk_df[
                "id"
            ]

        ]


        fig = px.line(

            risk_df,

            x="Inspection",

            y="risk_score",

            markers=True

        )


        fig.update_yaxes(

            range=[
                0,
                100
            ]

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


        st.markdown(
            "---"
        )


        st.markdown(
            "### 🚨 High-Risk Inspections"
        )


        high_risk = df[

            df[
                "risk_score"
            ]

            >= 60

        ]


        if high_risk.empty:

            st.success(
                "🎉 No high-risk issues found."
            )

        else:

            st.dataframe(

                high_risk[

                    [

                        "id",

                        "prediction",

                        "severity",

                        "risk_score",

                        "repair_priority",

                        "status",

                        "inspection_date"

                    ]

                ],

                use_container_width=True,

                hide_index=True

            )


# =========================================================
# INSPECTION HISTORY
# =========================================================

elif st.session_state.page == "Inspection History":

    st.markdown(
        "## 📋 Inspection History"
    )

    conn = sqlite3.connect(
        DB_NAME
    )

    history = conn.execute(

        """
        SELECT

            id,

            prediction,

            confidence,

            severity,

            risk_score,

            repair_priority,

            status,

            inspection_date

        FROM inspections

        ORDER BY id DESC

        """

    ).fetchall()

    conn.close()


    if not history:

        st.info(
            "No inspection history available."
        )

    else:

        col1, col2 = st.columns(
            2
        )


        with col1:

            search = st.text_input(
                "🔍 Search Damage Type"
            )


        with col2:

            severity_filter = st.selectbox(

                "⚠️ Filter Severity",

                [

                    "All",

                    "Critical",

                    "High",

                    "Moderate",

                    "Low"

                ]

            )


        found = False


        for record in history:

            (

                inspection_id,

                prediction,

                confidence,

                severity,

                risk_score,

                priority,

                status,

                date

            ) = record


            if (

                search.lower()

                not in

                prediction.lower()

            ):

                continue


            if (

                severity_filter != "All"

                and severity != severity_filter

            ):

                continue


            found = True


            with st.expander(

                f"#{inspection_id} | "
                f"{prediction} | "
                f"Risk: {risk_score}/100"

            ):

                c1, c2, c3, c4 = st.columns(
                    4
                )


                c1.metric(

                    "CNN Confidence",

                    f"{confidence * 100:.1f}%"

                )


                c2.metric(

                    "Severity",

                    severity

                )


                c3.metric(

                    "Risk Score",

                    f"{risk_score}/100"

                )


                c4.metric(

                    "Status",

                    status

                )


                st.write(

                    f"**Repair Priority:** "
                    f"{priority}"

                )


                st.write(

                    f"**Inspection Date:** "
                    f"{date}"

                )


        if not found:

            st.warning(
                "No matching inspection found."
            )


# =========================================================
# NEW INSPECTION
# =========================================================

elif st.session_state.page == "New Inspection":

    st.markdown(
    """
    <div class="upload-card">
        <h2>📤 Start Infrastructure Inspection</h2>
        <p>
            Upload a road image and let
            VisionGuard AI analyze
            infrastructure damage.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


    uploaded = st.file_uploader(

        "Upload Road Image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )


    if uploaded is not None:

        image = Image.open(
            uploaded
        ).convert(
            "RGB"
        )


        st.image(

            image,

            caption="📷 Uploaded Infrastructure Image",

            use_container_width=True

        )


        if st.button(

            "🔍 ANALYZE INFRASTRUCTURE",

            use_container_width=True

        ):

            with st.spinner(

                "🔍 CNN is analyzing infrastructure damage..."

            ):

                (

                    prediction,

                    confidence,

                    heatmap,

                    probabilities

                ) = predict(

                    image

                )


            st.session_state.prediction = prediction

            st.session_state.confidence = confidence

            st.session_state.image = image

            st.session_state.heatmap = heatmap

            st.session_state.probabilities = probabilities

            st.session_state.analysis_complete = True

            st.rerun()


    # =====================================================
    # RESULTS
    # =====================================================

    if st.session_state.analysis_complete:

        prediction = st.session_state.prediction

        confidence = st.session_state.confidence

        image = st.session_state.image

        heatmap = st.session_state.heatmap

        probabilities = st.session_state.probabilities


        st.markdown(
            "---"
        )


        st.markdown(
            "## 🔍 Inspection Results"
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        c1.metric(

            "Detected Issue",

            prediction

        )


        c2.metric(

            "CNN Confidence",

            f"{confidence * 100:.2f}%"

        )


        c3.metric(

            "AI Status",

            "Ready"

        )


        c4.metric(

            "Inspection",

            "Completed"

        )


        st.markdown(
            "---"
        )


        left, right = st.columns(
            [
                1.1,
                1
            ]
        )


        with left:

            tab1, tab2 = st.tabs(

                [

                    "📷 Original Image",

                    "🔥 AI Focus Area"

                ]

            )


            with tab1:

                st.image(

                    image,

                    use_container_width=True

                )


            with tab2:

                overlay = create_heatmap_overlay(

                    image,

                    heatmap

                )


                if overlay is not None:

                    st.image(

                        overlay,

                        caption=
                        "Areas influencing CNN prediction",

                        use_container_width=True

                    )

                else:

                    st.warning(

                        "Grad-CAM visualization unavailable."

                    )


        with right:

            st.markdown(
               f"""
               <div class="result-card">

                  <h2>🧠 CNN Detection</h2>

                  <h1>{prediction}</h1>

                  <p>
                      The trained deep learning model
                      identified this infrastructure category.
                  </p>

               </div>
               """,
               unsafe_allow_html=True
)


            st.markdown(

                "### 📊 Class Probability Distribution"

            )


            for class_name, probability in (

                probabilities.items()

            ):

                st.write(

                    f"**{class_name}** — "
                    f"{probability * 100:.2f}%"

                )


                st.progress(

                    probability

                )


        st.markdown(
            "---"
        )


        st.markdown(

            "## 🤖 AI Infrastructure Assessment"

        )


        if st.session_state.ai_result is None:

            if st.button(

                "🤖 GENERATE AI INSPECTION ASSESSMENT",

                use_container_width=True

            ):

                with st.spinner(

                    "🤖 AI is evaluating infrastructure risk..."

                ):

                    ai_result = generate_ai_report(

                        image,

                        prediction,

                        confidence

                    )


                if not isinstance(

                    ai_result,

                    dict

                ):

                    st.error(

                        "AI report format is invalid."

                    )

                else:

                    st.session_state.ai_result = ai_result

                    save_inspection(

                        prediction,

                        confidence,

                        ai_result

                    )

                    st.rerun()


        else:

            ai_result = st.session_state.ai_result


            severity = ai_result.get(

                "severity",

                "Unknown"

            )


            risk_score = int(

                ai_result.get(

                    "risk_score",

                    0

                )

            )


            priority = ai_result.get(

                "repair_priority",

                "Routine"

            )


            if risk_score >= 80:

                risk_label = "🔴 Critical Risk"

            elif risk_score >= 60:

                risk_label = "🟠 High Risk"

            elif risk_score >= 35:

                risk_label = "🟡 Moderate Risk"

            else:

                risk_label = "🟢 Low Risk"


            c1, c2, c3, c4 = st.columns(
                4
            )


            c1.metric(

                "Severity",

                severity

            )


            c2.metric(

                "Risk Score",

                f"{risk_score}/100"

            )


            c3.metric(

                "Priority",

                priority

            )


            c4.metric(

                "Risk Level",

                risk_label

            )


            st.markdown(
                "---"
            )


            left, right = st.columns(
                2
            )


            with left:

                st.markdown(

                    "### ⚠️ Public Safety Risk"

                )


                st.info(

                    ai_result.get(

                        "public_safety_risk",

                        "Not available."

                    )

                )


                st.markdown(

                    "### 🛠️ Recommended Action"

                )


                st.success(

                    ai_result.get(

                        "recommended_action",

                        "Manual inspection required."

                    )

                )


            with right:

                st.markdown(

                    "### 🔍 Possible Causes"

                )


                causes = ai_result.get(

                    "possible_causes",

                    []

                )


                for cause in causes:

                    st.write(

                        f"• {cause}"

                    )


                st.markdown(

                    "### 🛡️ Preventive Measures"

                )


                measures = ai_result.get(

                    "preventive_measures",

                    []

                )


                for measure in measures:

                    st.write(

                        f"• {measure}"

                    )


            st.markdown(
                "---"
            )


            st.markdown(

                "### 👷 Inspector's Remarks"

            )


            st.write(

                ai_result.get(

                    "inspector_remarks",

                    "No remarks available."

                )

            )


            report_text = f"""

VISIONGUARD AI

AI INFRASTRUCTURE INSPECTION REPORT

====================================

CNN PREDICTION:
{prediction}

CNN CONFIDENCE:
{confidence * 100:.2f}%

SEVERITY:
{severity}

RISK SCORE:
{risk_score}/100

REPAIR PRIORITY:
{priority}

PUBLIC SAFETY RISK:
{ai_result.get("public_safety_risk", "")}

POSSIBLE CAUSES:
{chr(10).join("- " + str(x) for x in ai_result.get("possible_causes", []))}

RECOMMENDED ACTION:
{ai_result.get("recommended_action", "")}

PREVENTIVE MEASURES:
{chr(10).join("- " + str(x) for x in ai_result.get("preventive_measures", []))}

INSPECTOR'S REMARKS:
{ai_result.get("inspector_remarks", "")}

NOTE:

This report is an AI-assisted assessment and does not replace
a certified engineering inspection.

"""


            st.download_button(

                "📥 DOWNLOAD INSPECTION REPORT",

                data=report_text,

                file_name=(

                    "VisionGuard_Report_"

                    +

                    datetime.now().strftime(

                        "%Y%m%d_%H%M%S"

                    )

                    +

                    ".txt"

                ),

                mime="text/plain",

                use_container_width=True

            )


        st.markdown(
            "---"
        )


        st.markdown(

            "## 📊 Model Probability Analysis"

        )


        chart_data = pd.DataFrame(

            {

                "Class":

                list(

                    probabilities.keys()

                ),

                "Probability":

                [

                    value * 100

                    for value in probabilities.values()

                ]

            }

        )


        fig = px.bar(

            chart_data,

            x="Class",

            y="Probability",

            text_auto=".2f"

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🚧 VisionGuard AI v3.0
        <br>
        Built with ❤️ using Streamlit,
        TensorFlow, Computer Vision
        &amp; Generative AI
    </div>
    """,
    unsafe_allow_html=True
)