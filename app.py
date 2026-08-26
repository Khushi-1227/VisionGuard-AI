import sqlite3
import os
from datetime import datetime, date
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from ai_helper import generate_ai_report, client as groq_client
from streamlit_js_eval import streamlit_js_eval

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

import io


# =========================================================
# MULTI-LANGUAGE SUPPORT
# =========================================================

TRANSLATIONS = {

    "new_inspection_title": {
        "English": "📤 Start Infrastructure Inspection",
        "Hindi": "📤 इंस्पेक्शन शुरू करें",
        "Gujarati": "📤 ઇન્સ્પેક્શન શરૂ કરો"
    },
    "new_inspection_caption": {
        "English": "Upload a road image and let VisionGuard "
                   "AI analyze infrastructure damage.",
        "Hindi": "सड़क की फोटो अपलोड करें, VisionGuard AI "
                 "नुकसान का विश्लेषण करेगा।",
        "Gujarati": "રોડનો ફોટો અપલોડ કરો, VisionGuard AI "
                    "નુકસાનનું વિશ્લેષણ કરશે."
    },
    "track_complaint_title": {
        "English": "🔍 Track My Complaint",
        "Hindi": "🔍 अपनी शिकायत ट्रैक करें",
        "Gujarati": "🔍 તમારી ફરિયાદ ટ્રેક કરો"
    },
    "track_complaint_caption": {
        "English": "Enter your Complaint ID to check the "
                   "current status.",
        "Hindi": "स्थिति देखने के लिए अपना कंप्लेंट आईडी डालें।",
        "Gujarati": "સ્થિતિ જોવા માટે તમારો ફરિયાદ ID દાખલ કરો."
    },
    "officer_dashboard_title": {
        "English": "🏛️ Municipality Officer — "
                   "Complaints Dashboard",
        "Hindi": "🏛️ नगर निगम अधिकारी — शिकायत डैशबोर्ड",
        "Gujarati": "🏛️ મ્યુનિસિપાલિટી ઓફિસર — ફરિયાદ ડેશબોર્ડ"
    },
    "inspector_dashboard_title": {
        "English": "🛠️ Inspector — My Assigned Work",
        "Hindi": "🛠️ इंस्पेक्टर — मेरा सौंपा गया कार्य",
        "Gujarati": "🛠️ ઇન્સ્પેક્ટર — મને સોંપાયેલ કામ"
    },
    "smart_dashboard_title": {
        "English": "📊 Infrastructure Intelligence Dashboard",
        "Hindi": "📊 इंफ्रास्ट्रक्चर इंटेलिजेंस डैशबोर्ड",
        "Gujarati": "📊 ઇન્ફ્રાસ્ટ્રક્ચર ઇન્ટેલિજન્સ ડેશબોર્ડ"
    },
    "ai_assistant_title": {
        "English": "💬 VisionGuard AI Assistant",
        "Hindi": "💬 VisionGuard AI सहायक",
        "Gujarati": "💬 VisionGuard AI સહાયક"
    },
    "upload_button": {
        "English": "📷 Upload Road Image",
        "Hindi": "📷 सड़क की फोटो अपलोड करें",
        "Gujarati": "📷 રોડનો ફોટો અપલોડ કરો"
    },
    "check_status_button": {
        "English": "🔍 Check Status",
        "Hindi": "🔍 स्थिति देखें",
        "Gujarati": "🔍 સ્થિતિ તપાસો"
    },
    "menu_new_inspection": {
        "English": "🏠 New Inspection",
        "Hindi": "🏠 नई शिकायत",
        "Gujarati": "🏠 નવી ફરિયાદ"
    },
    "menu_track_complaint": {
        "English": "🔍 Track My Complaint",
        "Hindi": "🔍 शिकायत ट्रैक करें",
        "Gujarati": "🔍 ફરિયાદ ટ્રેક કરો"
    },
    "menu_complaints_dashboard": {
        "English": "🏛️ Complaints Dashboard",
        "Hindi": "🏛️ शिकायत डैशबोर्ड",
        "Gujarati": "🏛️ ફરિયાદ ડેશબોર્ડ"
    },
    "menu_smart_dashboard": {
        "English": "📊 Smart Dashboard",
        "Hindi": "📊 स्मार्ट डैशबोर्ड",
        "Gujarati": "📊 સ્માર્ટ ડેશબોર્ડ"
    },
    "menu_assigned_work": {
        "English": "🛠️ My Assigned Work",
        "Hindi": "🛠️ मुझे सौंपा गया कार्य",
        "Gujarati": "🛠️ મને સોંપાયેલ કામ"
    },
    "menu_admin_dashboard": {
        "English": "👑 Super Admin Dashboard",
        "Hindi": "👑 सुपर एडमिन डैशबोर्ड",
        "Gujarati": "👑 સુપર એડમિન ડેશબોર્ડ"
    },
    "menu_public_map": {
        "English": "🗺️ Public Transparency Map",
        "Hindi": "🗺️ सार्वजनिक पारदर्शिता मानचित्र",
        "Gujarati": "🗺️ જાહેર પારદર્શિતા નકશો"
    },
    "menu_ai_assistant": {
        "English": "💬 AI Assistant",
        "Hindi": "💬 AI सहायक",
        "Gujarati": "💬 AI સહાયક"
    },
    "upload_image_label": {
        "English": "📷 Upload Road Image",
        "Hindi": "📷 सड़क की फोटो अपलोड करें",
        "Gujarati": "📷 રોડનો ફોટો અપલોડ કરો"
    },
    "voice_section_title": {
        "English": "🎙️ Voice Description (optional)",
        "Hindi": "🎙️ आवाज़ में विवरण (वैकल्पिक)",
        "Gujarati": "🎙️ અવાજ વર્ણન (વૈકલ્પિક)"
    },
    "voice_caption": {
        "English": "Tap the mic and speak to describe the "
                   "issue — it will be transcribed "
                   "automatically.",
        "Hindi": "माइक पर टैप करें और समस्या के बारे में बोलें — "
                 "यह अपने आप टेक्स्ट में बदल जाएगा।",
        "Gujarati": "માઇક પર ટેપ કરો અને સમસ્યા વિશે બોલો — "
                    "તે આપમેળે ટેક્સ્ટમાં રૂપાંતરિત થશે."
    },
    "voice_upload_label": {
        "English": "🎙️ Tap to Record",
        "Hindi": "🎙️ रिकॉर्ड करने के लिए टैप करें",
        "Gujarati": "🎙️ રેકોર્ડ કરવા ટેપ કરો"
    },
    "complaint_id_label": {
        "English": "🧾 Complaint ID",
        "Hindi": "🧾 शिकायत आईडी",
        "Gujarati": "🧾 ફરિયાદ ID"
    },
    "status_label": {
        "English": "Status",
        "Hindi": "स्थिति",
        "Gujarati": "સ્થિતિ"
    },
    "severity_label": {
        "English": "Severity",
        "Hindi": "गंभीरता",
        "Gujarati": "ગંભીરતા"
    },
    "risk_score_label": {
        "English": "Risk Score",
        "Hindi": "जोखिम स्कोर",
        "Gujarati": "જોખમ સ્કોર"
    },

}


def t(key):

    return TRANSLATIONS.get(
        key,
        {}
    ).get(
        st.session_state.language,
        key
    )




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

    /* ============ FORCE CONSISTENT LIGHT THEME ============ */
    /* Prevents invisible text when browser/device is in dark mode */

    html, body, [class*="css"] {
        color-scheme: light !important;
    }

    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp li, .stApp div {
        color: #0f2027;
    }

    .stApp {
        background-color: #eef2f3 !important;
    }

    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
    }

    div[data-testid="stExpander"] * {
        color: #0f2027 !important;
    }

    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        color: #0f2027 !important;
    }

    div[data-testid="stMetric"] * {
        color: #0f2027 !important;
    }

    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        color: #0f2027 !important;
        border-radius: 14px;
    }

    div[data-testid="stChatMessage"] * {
        color: #0f2027 !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #0f2027 !important;
    }

    div[data-testid="stSelectbox"] * {
        color: #0f2027;
    }

    /* ========================================================= */

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

    .result-card {
        background: white;
        color: #1e293b;
        padding: 25px;
        border-radius: 20px;

        box-shadow:
        0px 8px 22px
        rgba(0, 0, 0, 0.12);

        margin-bottom: 18px;
        border-left: 6px solid #2c5364;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 45px;
        padding: 20px;
        font-size: 14px;
    }

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

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0f2027 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="popover"] li {
        color: #0f2027 !important;
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

    /* ================= HIGH-LEVEL VISUAL UPGRADE ================= */

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main-header {
        animation: fadeInUp 0.6s ease-out;
        background-size: 200% 200%;
        animation: fadeInUp 0.6s ease-out,
                   gradientShift 8s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .upload-card, .result-card {
        transition: transform 0.25s ease,
                    box-shadow 0.25s ease;
        animation: fadeInUp 0.5s ease-out;
    }

    .upload-card:hover, .result-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow:
        0px 16px 40px
        rgba(44, 83, 100, 0.25);
    }

    div[data-testid="stExpander"] {
        border-radius: 18px !important;
        transition: transform 0.2s ease,
                    box-shadow 0.2s ease;
        box-shadow:
        0px 6px 18px
        rgba(0, 0, 0, 0.08);
    }

    div[data-testid="stExpander"]:hover {
        transform: translateY(-3px);
        box-shadow:
        0px 12px 28px
        rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #ffffff,
            #eef2f7
        );
        border-radius: 16px;
        padding: 12px;
        box-shadow:
        0px 6px 16px
        rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.03);
    }

    .stButton > button {
        border-radius: 14px !important;
        background: linear-gradient(
            135deg,
            #2c5364,
            #0f2027
        ) !important;
        color: white !important;
        border: none !important;
        transition: transform 0.2s ease,
                    box-shadow 0.2s ease;
        box-shadow:
        0px 6px 16px
        rgba(15, 32, 39, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow:
        0px 10px 24px
        rgba(15, 32, 39, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0px) scale(0.98);
    }

    .st-key-mic_row {
        margin-bottom: -12px;
    }

    .st-key-mic_row div[data-testid="stPopover"] button {
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        padding: 0 !important;
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

if "role" not in st.session_state:
    st.session_state.role = "Citizen"

if "language" not in st.session_state:
    st.session_state.language = "English"

if "voice_note_text" not in st.session_state:
    st.session_state.voice_note_text = ""

if "inspector_login_name" not in st.session_state:
    st.session_state.inspector_login_name = ""

if "officer_municipality" not in st.session_state:
    st.session_state.officer_municipality = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

if "location" not in st.session_state:
    st.session_state.location = ""

if "latitude" not in st.session_state:
    st.session_state.latitude = 0.0

if "longitude" not in st.session_state:
    st.session_state.longitude = 0.0

if "gps_detected" not in st.session_state:
    st.session_state.gps_detected = False

if "address" not in st.session_state:
    st.session_state.address = ""

if "assigned_municipality" not in st.session_state:
    st.session_state.assigned_municipality = ""

if "target_repair_date" not in st.session_state:
    st.session_state.target_repair_date = date.today()


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

            inspection_id TEXT UNIQUE,

            prediction TEXT,

            confidence REAL,

            severity TEXT,

            risk_score INTEGER,

            repair_priority TEXT,

            public_safety_risk TEXT,

            recommended_action TEXT,

            status TEXT,

            assigned_municipality TEXT,

            assigned_inspector TEXT,

            target_repair_date TEXT,

            inspection_date TEXT,

            location TEXT,

            latitude REAL,

            longitude REAL

        )
        """
    )

    existing_columns = [

        row[1]

        for row in cursor.execute(
            "PRAGMA table_info(inspections)"
        ).fetchall()

    ]

    if "inspection_id" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN inspection_id TEXT
            """
        )

    if "location" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN location TEXT
            """
        )

    if "latitude" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN latitude REAL
            """
        )

    if "longitude" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN longitude REAL
            """
        )

    if "assigned_municipality" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN assigned_municipality TEXT
            """
        )

    if "assigned_inspector" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN assigned_inspector TEXT
            """
        )

    if "target_repair_date" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN target_repair_date TEXT
            """
        )
    if "repair_start_date" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN repair_start_date TEXT
            """
        )

    if "repair_completion_date" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN repair_completion_date TEXT
            """
        )

    if "estimated_repair_cost" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN estimated_repair_cost REAL
            """
        )

    if "repair_notes" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN repair_notes TEXT
            """
        )

    if "repair_photo_path" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN repair_photo_path TEXT
            """
        )

    if "citizen_voice_note" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE inspections
            ADD COLUMN citizen_voice_note TEXT
            """
        )

    conn.commit()

    conn.close()


init_db()


# =========================================================
# REVERSE GEOCODING
# =========================================================

@st.cache_data(ttl=3600)
def get_address_from_coordinates(
    latitude,
    longitude
):

    try:

        geolocator = Nominatim(
            user_agent="VisionGuardAI"
        )

        location = geolocator.reverse(

            (
                latitude,
                longitude
            ),

            language="en",

            exactly_one=True,

            timeout=10

        )

        if location:

            return location.address

        return "Address not found"

    except (
        GeocoderTimedOut,
        GeocoderServiceError
    ):

        return "Unable to detect address"

    except Exception:

        return "Address detection failed"


# =========================================================
# GENERATE UNIQUE INSPECTION ID
# =========================================================

def generate_inspection_id():

    date_part = datetime.now().strftime(
        "%Y%m%d"
    )

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM inspections
        """
    )

    count = cursor.fetchone()[0] + 1

    conn.close()

    return (
        f"VG-{date_part}-{count:04d}"
    )


# =========================================================
# DIGITAL COMPLAINT RECEIPT (WITH QR CODE)
# =========================================================

def generate_complaint_receipt(

    inspection_id,
    prediction,
    severity,
    risk_score,
    location,
    assigned_municipality,
    target_repair_date

):

    width, height = 700, 500

    receipt = Image.new(
        "RGB",
        (width, height),
        color="white"
    )

    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(receipt)

    try:
        title_font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf", 26
        )
        text_font = ImageFont.truetype(
            "DejaVuSans.ttf", 18
        )
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    draw.rectangle(
        [(0, 0), (width, 70)],
        fill=(15, 32, 39)
    )

    draw.text(
        (20, 20),
        "VisionGuard AI - Complaint Receipt",
        fill="white",
        font=title_font
    )

    lines = [
        f"Complaint ID: {inspection_id}",
        f"Detected Issue: {prediction}",
        f"Severity: {severity}",
        f"Risk Score: {risk_score}/100",
        f"Location: {location}",
        f"Assigned Municipality: {assigned_municipality}",
        f"Target Repair Date: {target_repair_date or 'Pending'}",
        "Status: New",
    ]

    y = 100

    for line in lines:

        draw.text(
            (20, y),
            line,
            fill=(15, 32, 39),
            font=text_font
        )

        y += 34

    if QR_AVAILABLE:

        qr_img = qrcode.make(
            f"VisionGuard Complaint: {inspection_id} | "
            f"Status: New | Location: {location}"
        )

        qr_img = qr_img.resize((160, 160))

        receipt.paste(
            qr_img,
            (width - 180, height - 180)
        )

    else:

        draw.text(
            (width - 260, height - 60),
            "(Install 'qrcode' package for QR code)",
            fill=(150, 0, 0),
            font=text_font
        )

    buffer = io.BytesIO()

    receipt.save(buffer, format="PNG")

    return buffer.getvalue()


# =========================================================
# SLA / OVERDUE CHECK
# =========================================================

def is_overdue(target_repair_date, status):

    if status == "Resolved":
        return False

    if not target_repair_date:
        return False

    try:

        target = datetime.strptime(
            str(target_repair_date)[:10],
            "%Y-%m-%d"
        ).date()

        return target < date.today()

    except (ValueError, TypeError):

        return False


# =========================================================
# AUTO-ESCALATION
# =========================================================

def is_escalated(target_repair_date, status, threshold_days=5):

    if status == "Resolved":
        return False

    if not target_repair_date:
        return False

    try:

        target = datetime.strptime(
            str(target_repair_date)[:10],
            "%Y-%m-%d"
        ).date()

        days_overdue = (
            date.today() - target
        ).days

        return days_overdue >= threshold_days

    except (ValueError, TypeError):

        return False


# =========================================================
# DUPLICATE COMPLAINT DETECTION
# =========================================================

def find_duplicate_complaints(
    prediction,
    latitude,
    longitude,
    radius_deg=0.002
):

    try:

        latitude = float(latitude)
        longitude = float(longitude)

    except (TypeError, ValueError):

        return []

    if latitude == 0.0 and longitude == 0.0:
        return []

    conn = sqlite3.connect(
        DB_NAME
    )

    rows = conn.execute(

        """

        SELECT inspection_id, status, inspection_date

        FROM inspections

        WHERE prediction = ?

        AND status != 'Resolved'

        AND ABS(latitude - ?) <= ?

        AND ABS(longitude - ?) <= ?

        """,

        (

            prediction,

            latitude,

            radius_deg,

            longitude,

            radius_deg

        )

    ).fetchall()

    conn.close()

    return rows


# =========================================================
# AUTO-ASSIGN MUNICIPALITY DEPARTMENT
# =========================================================

@st.cache_data(ttl=3600)
def get_municipality_department(latitude, longitude):

    try:

        geolocator = Nominatim(
            user_agent="VisionGuardAI"
        )

        location = geolocator.reverse(

            (
                latitude,
                longitude
            ),

            language="en",

            exactly_one=True,

            timeout=10

        )

        if not location:
            return "General Municipal Corporation"

        address = location.raw.get(
            "address",
            {}
        )

        area_name = (

            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("village")
            or address.get("county")
            or address.get("state_district")

        )

        if area_name:
            return f"{area_name} Municipal Corporation"

        return "General Municipal Corporation"

    except (
        GeocoderTimedOut,
        GeocoderServiceError
    ):

        return "General Municipal Corporation"


# =========================================================
# SAVE INSPECTION
# =========================================================

def save_inspection(
    prediction,
    confidence,
    ai_result,
    location,
    latitude,
    longitude,
    assigned_municipality,
    target_repair_date,
    citizen_voice_note=""
):

    inspection_id = generate_inspection_id()

    try:

        latitude = float(
            latitude
        )

    except:

        latitude = 0.0

    try:

        longitude = float(
            longitude
        )

    except:

        longitude = 0.0

    if not location:

        location = "GPS Location"

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO inspections (

            inspection_id,

            prediction,

            confidence,

            severity,

            risk_score,

            repair_priority,

            public_safety_risk,

            recommended_action,

            status,

            assigned_municipality,

            target_repair_date,

            inspection_date,

            location,

            latitude,

            longitude,

            citizen_voice_note

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """,

        (

            inspection_id,

            prediction,

            float(
                confidence
            ),

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

            assigned_municipality,

            target_repair_date,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            location,

            latitude,

            longitude,

            citizen_voice_note

        )

    )

    conn.commit()

    conn.close()

    return inspection_id


# =========================================================
# UPDATE INSPECTION STATUS
# =========================================================

def update_inspection_status(
    inspection_id,
    new_status
):

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE inspections

        SET status = ?

        WHERE id = ?

        """,

        (

            new_status,

            inspection_id

        )

    )

    conn.commit()

    conn.close()


# =========================================================
# ASSIGN INSPECTOR (MUNICIPALITY OFFICER ACTION)
# =========================================================

def assign_inspector(
    inspection_id,
    inspector_name
):

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE inspections

        SET assigned_inspector = ?

        WHERE id = ?

        """,

        (

            inspector_name,

            inspection_id

        )

    )

    conn.commit()

    conn.close()


def update_repair_details(
    inspection_id,
    repair_start_date,
    repair_completion_date,
    estimated_repair_cost,
    repair_notes,
    repair_photo_path=None
):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    if repair_photo_path:

        cursor.execute(
            """
            UPDATE inspections

            SET
                repair_start_date = ?,
                repair_completion_date = ?,
                estimated_repair_cost = ?,
                repair_notes = ?,
                repair_photo_path = ?

            WHERE id = ?

            """,

            (

                repair_start_date,

                repair_completion_date,

                estimated_repair_cost,

                repair_notes,

                repair_photo_path,

                inspection_id

            )

        )

    else:

        cursor.execute(
            """
            UPDATE inspections

            SET
                repair_start_date = ?,
                repair_completion_date = ?,
                estimated_repair_cost = ?,
                repair_notes = ?

            WHERE id = ?

            """,

            (

                repair_start_date,

                repair_completion_date,

                estimated_repair_cost,

                repair_notes,

                inspection_id

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

        class_indices = json.load(
            f
        )

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
            )(
                base_output
            )

            x = model.get_layer(
                "dense_2"
            )(
                x
            )

            x = model.get_layer(
                "dropout_1"
            )(
                x,
                training=False
            )

            predictions = model.get_layer(
                "dense_3"
            )(
                x
            )

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
            axis=(
                0,
                1,
                2
            )
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
        ).numpy()

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

    st.session_state.location = ""

    st.session_state.latitude = 0.0

    st.session_state.longitude = 0.0

    st.session_state.gps_detected = False

    st.session_state.address = ""

    st.session_state.assigned_municipality = ""

    st.session_state.target_repair_date = date.today()

    st.session_state.voice_note_text = ""


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

    st.markdown(
        "---"
    )

    st.markdown(
        "### 🌐 Language"
    )

    st.session_state.language = st.selectbox(

        "Language",

        [
            "English",
            "Hindi",
            "Gujarati"
        ],

        index=[
            "English",
            "Hindi",
            "Gujarati"
        ].index(
            st.session_state.language
        ),

        label_visibility="collapsed"

    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 🔑 Login As"
    )

    selected_role = st.selectbox(

        "Choose your role",

        [
            "👤 Citizen",
            "🏛️ Municipality Officer",
            "🛠️ Inspector",
            "👑 Super Admin"
        ],

        index=(
            [
                "👤 Citizen",
                "🏛️ Municipality Officer",
                "🛠️ Inspector",
                "👑 Super Admin"
            ].index(
                st.session_state.role
            )
            if st.session_state.role in [
                "👤 Citizen",
                "🏛️ Municipality Officer",
                "🛠️ Inspector",
                "👑 Super Admin"
            ]
            else 0
        ),

        label_visibility="collapsed"

    )

    if selected_role != st.session_state.role:

        st.session_state.role = selected_role

        if selected_role == "👤 Citizen":
            st.session_state.page = "New Inspection"

        elif selected_role == "🏛️ Municipality Officer":
            st.session_state.page = "Complaints Dashboard"

        elif selected_role == "🛠️ Inspector":
            st.session_state.page = "My Assigned Work"

        elif selected_role == "👑 Super Admin":
            st.session_state.page = "Super Admin Dashboard"

        st.rerun()

    st.markdown(
        "---"
    )

    # =====================================================
    # CITIZEN MENU
    # =====================================================

    if st.session_state.role == "👤 Citizen":

        if st.button(

            t('menu_new_inspection'),

            use_container_width=True

        ):

            st.session_state.page = (
                "New Inspection"
            )

            reset_inspection()

            st.rerun()

        if st.button(

            t('menu_track_complaint'),

            use_container_width=True

        ):

            st.session_state.page = (
                "Track My Complaint"
            )

            st.rerun()

    # =====================================================
    # MUNICIPALITY OFFICER MENU
    # =====================================================

    elif st.session_state.role == "🏛️ Municipality Officer":

        if st.button(

            t('menu_complaints_dashboard'),

            use_container_width=True

        ):

            st.session_state.page = (
                "Complaints Dashboard"
            )

            st.rerun()

        if st.button(

            t('menu_smart_dashboard'),

            use_container_width=True

        ):

            st.session_state.page = (
                "Smart Dashboard"
            )

            st.rerun()

    # =====================================================
    # INSPECTOR MENU
    # =====================================================

    elif st.session_state.role == "🛠️ Inspector":

        if st.button(

            t('menu_assigned_work'),

            use_container_width=True

        ):

            st.session_state.page = (
                "My Assigned Work"
            )

            st.rerun()

    # =====================================================
    # SUPER ADMIN MENU
    # =====================================================

    elif st.session_state.role == "👑 Super Admin":

        if st.button(

            t('menu_admin_dashboard'),

            use_container_width=True

        ):

            st.session_state.page = (
                "Super Admin Dashboard"
            )

            st.rerun()

    st.markdown(
        "---"
    )

    if st.button(

        t('menu_public_map'),

        use_container_width=True

    ):

        st.session_state.page = (
            "Public Map"
        )

        st.rerun()

    if st.button(

        t('menu_ai_assistant'),

        use_container_width=True

    ):

        st.session_state.page = (
            "AI Assistant"
        )

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

    if st.session_state.role != "🏛️ Municipality Officer":

        st.warning(
            "🚫 This section is only for Municipality Officers."
        )

        st.stop()

    st.markdown(
        f"## {t('smart_dashboard_title')}"
    )

    st.session_state.officer_municipality = st.text_input(

        "🏛️ Your Municipality / Area "
        "(leave blank for city-wide view)",

        value=st.session_state.officer_municipality,

        placeholder="Example: Ahmedabad"

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

    if st.session_state.officer_municipality.strip():

        df = df[

            df["assigned_municipality"]

            .astype(str)

            .str.lower()

            .str.contains(

                st.session_state.officer_municipality.strip().lower(),

                na=False

            )

        ]

        st.caption(
            f"📍 Showing analytics for complaints matching "
            f"\"{st.session_state.officer_municipality.strip()}\""
        )

    else:

        st.caption(
            "🌐 Showing city-wide analytics "
            "(all municipalities)"
        )

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

                df["severity"]

                .astype(str)

                .str.lower()

                == "critical"

            ]

        )

        high_priority = len(

            df[

                df["repair_priority"]

                .astype(str)

                .str.lower()

                .str.contains(

                    "immediate|urgent",

                    na=False

                )

            ]

        )

        avg_risk = round(

            df["risk_score"].mean(),

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

        new_issues = len(

            df[

                df["status"]

                == "New"

            ]

        )

        under_inspection = len(

            df[

                df["status"]

                == "Under Inspection"

            ]

        )

        repair_in_progress = len(

            df[

                df["status"]

                == "Repair In Progress"

            ]

        )

        resolved = len(

            df[

                df["status"]

                == "Resolved"

            ]

        )

        st.markdown(
            "### 🎯 Issue Resolution Overview"
        )

        s1, s2, s3, s4 = st.columns(
            4
        )

        s1.metric(
            "🆕 New Issues",
            new_issues
        )

        s2.metric(
            "🔍 Under Inspection",
            under_inspection
        )

        s3.metric(
            "🛠️ Repair In Progress",
            repair_in_progress
        )

        s4.metric(
            "✅ Resolved",
            resolved
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

                df["prediction"]

                .value_counts()

                .reset_index()

            )

            damage_counts.columns = [

                "Damage Type",

                "Count"

            ]

            fig_damage = px.pie(

                damage_counts,

                names="Damage Type",

                values="Count",

                hole=0.45

            )

            st.plotly_chart(

                fig_damage,

                use_container_width=True,

                key="damage_distribution_chart"

            )

        with col2:

            st.markdown(
                "### ⚠️ Severity Distribution"
            )

            severity_counts = (

                df["severity"]

                .value_counts()

                .reset_index()

            )

            severity_counts.columns = [

                "Severity",

                "Count"

            ]

            fig_severity = px.bar(

                severity_counts,

                x="Severity",

                y="Count",

                text="Count"

            )

            st.plotly_chart(

                fig_severity,

                use_container_width=True,

                key="severity_distribution_chart"

            )

        st.markdown(
            "---"
        )

        st.markdown(
            "### 🎯 Issue Resolution Status"
        )

        status_counts = (

            df["status"]

            .value_counts()

            .reset_index()

        )

        status_counts.columns = [

            "Status",

            "Count"

        ]

        fig_status = px.bar(

            status_counts,

            x="Status",

            y="Count",

            text="Count",

            title="Infrastructure Issue Progress"

        )

        st.plotly_chart(

            fig_status,

            use_container_width=True,

            key="status_distribution_chart"

        )

        st.markdown(
            "---"
        )

        st.markdown(
            "### 📈 Risk Score Timeline"
        )

        risk_df = df.copy()

        risk_df["Inspection"] = [

            f"Inspection #{x}"

            for x in risk_df["id"]

        ]

        fig_risk = px.line(

            risk_df,

            x="Inspection",

            y="risk_score",

            markers=True

        )

        fig_risk.update_yaxes(

            range=[

                0,

                100

            ]

        )

        st.plotly_chart(

            fig_risk,

            use_container_width=True,

            key="risk_timeline_chart"

        )

        st.markdown(
            "---"
        )

        st.markdown(
            "### 🚨 High-Risk Inspections"
        )

        high_risk = df[

            df["risk_score"] >= 60

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

                        "inspection_id",

                        "prediction",

                        "severity",

                        "risk_score",

                        "repair_priority",

                        "assigned_municipality",

                        "assigned_inspector",

                        "target_repair_date",

                        "status",

                        "inspection_date"

                    ]

                ],

                use_container_width=True,

                hide_index=True

            )

        st.markdown(
            "---"
        )

        st.markdown(
            "### 🗺️ Infrastructure Risk Map"
        )

        st.caption(

            "Visualize inspected infrastructure issues "

            "based on their geographical location."

        )

        map_columns = [

            "id",

            "prediction",

            "severity",

            "risk_score",

            "status",

            "location",

            "latitude",

            "longitude"

        ]

        missing_columns = [

            column

            for column in map_columns

            if column not in df.columns

        ]

        if missing_columns:

            st.error(

                f"Missing database columns: "

                f"{missing_columns}"

            )

        else:

            map_df = df[

                map_columns

            ].copy()

            map_df["latitude"] = pd.to_numeric(

                map_df["latitude"],

                errors="coerce"

            )

            map_df["longitude"] = pd.to_numeric(

                map_df["longitude"],

                errors="coerce"

            )

            map_df = map_df.dropna(

                subset=[

                    "latitude",

                    "longitude"

                ]

            )

            map_df = map_df[

                (

                    map_df["latitude"]

                    >= -90

                )

                &

                (

                    map_df["latitude"]

                    <= 90

                )

                &

                (

                    map_df["longitude"]

                    >= -180

                )

                &

                (

                    map_df["longitude"]

                    <= 180

                )

                &

                (

                    map_df["latitude"]

                    != 0

                )

                &

                (

                    map_df["longitude"]

                    != 0

                )

            ]

            if map_df.empty:

                st.info(

                    "📍 No location-based inspection "

                    "data available yet."

                )

                st.caption(

                    "GPS coordinates will automatically "

                    "appear here after inspection."

                )

            else:

                fig_map = px.scatter_mapbox(

                    map_df,

                    lat="latitude",

                    lon="longitude",

                    hover_name="prediction",

                    hover_data={

                        "id": True,

                        "severity": True,

                        "risk_score": True,

                        "status": True,

                        "location": True,

                        "latitude": False,

                        "longitude": False

                    },

                    color="severity",

                    size="risk_score",

                    zoom=10,

                    height=600,

                    mapbox_style="open-street-map",

                    title=(

                        "📍 Infrastructure "

                        "Inspection Locations"

                    )

                )

                st.plotly_chart(

                    fig_map,

                    use_container_width=True,

                    key="infrastructure_risk_map"

                )


# =========================================================
# INSPECTION HISTORY
# =========================================================

elif st.session_state.page in (
    "Complaints Dashboard",
    "My Assigned Work"
):

    if st.session_state.role not in (
        "🏛️ Municipality Officer",
        "🛠️ Inspector"
    ):

        st.warning(
            "🚫 This section is only for Municipality "
            "Officers and Inspectors."
        )

        st.stop()

    is_officer = (
        st.session_state.role == "🏛️ Municipality Officer"
    )

    is_inspector = (
        st.session_state.role == "🛠️ Inspector"
    )

    if is_officer:

        st.markdown(
            f"## {t('officer_dashboard_title')}"
        )

        st.caption(
            "Enter your municipality/area name to view "
            "complaints from your jurisdiction only."
        )

        st.session_state.officer_municipality = st.text_input(

            "🏛️ Your Municipality / Area",

            value=st.session_state.officer_municipality,

            placeholder="Example: Ahmedabad"

        )

    else:

        st.markdown(
            f"## {t('inspector_dashboard_title')}"
        )

        st.caption(
            "Enter your name to view complaints "
            "assigned to you."
        )

        st.session_state.inspector_login_name = st.text_input(

            "👷 Your Name",

            value=st.session_state.inspector_login_name,

            placeholder="Example: Rahul Sharma"

        )

    conn = sqlite3.connect(
        DB_NAME
    )

    history = conn.execute(

        """

        SELECT

    id,

    inspection_id,

    prediction,

    confidence,

    severity,

    risk_score,

    repair_priority,

    status,

    inspection_date,

    location,

    latitude,

    longitude,

    assigned_municipality,

    assigned_inspector,

    target_repair_date,

    citizen_voice_note

FROM inspections

ORDER BY risk_score DESC, id DESC

        """

    ).fetchall()

    conn.close()

    if is_inspector and not st.session_state.inspector_login_name.strip():

        st.info(
            "👆 Enter your name above to see your "
            "assigned complaints."
        )

        history = []

    elif is_inspector:

        history = [

            record
            for record in history
            if (record[13] or "").strip().lower()
            == st.session_state.inspector_login_name.strip().lower()

        ]

    elif is_officer and not st.session_state.officer_municipality.strip():

        st.info(
            "👆 Enter your municipality/area above to see "
            "complaints from your jurisdiction."
        )

        history = []

    elif is_officer:

        history = [

            record
            for record in history
            if st.session_state.officer_municipality.strip().lower()
            in (record[12] or "").strip().lower()

        ]

    history = sorted(

        history,

        key=lambda r: not is_escalated(r[14], r[7])

    )

    if not history:

        if not (
            (
                is_inspector
                and not st.session_state.inspector_login_name.strip()
            )
            or (
                is_officer
                and not st.session_state.officer_municipality.strip()
            )
        ):

            st.info(
                "No complaints available."
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

    database_id,

    inspection_id,

    prediction,

    confidence,

    severity,

    risk_score,

    priority,

    status,

    inspection_date,

    location,

    latitude,

    longitude,

    assigned_municipality,

    assigned_inspector,

    target_repair_date,

    citizen_voice_note

) = record
            if search.lower() not in prediction.lower():

                continue

            if (

                severity_filter != "All"

                and severity != severity_filter

            ):

                continue

            found = True

            with st.expander(

                f"#{inspection_id or database_id} | "

                f"{prediction} | "

                f"Risk: {risk_score}/100"

                + (
                    " | 🔺 ESCALATED"
                    if is_escalated(target_repair_date, status)
                    else (
                        " | 🔴 OVERDUE"
                        if is_overdue(target_repair_date, status)
                        else ""
                    )
                )

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

                    f"**🏛️ Assigned Municipality:** "

                    f"{assigned_municipality or 'Not Assigned'}"

                )

                st.write(

                    f"**👷 Assigned Inspector:** "

                    f"{assigned_inspector or 'Not Assigned'}"

                )

                st.write(

                    f"**📅 Target Repair Date:** "

                    f"{target_repair_date or 'Not Set'}"

                )

                if is_overdue(target_repair_date, status):

                    st.error(
                        "🔴 Overdue — past target repair date"
                    )

                if citizen_voice_note:

                    st.info(
                        f"🎙️ **Citizen Voice Note:** "
                        f"{citizen_voice_note}"
                    )

                st.write(

                    f"**Inspection Date:** "

                    f"{inspection_date}"

                )

                st.write(

                    f"**📍 Location:** "

                    f"{location}"

                )

                st.write(

                    f"**Latitude:** "

                    f"{latitude}"

                )

                st.write(

                    f"**Longitude:** "

                    f"{longitude}"

                )

                st.markdown(
                    "---"
                )

                st.markdown(
                    "### 🎯 Smart Action Center"
                )
                st.markdown(
                    "---"
                )

                if is_officer:

                    st.markdown(
                        "### 👷 Assign Inspector"
                    )

                    st.caption(
                        "Assign an inspector to visit "
                        "the site and verify the issue."
                    )

                    inspector_input_col, inspector_btn_col = (
                        st.columns([3, 1])
                    )

                    with inspector_input_col:

                        inspector_name_input = st.text_input(

                            "👷 Inspector Name",

                            value=(
                                assigned_inspector or ""
                            ),

                            placeholder="Example: Rahul Sharma",

                            key=f"inspector_name_{record[0]}"

                        )

                    with inspector_btn_col:

                        st.write(
                            ""
                        )

                        if st.button(

                            "💾 Assign",

                            key=f"assign_inspector_{record[0]}",

                            use_container_width=True

                        ):

                            assign_inspector(

                                database_id,

                                inspector_name_input

                            )

                            st.success(
                                "✅ Inspector assigned successfully!"
                            )

                            st.rerun()

                    st.markdown(
                        "---"
                    )

                if is_inspector:

                    st.markdown(
                        "### 🛠️ Repair Tracking"
                    )

                    st.caption(
                        "Track repair progress and maintenance details."
                    )

                    repair_col1, repair_col2 = st.columns(
                        2
                    )

                    with repair_col1:

                        repair_start_date = st.date_input(
                          "📅 Repair Start Date",
                          key=f"repair_start_{record[0]}"
    )

                    with repair_col2:

                        repair_completion_date = st.date_input(
                         "✅ Repair Completion Date",
                         key=f"repair_completion_{record[0]}"
    )

                    estimated_repair_cost = st.number_input(
                        "💰 Estimated Repair Cost",
                        min_value=0.0,
                        step=100.0,
                        key=f"repair_cost_{record[0]}"
                    )

                    repair_notes = st.text_area(
                        "📝 Repair Notes",
                        placeholder=(
                            "Example: Pothole filling and road surface repair required."
                        ),
                        key=f"repair_notes_{record[0]}"
                    )

                    repair_photo = st.file_uploader(

                        "📸 After-Repair Photo (proof of work)",

                        type=["jpg", "jpeg", "png"],

                        key=f"repair_photo_{record[0]}"

                    )

                    if st.button(
                        "💾 SAVE REPAIR DETAILS",
                        key=f"save_repair_{record[0]}",
                        use_container_width=True
                    ):

                        saved_photo_path = None

                        if repair_photo is not None:

                            os.makedirs(
                                "repair_photos",
                                exist_ok=True
                            )

                            saved_photo_path = (
                                f"repair_photos/"
                                f"{inspection_id}_"
                                f"{repair_photo.name}"
                            )

                            with open(
                                saved_photo_path,
                                "wb"
                            ) as f:

                                f.write(
                                    repair_photo.getbuffer()
                                )

                        update_repair_details(

                            inspection_id,

                            str(repair_start_date),

                            str(repair_completion_date),

                            estimated_repair_cost,

                            repair_notes,

                            saved_photo_path

                        )

                        st.success(
                            "✅ Repair details saved successfully!"
                        )

                        st.rerun()

                    st.caption(

                        "Manage the current issue status "

                        "and track repair progress."

                    )

                    rb1, rb2 = st.columns(
                        2
                    )

                    with rb1:

                        if st.button(

                            "🛠️ Repair In Progress",

                            key=(

                                f"repair_"

                                f"{database_id}"

                            ),

                            use_container_width=True

                        ):

                            update_inspection_status(

                                database_id,

                                "Repair In Progress"

                            )

                            st.success(

                                "Repair status updated "

                                "successfully!"

                            )

                            st.rerun()

                    with rb2:

                        if st.button(

                            "✅ Resolved",

                            key=(

                                f"resolved_"

                                f"{database_id}"

                            ),

                            use_container_width=True

                        ):

                            update_inspection_status(

                                database_id,

                                "Resolved"

                            )

                            st.success(

                                "Issue marked as resolved!"

                            )

                            st.rerun()

                if is_officer:

                    st.caption(

                        "Once an inspector is assigned, mark "

                        "the complaint as under inspection."

                    )

                    if st.button(

                        "🔍 Mark Under Inspection",

                        key=(

                            f"inspect_"

                            f"{database_id}"

                        ),

                        use_container_width=True

                    ):

                        update_inspection_status(

                            database_id,

                            "Under Inspection"

                        )

                        st.success(

                            "Inspection started "

                            "successfully!"

                        )

                        st.rerun()

        if not found:

            st.warning(

                "No matching inspection found."

            )

        if is_officer and history:

            st.markdown(
                "---"
            )

            st.markdown(
                "### 📈 Inspector Performance"
            )

            inspector_stats = {}

            for record in history:

                inspector_name = (
                    record[13] or "Unassigned"
                ).strip() or "Unassigned"

                record_status = record[7]

                stats = inspector_stats.setdefault(

                    inspector_name,

                    {
                        "total": 0,
                        "resolved": 0
                    }

                )

                stats["total"] += 1

                if record_status == "Resolved":
                    stats["resolved"] += 1

            perf_df = pd.DataFrame(

                [

                    {

                        "Inspector": name,

                        "Total Assigned": s["total"],

                        "Resolved": s["resolved"],

                        "Pending": s["total"] - s["resolved"]

                    }

                    for name, s in inspector_stats.items()

                    if name != "Unassigned"

                ]

            )

            if perf_df.empty:

                st.caption(
                    "No inspectors assigned yet in this area."
                )

            else:

                st.dataframe(

                    perf_df,

                    use_container_width=True,

                    hide_index=True

                )


# =========================================================
# TRACK MY COMPLAINT (CITIZEN)
# =========================================================

elif st.session_state.page == "Track My Complaint":

    st.markdown(
        f"## {t('track_complaint_title')}"
    )

    st.caption(
        "Enter your Complaint ID to check the current status."
    )

    lookup_id = st.text_input(

        t('complaint_id_label'),

        placeholder="Example: VG-20260730-0001"

    )

    if st.button(

        t('check_status_button'),

        use_container_width=True

    ):

        if not lookup_id.strip():

            st.warning(
                "Please enter a Complaint ID."
            )

        else:

            conn = sqlite3.connect(
                DB_NAME
            )

            row = conn.execute(

                """

                SELECT

                    inspection_id,

                    prediction,

                    severity,

                    risk_score,

                    repair_priority,

                    status,

                    assigned_municipality,

                    target_repair_date,

                    inspection_date,

                    location,

                    repair_photo_path

                FROM inspections

                WHERE inspection_id = ?

                """,

                (
                    lookup_id.strip(),
                )

            ).fetchone()

            conn.close()

            if not row:

                st.error(
                    "❌ No complaint found with this ID. "
                    "Please check and try again."
                )

            else:

                (
                    inspection_id,
                    prediction,
                    severity,
                    risk_score,
                    repair_priority,
                    status,
                    assigned_municipality,
                    target_repair_date,
                    inspection_date,
                    location,
                    repair_photo_path
                ) = row

                st.success(
                    f"✅ Complaint #{inspection_id} found"
                )

                if is_overdue(target_repair_date, status):

                    st.error(
                        "🔴 This complaint is Overdue — "
                        "past the target repair date."
                    )

                t1, t2, t3 = st.columns(3)

                t1.metric(
                    t('status_label'),
                    status
                )

                t2.metric(
                    t('severity_label'),
                    severity
                )

                t3.metric(
                    t('risk_score_label'),
                    f"{risk_score}/100"
                )

                st.write(
                    f"**🛣️ Detected Issue:** {prediction}"
                )

                st.write(
                    f"**🎯 Repair Priority:** {repair_priority}"
                )

                st.write(
                    f"**🏛️ Assigned Municipality:** "
                    f"{assigned_municipality or 'Not Assigned Yet'}"
                )

                st.write(
                    f"**📅 Target Repair Date:** "
                    f"{target_repair_date or 'Not Set Yet'}"
                )

                st.write(
                    f"**📍 Location:** {location}"
                )

                st.write(
                    f"**🗓️ Complaint Date:** {inspection_date}"
                )

                progress_map = {

                    "New": 0.25,

                    "Under Inspection": 0.5,

                    "Repair In Progress": 0.75,

                    "Resolved": 1.0

                }

                st.progress(
                    progress_map.get(status, 0.1)
                )

                if repair_photo_path and os.path.exists(
                    repair_photo_path
                ):

                    st.markdown(
                        "#### 📸 After-Repair Photo"
                    )

                    st.image(
                        repair_photo_path,
                        use_container_width=True
                    )


# =========================================================
# SUPER ADMIN DASHBOARD
# =========================================================

elif st.session_state.page == "Super Admin Dashboard":

    if st.session_state.role != "👑 Super Admin":

        st.warning(
            "🚫 This section is only for Super Admins."
        )

        st.stop()

    st.markdown(
        "## 👑 Super Admin — City-Wide Oversight"
    )

    st.caption(
        "Combined view across all municipalities and areas."
    )

    conn = sqlite3.connect(
        DB_NAME
    )

    admin_df = pd.read_sql_query(

        """

        SELECT *

        FROM inspections

        """,

        conn

    )

    conn.close()

    if admin_df.empty:

        st.info(
            "No complaints recorded yet."
        )

    else:

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Total Complaints",
            len(admin_df)
        )

        a2.metric(
            "Resolved",
            len(
                admin_df[
                    admin_df["status"] == "Resolved"
                ]
            )
        )

        a3.metric(
            "Critical Severity",
            len(
                admin_df[
                    admin_df["severity"] == "Critical"
                ]
            )
        )

        overdue_count = sum(

            1
            for _, r in admin_df.iterrows()
            if is_overdue(
                r.get("target_repair_date"),
                r.get("status")
            )

        )

        a4.metric(
            "🔴 Overdue",
            overdue_count
        )

        st.markdown(
            "---"
        )

        st.markdown(
            "### 🏛️ Area-Wise Breakdown"
        )

        area_summary = []

        for area, group in admin_df.groupby(
            "assigned_municipality"
        ):

            total = len(group)

            resolved = len(
                group[group["status"] == "Resolved"]
            )

            area_overdue = sum(

                1
                for _, r in group.iterrows()
                if is_overdue(
                    r.get("target_repair_date"),
                    r.get("status")
                )

            )

            area_summary.append(
                {
                    "Area": area or "Unassigned",
                    "Total Complaints": total,
                    "Resolved": resolved,
                    "Resolution %": (
                        round(
                            resolved / total * 100,
                            1
                        )
                        if total
                        else 0
                    ),
                    "🔴 Overdue": area_overdue,
                    "Avg Risk Score": round(
                        group["risk_score"].mean(),
                        1
                    )
                }
            )

        area_df = pd.DataFrame(
            area_summary
        ).sort_values(

            "🔴 Overdue",

            ascending=False

        )

        st.dataframe(

            area_df,

            use_container_width=True,

            hide_index=True

        )

        st.markdown(
            "---"
        )

        st.markdown(
            "### 📈 Complaints by Status"
        )

        status_fig = px.pie(

            admin_df,

            names="status",

            hole=0.45

        )

        st.plotly_chart(

            status_fig,

            use_container_width=True

        )


# =========================================================
# PUBLIC TRANSPARENCY MAP
# =========================================================

elif st.session_state.page == "Public Map":

    st.markdown(
        "## 🗺️ Public Transparency Map"
    )

    st.caption(
        "Live view of all reported infrastructure issues — "
        "open to everyone for transparency."
    )

    # -----------------------------------------------------
    # CONNECT TO DATABASE
    # -----------------------------------------------------

    conn = sqlite3.connect(DB_NAME)

    map_df = pd.read_sql_query(
        """
        SELECT
            inspection_id,
            prediction,
            severity,
            risk_score,
            status,
            assigned_municipality,
            latitude,
            longitude
        FROM inspections
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND latitude != 0
          AND longitude != 0
        """,
        conn
    )

    conn.close()

    # -----------------------------------------------------
    # CLEAN LOCATION DATA
    # -----------------------------------------------------

    if not map_df.empty:

        map_df["latitude"] = pd.to_numeric(
            map_df["latitude"],
            errors="coerce"
        )

        map_df["longitude"] = pd.to_numeric(
            map_df["longitude"],
            errors="coerce"
        )

        map_df["risk_score"] = pd.to_numeric(
            map_df["risk_score"],
            errors="coerce"
        )

        # Remove invalid coordinates
        map_df = map_df.dropna(
            subset=["latitude", "longitude"]
        )

        # Keep only valid geographical coordinates
        map_df = map_df[
            (map_df["latitude"].between(-90, 90)) &
            (map_df["longitude"].between(-180, 180))
        ]

        # Prevent zero/negative marker size
        map_df["risk_score"] = map_df["risk_score"].fillna(1)

        map_df["risk_score"] = map_df["risk_score"].clip(
            lower=1
        )

    # -----------------------------------------------------
    # DEBUG INFORMATION
    # -----------------------------------------------------

    st.write(
        "📍 Locations available:",
        len(map_df)
    )

    # -----------------------------------------------------
    # MAP
    # -----------------------------------------------------

    if map_df.empty:

        st.warning(
            "⚠️ No valid complaint location data is available."
        )

        st.info(
            "Complaints must contain valid latitude and longitude "
            "values before they can appear on the public map."
        )

    else:

        # Center map automatically around complaint locations
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        map_fig = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            color="severity",
            size="risk_score",
            hover_name="inspection_id",
            hover_data=[
                "prediction",
                "status",
                "assigned_municipality",
                "latitude",
                "longitude"
            ],
            zoom=10,
            height=560,
            map_style="open-street-map"
        )

        map_fig.update_layout(
            map={
                "center": {
                    "lat": center_lat,
                    "lon": center_lon
                },
                "zoom": 10
            },
            margin={
                "r": 0,
                "t": 0,
                "l": 0,
                "b": 0
            }
        )

        st.plotly_chart(
            map_fig,
            width="stretch"
        )

        # -------------------------------------------------
        # MAP STATISTICS
        # -------------------------------------------------

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Total Reported Issues",
            len(map_df)
        )

        m2.metric(
            "Resolved",
            len(
                map_df[
                    map_df["status"] == "Resolved"
                ]
            )
        )

        m3.metric(
            "Open Issues",
            len(
                map_df[
                    map_df["status"] != "Resolved"
                ]
            )
        )
# =========================================================
# AI ASSISTANT (CHATBOT)
# =========================================================

elif st.session_state.page == "AI Assistant":

    st.markdown(
        f"## {t('ai_assistant_title')}"
    )

    st.caption(
        "Ask about how VisionGuard AI works, your complaint "
        "process, infrastructure damage types, or general help."
    )

    for chat_message in st.session_state.chat_history:

        with st.chat_message(
            chat_message["role"]
        ):

            st.write(
                chat_message["content"]
            )

    with st.container(key="mic_row"):

        mic_col, spacer_col = st.columns(
            [1, 6]
        )

        with mic_col:

            with st.popover(
                "🎙️"
            ):

                chat_voice = st.audio_input(

                    "Tap to Record",

                    key="chat_voice_recorder"

                )

    voice_question = None

    if chat_voice is not None:

        chat_voice_bytes = chat_voice.getvalue()

        if st.session_state.get(
            "_last_chat_voice_len"
        ) != len(chat_voice_bytes):

            with st.spinner(
                "🎙️ Transcribing your question..."
            ):

                try:

                    chat_transcription = (
                        groq_client.audio.transcriptions.create(

                            file=(
                                "chat_voice.wav",
                                chat_voice_bytes
                            ),

                            model="whisper-large-v3"

                        )
                    )

                    voice_question = (
                        chat_transcription.text
                    )

                    st.session_state._last_chat_voice_len = (
                        len(chat_voice_bytes)
                    )

                except Exception as e:

                    st.error(
                        f"⚠️ Transcription failed: {str(e)}"
                    )

    try:

        prompt = st.chat_input(

            "Ask me anything about VisionGuard AI... "
            "(📎 attach a photo too)",

            accept_file=True,

            file_type=["jpg", "jpeg", "png"]

        )

        chat_input_supports_files = True

    except TypeError:

        prompt = st.chat_input(

            "Ask me anything about VisionGuard AI..."

        )

        chat_input_supports_files = False

    typed_question = None

    attached_photo = None

    if prompt:

        if chat_input_supports_files:

            typed_question = prompt.text

            if prompt.files:

                attached_photo = prompt.files[0]

        else:

            typed_question = prompt

    if attached_photo is not None:

        photo_image = Image.open(
            attached_photo
        )

        with st.chat_message("user"):

            st.image(
                photo_image,
                width=250
            )

            if typed_question:

                st.write(
                    typed_question
                )

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": (
                    typed_question
                    or "📷 [Shared a photo]"
                )
            }
        )

        with st.spinner(
            "🤖 Analyzing photo..."
        ):

            try:

                (
                    photo_prediction,
                    photo_confidence,
                    _,
                    _
                ) = predict(
                    photo_image
                )

                photo_reply = (

                    f"🔍 This looks like **"
                    f"{photo_prediction}** "
                    f"({photo_confidence * 100:.1f}% "
                    f"confidence).\n\nIf you'd like to "
                    f"file an official complaint for "
                    f"this, go to **New Inspection** "
                    f"from the sidebar — I've already "
                    f"identified the damage type for you."

                )

            except Exception as e:

                photo_reply = (
                    f"⚠️ Couldn't analyze this photo. "
                    f"({str(e)})"
                )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": photo_reply
            }
        )

        with st.chat_message("assistant"):

            st.write(
                photo_reply
            )

    user_question = typed_question or voice_question

    if user_question and attached_photo is None:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.write(
                user_question
            )

        system_prompt = (

            "You are the VisionGuard AI Assistant, a helpful "
            "support chatbot for a citizen infrastructure "
            "complaint platform. Citizens upload photos of road "
            "damage (potholes, damaged roads, broken road signs), "
            "GPS is auto-detected, an AI model analyzes risk and "
            "severity, and the complaint is automatically routed "
            "to the nearest Municipal Corporation. A Municipality "
            "Officer assigns an Inspector, who tracks repair "
            "progress through statuses: New -> Under Inspection "
            "-> Repair In Progress -> Resolved. Citizens can track "
            "their complaint using their Complaint ID on the "
            "'Track My Complaint' page. Answer clearly and "
            "concisely, and if asked something unrelated to this "
            "platform, gently redirect back to how you can help "
            "with infrastructure complaints."

        )

        try:

            with st.spinner(
                "🤖 Thinking..."
            ):

                response = groq_client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=(
                        [
                            {
                                "role": "system",
                                "content": system_prompt
                            }
                        ]
                        + st.session_state.chat_history
                    ),

                    temperature=0.4,

                    max_tokens=500

                )

                bot_reply = (
                    response.choices[0].message.content
                )

        except Exception as e:

            bot_reply = (
                f"⚠️ AI Assistant is currently unavailable. "
                f"({str(e)})"
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": bot_reply
            }
        )

        with st.chat_message("assistant"):

            st.write(
                bot_reply
            )

    if st.session_state.chat_history and st.button(

        "🗑️ Clear Chat"

    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# NEW INSPECTION
# =========================================================

elif st.session_state.page == "New Inspection":

    if st.session_state.role != "👤 Citizen":

        st.warning(
            "🚫 Switch to the Citizen role to submit a "
            "new complaint."
        )

        st.stop()

    st.markdown(
        f"""
        <div class="upload-card">
            <h2>{t('new_inspection_title')}</h2>
            <p>
                {t('new_inspection_caption')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(

        t('upload_image_label'),

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )

    # =====================================================
    # VOICE COMPLAINT DESCRIPTION (OPTIONAL)
    # =====================================================

    st.markdown(
        f"### {t('voice_section_title')}"
    )

    st.caption(
        t('voice_caption')
    )

    try:

        voice_note = st.audio_input(

            t('voice_upload_label'),

            key="voice_note_recorder"

        )

    except AttributeError:

        st.warning(
            "🎙️ Live mic recording needs a newer Streamlit "
            "version. Falling back to file upload."
        )

        voice_note = st.file_uploader(

            t('voice_upload_label'),

            type=[
                "wav",
                "mp3",
                "m4a"
            ],

            key="voice_note_uploader"

        )

    if voice_note is not None:

        voice_bytes = voice_note.getvalue()

        if st.session_state.get(
            "_last_voice_bytes_len"
        ) != len(voice_bytes):

            with st.spinner(
                "🎙️ Transcribing your voice note..."
            ):

                try:

                    transcription = (
                        groq_client.audio.transcriptions.create(

                            file=(
                                "voice_note.wav",
                                voice_bytes
                            ),

                            model="whisper-large-v3"

                        )
                    )

                    st.session_state.voice_note_text = (
                        transcription.text
                    )

                    st.session_state._last_voice_bytes_len = (
                        len(voice_bytes)
                    )

                except Exception as e:

                    st.error(
                        f"⚠️ Transcription failed: {str(e)}"
                    )

    if st.session_state.get("voice_note_text"):

        st.text_area(

            "📝 Transcribed Description",

            value=st.session_state.voice_note_text,

            key="voice_note_display"

        )


    # =====================================================
    # AUTOMATIC GPS LOCATION DETECTION
    # =====================================================

    st.markdown(
        "### 📍 Inspection Location"
    )

    st.caption(

        "📡 Your current GPS location "

        "will be detected automatically."

    )


    # =====================================================
    # GPS DETECTION
    # =====================================================

    location_data = streamlit_js_eval(

        js_expressions="""

        new Promise((resolve) => {

            if (!navigator.geolocation) {

                resolve({

                    latitude: 0,

                    longitude: 0

                });

                return;

            }

            navigator.geolocation.getCurrentPosition(

                (position) => {

                    resolve({

                        latitude:

                        position.coords.latitude,

                        longitude:

                        position.coords.longitude

                    });

                },

                () => {

                    resolve({

                        latitude: 0,

                        longitude: 0

                    });

                }

            );

        })

        """,

        key="gps_location"

    )


    # =====================================================
    # SAVE GPS COORDINATES
    # =====================================================

    if (

        location_data

        and

        isinstance(

            location_data,

            dict

        )

        and

        location_data.get(

            "latitude",

            0

        ) != 0

    ):

        st.session_state.latitude = float(

            location_data[

                "latitude"

            ]

        )

        st.session_state.longitude = float(

            location_data[

                "longitude"

            ]

        )

        st.session_state.gps_detected = True


    # =====================================================
    # DISPLAY GPS STATUS
    # =====================================================

    if st.session_state.gps_detected:

        st.success(

            "✅ GPS Location Detected Automatically"

        )

        if not st.session_state.address:

            with st.spinner(

                "📍 Detecting address from GPS..."

            ):

                st.session_state.address = (

                    get_address_from_coordinates(

                        st.session_state.latitude,

                        st.session_state.longitude

                    )

                )

        st.info(

            f"📌 Detected Address: "

            f"{st.session_state.address}"

        )

    else:

        st.warning(

            "📍 Please allow location access "

            "in your browser."

        )


    # =====================================================
    # LOCATION DETAILS
    # =====================================================

    loc_col1, loc_col2 = st.columns(
        2
    )

    with loc_col1:

        latitude = st.number_input(

            "📍 Latitude",

            min_value=-90.0,

            max_value=90.0,

            value=float(

                st.session_state.latitude

            ),

            format="%.6f"

        )

    with loc_col2:

        longitude = st.number_input(

            "📍 Longitude",

            min_value=-180.0,

            max_value=180.0,

            value=float(

                st.session_state.longitude

            ),

            format="%.6f"

        )

    st.session_state.latitude = latitude

    st.session_state.longitude = longitude


    # =====================================================
    # LOCATION NAME
    # =====================================================

    location = st.text_input(

        "📌 Location Name",

        value=(

            st.session_state.location

            or

            st.session_state.address

        ),

        placeholder=(

            "Example: SG Highway, Ahmedabad"

        )

    )

    st.session_state.location = location


    # =====================================================
    # MAP PREVIEW
    # =====================================================

    if (

        latitude != 0

        and

        longitude != 0

    ):

        st.markdown(

            "### 🗺️ Inspection Location Preview"

        )

        map_data = pd.DataFrame(

            {

                "latitude": [

                    latitude

                ],

                "longitude": [

                    longitude

                ]

            }

        )

        st.map(

            map_data,

            zoom=15

        )


    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    if uploaded is not None:

        image = Image.open(

            uploaded

        ).convert(

            "RGB"

        )

        st.image(

            image,

            caption=(

                "📷 Uploaded Infrastructure Image"

            ),

            use_container_width=True

        )

        if st.button(

            "🔍 ANALYZE INFRASTRUCTURE",

            use_container_width=True

        ):

            with st.spinner(

                "🔍 CNN is analyzing "

                "infrastructure damage..."

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

        prediction = (

            st.session_state.prediction

        )

        confidence = (

            st.session_state.confidence

        )

        image = (

            st.session_state.image

        )

        heatmap = (

            st.session_state.heatmap

        )

        probabilities = (

            st.session_state.probabilities

        )

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

                overlay = (

                    create_heatmap_overlay(

                        image,

                        heatmap

                    )

                )

                if overlay is not None:

                    st.image(

                        overlay,

                        caption=(

                            "Areas influencing "

                            "CNN prediction"

                        ),

                        use_container_width=True

                    )

                else:

                    st.warning(

                        "Grad-CAM visualization "

                        "unavailable."

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

            for (

                class_name,

                probability

            ) in probabilities.items():

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


        # =================================================
        # AUTOMATIC MUNICIPALITY ASSIGNMENT
        # =================================================

        st.markdown(

            "## 🏛️ Complaint Assignment"

        )

        assigned_municipality = get_municipality_department(
            st.session_state.latitude,
            st.session_state.longitude
        )

        st.session_state.assigned_municipality = (
            assigned_municipality
        )

        municipality_col, date_col = st.columns(

            2

        )

        with municipality_col:

            st.success(

                f"🏛️ Automatically Assigned To: "

                f"**{assigned_municipality}**"

            )

        with date_col:

            target_repair_date = st.date_input(

                "📅 Target Repair Date",

                value=(

                    st.session_state.target_repair_date

                ),

                min_value=date.today()

            )

            st.session_state.target_repair_date = (

                target_repair_date

            )

        st.info(

            "💡 The complaint is automatically routed to the "

            "municipal corporation nearest to your detected "

            "GPS location."

        )

        st.markdown(
            "---"
        )


        # =================================================
        # AI ASSESSMENT
        # =================================================

        st.markdown(

            "## 🤖 AI Infrastructure Assessment"

        )

        if st.session_state.ai_result is None:

            if st.button(

                "🤖 GENERATE AI INSPECTION ASSESSMENT",

                use_container_width=True

            ):

                with st.spinner(

                    "🤖 AI is evaluating "

                    "infrastructure risk..."

                ):

                    ai_result = generate_ai_report(

                        image,

                        prediction,

                        confidence,

                        st.session_state.location,

                        st.session_state.latitude,

                        st.session_state.longitude

                    )

                if not isinstance(

                    ai_result,

                    dict

                ):

                    st.error(

                        "AI report format is invalid."

                    )

                else:

                    st.session_state.ai_result = (

                        ai_result

                    )

                    inspection_id = save_inspection(

                        prediction,

                        confidence,

                        ai_result,

                        st.session_state.location,

                        st.session_state.latitude,

                        st.session_state.longitude,

                        st.session_state.assigned_municipality,

                        str(

                            st.session_state.target_repair_date

                        ),

                        st.session_state.voice_note_text

                    )

                    st.success(

                        f"✅ Inspection Saved: "

                        f"{inspection_id}"

                    )

                    duplicates = find_duplicate_complaints(

                        prediction,

                        st.session_state.latitude,

                        st.session_state.longitude

                    )

                    duplicates = [
                        d for d in duplicates
                        if d[0] != inspection_id
                    ]

                    if duplicates:

                        st.warning(
                            f"⚠️ Heads up: {len(duplicates)} "
                            f"similar unresolved complaint(s) "
                            f"already exist near this location "
                            f"(e.g. {duplicates[0][0]}, "
                            f"status: {duplicates[0][1]}). "
                            f"Your complaint has still been "
                            f"recorded and will help prioritize "
                            f"the repair."
                        )

                    st.markdown(
                        "#### 🧾 Digital Complaint Receipt"
                    )

                    receipt_bytes = generate_complaint_receipt(

                        inspection_id,

                        prediction,

                        ai_result.get("severity", "Unknown"),

                        ai_result.get("risk_score", 0),

                        st.session_state.location,

                        st.session_state.assigned_municipality,

                        str(st.session_state.target_repair_date)

                    )

                    st.image(
                        receipt_bytes,
                        caption="Scan the QR code to quickly "
                                "recall your Complaint ID.",
                        width=400
                    )

                    st.download_button(

                        "⬇️ Download Receipt",

                        data=receipt_bytes,

                        file_name=f"{inspection_id}_receipt.png",

                        mime="image/png",

                        use_container_width=True

                    )

        else:

            ai_result = (

                st.session_state.ai_result

            )

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

                risk_label = (

                    "🔴 Critical Risk"

                )

            elif risk_score >= 60:

                risk_label = (

                    "🟠 High Risk"

                )

            elif risk_score >= 35:

                risk_label = (

                    "🟡 Moderate Risk"

                )

            else:

                risk_label = (

                    "🟢 Low Risk"

                )

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


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

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


ASSIGNED MUNICIPALITY:

{st.session_state.assigned_municipality}


TARGET REPAIR DATE:

{st.session_state.target_repair_date}


LOCATION:

{st.session_state.location}


LATITUDE:

{st.session_state.latitude}


LONGITUDE:

{st.session_state.longitude}


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


        # =================================================
        # MODEL PROBABILITY ANALYSIS
        # =================================================

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

        fig_probability = px.bar(

            chart_data,

            x="Class",

            y="Probability",

            text_auto=".2f"

        )

        st.plotly_chart(

            fig_probability,

            use_container_width=True,

            key="model_probability_chart"

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
