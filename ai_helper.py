import os
import json

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================================================
# AI INSPECTION REPORT
# =========================================================

def generate_ai_report(
    image,
    prediction,
    confidence,
    location="Unknown",
    latitude=0.0,
    longitude=0.0
):

    prompt = f"""
You are a Civil Infrastructure Inspection AI.

A CNN deep learning model analyzed a road image.

CNN Prediction:
{prediction}

CNN Confidence:
{confidence * 100:.2f}%

Inspection Location:
{location}

GPS Latitude:
{latitude}

GPS Longitude:
{longitude}

Generate a professional infrastructure inspection assessment.

Return ONLY valid JSON.

Use exactly this format:

{{
    "severity": "Critical",
    "risk_score": 85,
    "repair_priority": "Immediate",
    "public_safety_risk": "Description",
    "possible_causes": [
        "Cause 1",
        "Cause 2",
        "Cause 3"
    ],
    "recommended_action": "Recommended action",
    "preventive_measures": [
        "Measure 1",
        "Measure 2",
        "Measure 3"
    ],
    "inspector_remarks": "Professional inspector remarks",
    "estimated_repair_cost": "$500 - $1,500",
    "estimated_repair_duration": "2-4 Days",
    "required_workforce": "3-5 Workers"
}}

Rules:

- risk_score must be an integer from 0 to 100.

- severity must be one of:
  Critical, High, Moderate, Low.

- repair_priority must be one of:
  Immediate, Urgent, Scheduled, Routine.

- Do not invent exact location.

- Do not invent road name.

- Do not invent exact measurements.

- The assessment must be based on the CNN prediction.

- Use the provided GPS/location information only as inspection metadata.

- Do not make risk decisions solely based on location.

- This is an AI-assisted assessment.

- Return ONLY JSON.
"""

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=1200
        )

        result = response.choices[0].message.content

        # Remove markdown code fences if the model adds them
        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        )

        result = result.strip()

        # Convert JSON string into Python dictionary
        parsed_result = json.loads(result)

        # Add default values if fields are missing
        parsed_result.setdefault(
            "severity",
            "Unknown"
        )

        parsed_result.setdefault(
            "risk_score",
            0
        )

        parsed_result.setdefault(
            "repair_priority",
            "Routine"
        )

        parsed_result.setdefault(
            "public_safety_risk",
            "Not available"
        )

        parsed_result.setdefault(
            "possible_causes",
            []
        )

        parsed_result.setdefault(
            "recommended_action",
            "Manual inspection required."
        )

        parsed_result.setdefault(
            "preventive_measures",
            []
        )

        parsed_result.setdefault(
            "inspector_remarks",
            "AI-assisted assessment generated."
        )

        parsed_result.setdefault(
            "estimated_repair_cost",
            "Not available"
        )

        parsed_result.setdefault(
            "estimated_repair_duration",
            "Not available"
        )

        parsed_result.setdefault(
            "required_workforce",
            "Not available"
        )

        return parsed_result

    except Exception as e:

        return {

            "severity":
            "Unknown",

            "risk_score":
            0,

            "repair_priority":
            "Routine",

            "public_safety_risk":
            "AI assessment unavailable.",

            "possible_causes":
            [],

            "recommended_action":
            "Manual inspection required.",

            "preventive_measures":
            [],

            "inspector_remarks":
            f"AI Error: {str(e)}",

            "estimated_repair_cost":
            "Not available",

            "estimated_repair_duration":
            "Not available",

            "required_workforce":
            "Not available"

        }
