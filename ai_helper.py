import os
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_ai_report(
    image,
    prediction,
    confidence
):

    prompt = f"""

You are a Civil Infrastructure Inspection AI.

A CNN deep learning model analyzed a road image.

CNN Prediction:
{prediction}

CNN Confidence:
{confidence * 100:.2f}%

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
    "inspector_remarks": "Professional inspector remarks"
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
- This is an AI-assisted assessment.
"""


    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

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

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        )

        result = result.strip()


        return json.loads(
            result
        )


    except Exception as e:

        return {

            "severity": "Unknown",

            "risk_score": 0,

            "repair_priority": "Routine",

            "public_safety_risk":
            "AI assessment unavailable.",

            "possible_causes": [],

            "recommended_action":
            "Manual inspection required.",

            "preventive_measures": [],

            "inspector_remarks":
            f"AI Error: {str(e)}"

        }