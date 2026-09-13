import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()
model = "qwen/qwen3.6-27b"

system_prompt = """You are a financial decision assistant. You must analyze the given user data and return your final recommendation.

You must output your response STRICTLY as a single, valid JSON object with the following fields:
{
  "request_id": "string (the request ID)",
  "amount_safe_to_pay": "number (maximum amount safe to pay today, 0 to requested_amount)",
  "affordability_status": "string (one of: affordable_now, affordable_with_plan, affordable_later, not_affordable)",
  "recommended_payment_method": "string (one of: full_payment, partial_payment, installments, wait, not_recommended)",
  "payment_plan": "string (chronological payments format YYYY-MM-DD:amount|YYYY-MM-DD:amount or 'none')",
  "earliest_date_for_full_payment": "string (YYYY-MM-DD or empty string)",
  "spending_changes_needed": "string (e.g. stop:event_14|reduce_to:event_21:100 or 'none')",
  "decision_explanation": "string (grounded explanation of the recommendation and financial facts)",
  "supporting_references": "array of strings (list of event_ids, message_ids, image_ids, payment_option_ids)"
}

Do not include any conversational filler, markdown formatting (like ```json), or text before or after the JSON. Return only the raw JSON object."""

user_prompt = "Please analyze request_26 for user_26. Requested amount is 15656000 IDR. Available balance is 100845250 IDR. The user wants to complete it by 2025-10-07. No partial payments are allowed. It is fully affordable now under full_payment. No spending changes are needed."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

try:
    print("Testing call with response_format={'type': 'json_object'}...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=500
    )
    content = response.choices[0].message.content
    print("Model response content:")
    print(content)
    
    # Verify it parses as valid JSON
    parsed = json.loads(content)
    print("\nSuccessfully parsed JSON object:")
    print(json.dumps(parsed, indent=2))
except Exception as e:
    print("Error:", e)
