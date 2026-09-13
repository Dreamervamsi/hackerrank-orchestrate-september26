import os
import json
import re
import pandas as pd
from datetime import timedelta
from typing import Dict, Any
from groq import Groq

class StructuredFinancialAgent:
    """Single-turn, pre-fetched structured LLM agent for the Buy or Wait? challenge."""
    
    SYSTEM_PROMPT = """You are a financial decision agent for the Buy or Wait? challenge.

Your task is to analyze a purchase/payment request and determine whether the user can safely afford it.

CRITICAL CONSTRAINT:
Your thinking/reasoning process inside the <think> tag MUST be extremely brief (under 50 words). Get straight to the point and close the think tag quickly.

At the end of your response, output STRICTLY a single, valid JSON object with the following fields:
{
  "request_id": "string",
  "amount_safe_to_pay": number,
  "affordability_status": "string (affordable_now, affordable_with_plan, affordable_later, not_affordable)",
  "recommended_payment_method": "string (full_payment, partial_payment, installments, wait, not_recommended)",
  "payment_plan": "string (chronological payments format YYYY-MM-DD:amount|YYYY-MM-DD:amount or 'none')",
  "earliest_date_for_full_payment": "string (YYYY-MM-DD or empty)",
  "spending_changes_needed": "string (stop:event_id|reduce_to:event_id:new_amount or 'none')",
  "decision_explanation": "string (concise explanation of recommendation and financial facts)",
  "supporting_references": ["string (list of event_ids, message_ids, image_ids, payment_option_ids)"]
}

Do not wrap the JSON object in markdown codeblocks like ```json. Return ONLY the raw JSON string."""

    def __init__(self, data_manager, simulator, model: str = "qwen/qwen3.6-27b"):
        self.data_manager = data_manager
        self.simulator = simulator
        self.model = model
        
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)

    def extract_json_after_think(self, text: str, request_id: str = "") -> Dict[str, Any]:
        """Parse thinking response and return extracted JSON object with extreme resilience."""
        if "</think>" in text:
            text_after = text.split("</think>")[-1]
        else:
            text_after = text
            
        first_brace = text_after.find('{')
        last_brace = text_after.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text_after[first_brace:last_brace+1]
            
            # Clean up markdown code block tags if present
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            try:
                return json.loads(json_str)
            except Exception as e:
                # Attempt regex-based key-value extraction for robustness
                try:
                    reconstructed = {}
                    
                    # 1. Match string fields
                    for field in ['request_id', 'affordability_status', 'recommended_payment_method', 
                                  'payment_plan', 'earliest_date_for_full_payment', 
                                  'spending_changes_needed', 'decision_explanation']:
                        match = re.search(f'"{field}"\\s*:\\s*"([^"]*)"', json_str)
                        if match:
                            reconstructed[field] = match.group(1)
                            
                    # 2. Match numeric fields
                    match_num = re.search(r'"amount_safe_to_pay"\s*:\s*([\d\.]+)', json_str)
                    if match_num:
                        reconstructed['amount_safe_to_pay'] = float(match_num.group(1))
                        
                    # 3. Match array fields
                    match_ref = re.search(r'"supporting_references"\s*:\s*\[([^\]]*)\]', json_str)
                    if match_ref:
                        ref_str = match_ref.group(1)
                        reconstructed['supporting_references'] = [r.strip().replace('"', '') for r in ref_str.split(',') if r.strip()]
                    else:
                        reconstructed['supporting_references'] = []
                        
                    # Verify we got the essential fields
                    required = ['request_id', 'affordability_status', 'recommended_payment_method', 
                                'payment_plan', 'earliest_date_for_full_payment', 'spending_changes_needed', 
                                'decision_explanation']
                    if all(k in reconstructed for k in required):
                        if 'amount_safe_to_pay' not in reconstructed:
                            reconstructed['amount_safe_to_pay'] = 0.0
                        return reconstructed
                except Exception as inner_e:
                    pass
                
                raise ValueError(f"JSON loads failed and reconstruction failed: {str(e)}")
                
        raise ValueError("No valid JSON block found in response text")

    def process_request(self, request_id: str, user_id: str) -> Dict[str, Any]:
        """Process a request with deterministic Rule-First Fast Path and single-turn fallback LLM reasoning."""
        # 1. Gather all data deterministically
        request_data = self.data_manager.get_request(request_id)
        profile_data = self.data_manager.get_profile(user_id)
        
        req_date = request_data.get('request_date', '2025-08-03')
        requested_amount = float(request_data.get('requested_amount', 0.0))
        min_balance_to_keep = float(profile_data.get('minimum_balance_to_keep', 0.0))
        
        # Determine user preferences
        considered = profile_data.get('payment_methods_considered', profile_data.get('payment_methods_user_will_consider', []))
        if isinstance(considered, str):
            considered = considered.split('|')
            
        # ====================================================================
        # RULE-FIRST FAST PATH: Skip LLM completely for clear-cut safe requests!
        # ====================================================================
        safe_amount = self.simulator.calculate_safe_amount(user_id, req_date, requested_amount, min_balance_to_keep)
        earliest_date = self.simulator.calculate_earliest_full_payment_date(user_id, requested_amount, req_date, min_balance_to_keep)
        payment_options = self.data_manager.get_payment_options(request_id)
        
        # Fast Path A: Directly affordable under full_payment (most common case!)
        if safe_amount >= requested_amount and "full_payment" in considered:
            decision = {
                "request_id": request_id,
                "amount_safe_to_pay": requested_amount,
                "affordability_status": "affordable_now",
                "recommended_payment_method": "full_payment",
                "payment_plan": "none",
                "earliest_date_for_full_payment": req_date,
                "spending_changes_needed": "none",
                "decision_explanation": f"The request is fully affordable immediately. Paying the lump sum of {requested_amount:.2f} IDR leaves the available balance comfortably above the required minimum threshold of {min_balance_to_keep:.2f} IDR.",
                "supporting_references": ["system_simulator_fast_path"]
            }
            return {
                'decision': decision,
                'validation': {'passed': True, 'details': 'Fast-path passed check'},
                'token_stats': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'raw_content': "FAST_PATH_OK"
            }
            
        # Fast Path B: Not affordable now, but easily Waitable to earliest safe date (and no installments allowed)
        if safe_amount < requested_amount and earliest_date and "full_payment" in considered and len(payment_options) <= 1:
            decision = {
                "request_id": request_id,
                "amount_safe_to_pay": min(safe_amount, requested_amount),
                "affordability_status": "affordable_later",
                "recommended_payment_method": "wait",
                "payment_plan": "none",
                "earliest_date_for_full_payment": earliest_date,
                "spending_changes_needed": "none",
                "decision_explanation": f"Paying the full requested amount of {requested_amount:.2f} IDR today is not safe. It is highly recommended to wait until {earliest_date} when incoming cash flows restore the balance safely above the required minimum of {min_balance_to_keep:.2f} IDR.",
                "supporting_references": ["system_simulator_fast_path"]
            }
            return {
                'decision': decision,
                'validation': {'passed': True, 'details': 'Fast-path passed check'},
                'token_stats': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'raw_content': "FAST_PATH_OK"
            }

        events_json = self.data_manager.get_events(user_id)
        
        # Build highly compact events list (last 30 days prior to request_date)
        req_dt = pd.to_datetime(req_date)
        start_dt = req_dt - timedelta(days=30)
        
        recent_events = events_json.copy()
        recent_events['event_date_dt'] = pd.to_datetime(recent_events['event_date'])
        recent_events = recent_events[(recent_events['event_date_dt'] >= start_dt) & (recent_events['event_date_dt'] < req_dt)]
        if len(recent_events) == 0:
            recent_events = events_json.sort_values('event_date').tail(15)
            
        header = "event_id|type|desc|cat|dir|amount|curr|date|settle_date|status|flex"
        lines = [header]
        for idx, row in recent_events.iterrows():
            amount_val = f"{row['amount']:.2f}" if pd.notna(row['amount']) else "BLANK"
            line = f"{row['event_id']}|{row['event_type']}|{row['description']}|{row['category']}|{row['direction']}|{amount_val}|{row['currency']}|{row['event_date']}|{row['settlement_date']}|{row['status']}|{row['flexibility']}"
            lines.append(line)
        events_compact_text = "\n".join(lines)
        
        messages_data = [m for m in self.data_manager.messages_df.to_dict('records') if m['user_id'] == user_id]
        payment_options_data = self.data_manager.get_payment_options(request_id)
        
        # 2. Construct clean user prompt
        user_prompt = f"""You are analyzing purchase request {request_id} for user {user_id}.

I have pre-fetched all relevant financial data. Analyze it carefully and call the `make_decision` tool with your final decision.

### 1. REQUEST DETAILS
{json.dumps(request_data, indent=2)}

### 2. FINANCIAL PROFILE
{json.dumps(profile_data, indent=2)}

### 3. RECENT FINANCIAL EVENTS (COMPACT)
{events_compact_text}

### 4. RELEVANT MESSAGES
{json.dumps(messages_data, indent=2)}

### 5. AVAILABLE PAYMENT OPTIONS
{json.dumps(payment_options_data, indent=2)}

Based on these facts, run your financial reasoning, and output the required JSON object."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # 3. Call the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800
        )
        
        raw_content = response.choices[0].message.content
        
        # Get usage stats if available
        usage = getattr(response, 'usage', None)
        token_stats = {
            'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
            'completion_tokens': getattr(usage, 'completion_tokens', 0),
            'total_tokens': getattr(usage, 'total_tokens', 0)
        }

        # 4. Extract and parse JSON with robust fallback
        try:
            decision = self.extract_json_after_think(raw_content, request_id)
        except Exception as json_err:
            # Mathematical/programmatic fallback when JSON extraction fails
            # Formulate the safest programmatic plan
            if safe_amount >= requested_amount:
                affordability_status = "affordable_now"
                recommended_payment_method = "full_payment"
                payment_plan = "none"
                earliest_date_for_full_payment = req_date
                explanation = f"The requested amount of {requested_amount:.2f} is fully safe to pay immediately since the projected daily balance stays safely above the required minimum balance of {min_balance_to_keep:.2f} throughout the 90-day forecast."
            elif earliest_date:
                affordability_status = "affordable_later"
                recommended_payment_method = "wait"
                payment_plan = f"{earliest_date}:{requested_amount}"
                earliest_date_for_full_payment = earliest_date
                explanation = f"The request is affordable later. Waiting until {earliest_date} is recommended to avoid dropping the balance below the required minimum of {min_balance_to_keep:.2f}."
            else:
                affordability_status = "not_affordable"
                recommended_payment_method = "not_recommended"
                payment_plan = "none"
                earliest_date_for_full_payment = ""
                explanation = f"The requested payment is currently not affordable within the 90-day forecast period without violating the required minimum balance of {min_balance_to_keep:.2f}."
                
            decision = {
                "request_id": request_id,
                "amount_safe_to_pay": min(safe_amount, requested_amount),
                "affordability_status": affordability_status,
                "recommended_payment_method": recommended_payment_method,
                "payment_plan": payment_plan,
                "earliest_date_for_full_payment": earliest_date_for_full_payment,
                "spending_changes_needed": "none",
                "decision_explanation": explanation + " (Programmatic Fallback)",
                "supporting_references": ["system_simulator_fallback"]
            }

        # 5. Overwrite LLM's arithmetic with deterministic simulation values
        # Apply the mathematically precise numbers
        decision['amount_safe_to_pay'] = min(safe_amount, requested_amount)
        decision['earliest_date_for_full_payment'] = earliest_date
        
        # Validate using our simulator
        validation_passed, validation_details = self.simulator.validate_decision(
            decision, user_id, req_date, requested_amount, min_balance_to_keep
        )
        
        return {
            'decision': decision,
            'validation': {
                'passed': validation_passed,
                'details': validation_details
            },
            'token_stats': token_stats,
            'raw_content': raw_content
        }
