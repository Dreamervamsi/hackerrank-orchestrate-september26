import os
import json
import re
from groq import Groq
from typing import Dict, Any

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

    def extract_json_after_think(self, text: str) -> Dict[str, Any]:
        """Parse thinking response and return extracted JSON object."""
        if "</think>" in text:
            text_after = text.split("</think>")[-1]
        else:
            text_after = text
            
        first_brace = text_after.find('{')
        last_brace = text_after.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text_after[first_brace:last_brace+1]
            # Replace single quotes or unescaped strings if any, but standard json loads usually works
            return json.loads(json_str)
        raise ValueError("No valid JSON block found in response text")

    def process_request(self, request_id: str, user_id: str) -> Dict[str, Any]:
        """Pre-fetch all context and make a single-turn reasoning call to the model."""
        # 1. Gather all data deterministically
        request_data = self.data_manager.get_request(request_id)
        profile_data = self.data_manager.get_profile(user_id)
        
        req_date = request_data.get('request_date', '2025-08-03')
        events_data = self.data_manager.get_events(user_id)
        
        # Format events to compact
        events_json = self.data_manager.get_events(user_id)
        events_compact_data = self.data_manager.get_events(user_id)
        
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

        # 4. Extract and parse JSON
        decision = self.extract_json_after_think(raw_content)
        
        # 5. Overwrite LLM's arithmetic with deterministic simulation values
        min_balance_to_keep = float(profile_data.get('minimum_balance_to_keep', 0.0))
        requested_amount = float(request_data.get('requested_amount', 0.0))
        
        safe_amount = self.simulator.calculate_safe_amount(user_id, req_date, requested_amount, min_balance_to_keep)
        earliest_date = self.simulator.calculate_earliest_full_payment_date(user_id, requested_amount, req_date, min_balance_to_keep)
        
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
import pandas as pd
from datetime import timedelta
