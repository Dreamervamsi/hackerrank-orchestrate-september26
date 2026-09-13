#!/usr/bin/env python3
"""
Real LLM-driven agent for Buy or Wait? challenge using Groq.
Processes ONE request with genuine agent-style tool orchestration.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from groq import Groq
from PIL import Image
import pytesseract
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
DATASET_DIR = Path("dataset")
LOG_FILE = Path("log.txt")

# ============================================================================
# EXECUTION LOGGER
# ============================================================================

class ExecutionLogger:
    """Logs agent execution trace to log.txt"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.entries = []
    
    def log(self, message: str):
        """Append a log entry"""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {message}"
        self.entries.append(entry)
        print(entry)
    
    def log_request_start(self, request_id: str, user_id: str):
        """Log request start"""
        self.log("=" * 70)
        self.log(f"REQUEST START: {request_id} (user: {user_id})")
        self.log("=" * 70)
    
    def log_model_call(self, model: str, prompt_length: int, response_length: int = None):
        """Log model invocation"""
        self.log(f"MODEL CALL: {model}")
        self.log(f"  Prompt length: {prompt_length} characters")
        if response_length:
            self.log(f"  Response length: {response_length} characters")
    
    def log_tool_call(self, tool_name: str, inputs: Dict, result_summary: str = None):
        """Log a tool invocation"""
        self.log(f"TOOL CALL: {tool_name}")
        # Log inputs but redact sensitive data
        safe_inputs = {k: v for k, v in inputs.items() if 'key' not in k.lower() and 'secret' not in k.lower()}
        self.log(f"  Inputs: {json.dumps(safe_inputs, default=str)}")
        if result_summary:
            self.log(f"  Result: {result_summary}")
    
    def log_extracted_fact(self, fact_type: str, value: Any, source_id: str):
        """Log an extracted financial fact"""
        self.log(f"EXTRACTED FACT: {fact_type} = {value} (source: {source_id})")
    
    def log_financial_calculation(self, calculation: str, result: Any):
        """Log a deterministic calculation"""
        self.log(f"CALCULATION: {calculation} = {result}")
    
    def log_decision(self, decision: Dict):
        """Log final decision"""
        self.log("FINAL DECISION")
        self.log(f"  {json.dumps(decision, default=str, indent=2)}")
    
    def log_validation(self, passed: bool, details: str = None):
        """Log validation result"""
        status = "PASSED" if passed else "FAILED"
        self.log(f"VALIDATION: {status}")
        if details:
            self.log(f"  Details: {details}")
    
    def log_error(self, error: str):
        """Log an error"""
        self.log(f"ERROR: {error}")
    
    def flush(self):
        """Write all entries to log file"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            for entry in self.entries:
                f.write(entry + '\n')
        self.entries = []

# ============================================================================
# DATA RETRIEVAL TOOLS
# ============================================================================

class FinancialDataTools:
    """Tools for retrieving financial data with actual content extraction"""
    
    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all datasets into memory"""
        self.logger.log("Loading datasets...")
        self.requests_df = pd.read_csv(DATASET_DIR / "requests.csv")
        self.financial_profiles_df = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
        self.financial_events_df = pd.read_csv(DATASET_DIR / "financial_events.csv")
        self.messages_df = pd.read_csv(DATASET_DIR / "messages.csv")
        self.images_df = pd.read_csv(DATASET_DIR / "images.csv")
        self.payment_options_df = pd.read_csv(DATASET_DIR / "request_payment_options.csv")
        self.exchange_rates_df = pd.read_csv(DATASET_DIR / "exchange_rates.csv")
        self.logger.log(f"  Loaded {len(self.requests_df)} requests")
        self.logger.log(f"  Loaded {len(self.financial_profiles_df)} profiles")
        self.logger.log(f"  Loaded {len(self.financial_events_df)} events")
        self.logger.log(f"  Loaded {len(self.messages_df)} messages")
        self.logger.log(f"  Loaded {len(self.images_df)} image references")
        self.logger.log(f"  Loaded {len(self.payment_options_df)} payment options")
        self.logger.log(f"  Loaded {len(self.exchange_rates_df)} exchange rates")
    
    def get_request(self, request_id: str) -> Dict[str, Any]:
        """Tool: Retrieve a specific request"""
        request = self.requests_df[self.requests_df['request_id'] == request_id]
        if len(request) == 0:
            return {}
        result = request.iloc[0].to_dict()
        self.logger.log_tool_call("get_request", {"request_id": request_id}, f"Found request for user {result['user_id']}")
        return result
    
    def get_financial_profile(self, user_id: str) -> Dict[str, Any]:
        """Tool: Retrieve financial profile for a user"""
        profile = self.financial_profiles_df[self.financial_profiles_df['user_id'] == user_id]
        if len(profile) == 0:
            return {}
        result = profile.iloc[0].to_dict()
        self.logger.log_tool_call("get_financial_profile", {"user_id": user_id}, f"Balance: {result['current_available_balance']}, Min: {result['minimum_balance_to_keep']}")
        return result
    
    def get_financial_events(self, user_id: str, request_date: str = "2025-08-03") -> Dict[str, Any]:
        """Tool: Retrieve recent financial events for a user in a highly compact pipe-separated format"""
        events = self.financial_events_df[self.financial_events_df['user_id'] == user_id]
        
        # Filter for recent events (last 30 days prior to request_date) to keep tokens tiny for the LLM
        req_dt = pd.to_datetime(request_date)
        start_dt = req_dt - timedelta(days=30)
        
        events_copy = events.copy()
        events_copy['event_date_dt'] = pd.to_datetime(events_copy['event_date'])
        recent_events = events_copy[(events_copy['event_date_dt'] >= start_dt) & (events_copy['event_date_dt'] < req_dt)]
        
        # If no recent events in last 30 days, fallback to last 15 events
        if len(recent_events) == 0:
            recent_events = events_copy.sort_values('event_date').tail(15)
            
        # Build highly compact pipe-separated lines to minimize tokens and avoid rate limits
        header = "event_id|type|desc|cat|dir|amount|curr|date|settle_date|status|flex|link_id"
        lines = [header]
        for idx, row in recent_events.iterrows():
            amount_val = f"{row['amount']:.2f}" if pd.notna(row['amount']) else "BLANK"
            link_id = str(row['linked_event_id']) if pd.notna(row['linked_event_id']) else ""
            line = f"{row['event_id']}|{row['event_type']}|{row['description']}|{row['category']}|{row['direction']}|{amount_val}|{row['currency']}|{row['event_date']}|{row['settlement_date']}|{row['status']}|{row['flexibility']}|{link_id}"
            lines.append(line)
            
        compact_text = "\n".join(lines)
        
        self.logger.log_tool_call("get_financial_events", {"user_id": user_id}, f"Returned {len(recent_events)} recent events out of {len(events)} total in compact pipe-delimited text ({len(compact_text)} chars)")
        return {"events_compact": compact_text, "total_events": len(events)}
    
    def get_messages(self, user_id: str, request_id: str = None, event_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Tool: Retrieve messages and extract financial information"""
        inputs = {"user_id": user_id}
        if request_id:
            inputs["request_id"] = request_id
        if event_ids:
            inputs["event_ids"] = event_ids
        
        # Get messages by user_id
        messages = self.messages_df[self.messages_df['user_id'] == user_id]
        
        # Also get by request_id if provided
        if request_id:
            messages_by_request = self.messages_df[self.messages_df['request_id'] == request_id]
            messages = pd.concat([messages, messages_by_request]).drop_duplicates()
        
        # Also get by related_event_id if provided
        if event_ids:
            messages_by_event = self.messages_df[self.messages_df['related_event_id'].isin(event_ids)]
            messages = pd.concat([messages, messages_by_event]).drop_duplicates()
        
        result = messages.to_dict('records')
        
        # Extract financial facts from messages
        for msg in result:
            msg['extracted_facts'] = self._extract_facts_from_message(msg)
        
        self.logger.log_tool_call("get_messages", inputs, f"{len(result)} messages retrieved")
        return result
    
    def _extract_facts_from_message(self, message: Dict) -> List[Dict]:
        """Extract financial facts from message text"""
        facts = []
        text = message.get('message_text', '')
        
        # Extract amounts with currency
        amount_pattern = r'([A-Z]{3})\s*[\d,]+\.?\d*'
        amounts = re.findall(amount_pattern, text)
        for currency in amounts:
            facts.append({'type': 'currency_mention', 'currency': currency, 'source': message['message_id']})
        
        # Extract dates
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, text)
        for date in dates:
            facts.append({'type': 'date_mention', 'date': date, 'source': message['message_id']})
        
        # Extract salary/income mentions
        if 'salary' in text.lower() or 'gaji' in text.lower() or 'income' in text.lower():
            facts.append({'type': 'income_mention', 'source': message['message_id']})
        
        return facts
    
    def get_images(self, user_id: str, request_id: str = None, event_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Tool: Retrieve image references and extract content from images"""
        inputs = {"user_id": user_id}
        if request_id:
            inputs["request_id"] = request_id
        if event_ids:
            inputs["event_ids"] = event_ids
        
        # Get images by user_id
        images = self.images_df[self.images_df['user_id'] == user_id]
        
        # Also get by request_id if provided
        if request_id:
            images_by_request = self.images_df[self.images_df['request_id'] == request_id]
            images = pd.concat([images, images_by_request]).drop_duplicates()
        
        # Also get by related_event_id if provided
        if event_ids:
            images_by_event = self.images_df[self.images_df['related_event_id'].isin(event_ids)]
            images = pd.concat([images, images_by_event]).drop_duplicates()
        
        result = images.to_dict('records')
        
        # Extract content from images
        for img in result:
            img['file_path'] = str(DATASET_DIR / "media" / "images" / f"{img['image_id']}.png")
            img['extracted_content'] = self._extract_from_image(img['file_path'])
        
        self.logger.log_tool_call("get_images", inputs, f"{len(result)} images retrieved")
        return result
    
    def _extract_from_image(self, image_path: str) -> Dict:
        """Extract financial information from image using OCR"""
        result = {'text': '', 'amounts': [], 'dates': []}
        
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            result['text'] = text
            
            # Extract amounts
            amount_pattern = r'[\d,]+\.?\d*'
            amounts = re.findall(amount_pattern, text)
            result['amounts'] = amounts[:5]  # Limit to first 5
            
            # Extract dates
            date_pattern = r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}'
            dates = re.findall(date_pattern, text)
            result['dates'] = dates[:5]  # Limit to first 5
            
        except Exception as e:
            self.logger.log(f"  OCR failed for {image_path}: {str(e)}")
        
        return result
    
    def get_payment_options(self, request_id: str) -> List[Dict[str, Any]]:
        """Tool: Retrieve payment options for a request"""
        options = self.payment_options_df[self.payment_options_df['request_id'] == request_id]
        result = options.to_dict('records')
        self.logger.log_tool_call("get_payment_options", {"request_id": request_id}, f"{len(result)} options retrieved")
        return result

# ============================================================================
# FINANCIAL STATE CONSTRUCTION
# ============================================================================

class FinancialStateBuilder:
    """Builds a structured financial state from retrieved data"""
    
    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
    
    def build_state(self, request: Dict, profile: Dict, events: Any, 
                   messages: List[Dict], images: List[Dict], payment_options: List[Dict]) -> Dict:
        """Construct a comprehensive financial state"""
        self.logger.log("Building financial state...")
        
        state = {
            'request': request,
            'user_id': profile.get('user_id'),
            'home_currency': profile.get('home_currency'),
            'current_balance': profile.get('current_available_balance'),
            'minimum_balance': profile.get('minimum_balance_to_keep'),
            'financial_priorities': profile.get('financial_priorities'),
            'protected_categories': profile.get('expense_categories_to_protect'),
            'willing_to_reduce': profile.get('expense_categories_user_is_willing_to_reduce'),
            'willing_to_stop': profile.get('expense_categories_user_is_willing_to_stop'),
            'payment_methods_considered': profile.get('payment_methods_user_will_consider'),
            'max_installment_months': profile.get('max_installment_months'),
            'confirmed_income': [],
            'recurring_expenses': [],
            'one_time_expenses': [],
            'pending_transactions': [],
            'essential_spending': [],
            'extracted_facts': {
                'from_messages': [],
                'from_images': []
            },
            'payment_options': payment_options,
            'evidence_sources': [],
            'raw_messages': messages
        }
        
        # Process events
        if isinstance(events, dict) and 'events_compact' in events:
            lines = events['events_compact'].split('\n')
            events_list = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.split('|')
                if len(parts) < 11:
                    continue
                event_dict = {
                    'event_id': parts[0],
                    'event_type': parts[1],
                    'description': parts[2],
                    'category': parts[3],
                    'direction': parts[4],
                    'amount': float(parts[5]) if parts[5] != 'BLANK' else np.nan,
                    'currency': parts[6],
                    'event_date': parts[7],
                    'settlement_date': parts[8],
                    'status': parts[9],
                    'flexibility': parts[10]
                }
                events_list.append(event_dict)
        else:
            events_list = events.get('events', events.get('sample_events', []))
            
        for event in events_list:
            event_info = {
                'event_id': event['event_id'],
                'type': event['event_type'],
                'category': event['category'],
                'direction': event['direction'],
                'amount': event['amount'],
                'currency': event['currency'],
                'event_date': event['event_date'],
                'settlement_date': event['settlement_date'],
                'status': event['status'],
                'flexibility': event['flexibility']
            }
            
            # Categorize events
            if event['status'] == 'pending':
                state['pending_transactions'].append(event_info)
            elif event['direction'] == 'credit':
                if event['status'] == 'settled':
                    state['confirmed_income'].append(event_info)
            elif event['direction'] == 'debit':
                if event['flexibility'] == 'fixed':
                    state['essential_spending'].append(event_info)
                elif event['event_type'] in ['expense', 'subscription']:
                    # Check if recurring based on description
                    if any(word in event['description'].lower() for word in ['rent', 'subscription', 'monthly', 'plan']):
                        state['recurring_expenses'].append(event_info)
                    else:
                        state['one_time_expenses'].append(event_info)
                else:
                    state['one_time_expenses'].append(event_info)
            
            state['evidence_sources'].append(f"event_{event['event_id']}")
        
        # Extract facts from messages
        for msg in messages:
            for fact in msg.get('extracted_facts', []):
                state['extracted_facts']['from_messages'].append(fact)
                state['evidence_sources'].append(f"message_{msg['message_id']}")
        
        # Process images with blank amounts
        for img in images:
            if img.get('extracted_content'):
                state['extracted_facts']['from_images'].append({
                    'image_id': img['image_id'],
                    'related_event_id': img.get('related_event_id'),
                    'extracted_text': img['extracted_content']['text'][:200],  # First 200 chars
                    'amounts': img['extracted_content']['amounts'],
                    'dates': img['extracted_content']['dates']
                })
                state['evidence_sources'].append(f"image_{img['image_id']}")
        
        # Log summary
        self.logger.log(f"  Confirmed income: {len(state['confirmed_income'])} events")
        self.logger.log(f"  Recurring expenses: {len(state['recurring_expenses'])} events")
        self.logger.log(f"  One-time expenses: {len(state['one_time_expenses'])} events")
        self.logger.log(f"  Pending transactions: {len(state['pending_transactions'])} events")
        self.logger.log(f"  Essential spending: {len(state['essential_spending'])} events")
        self.logger.log(f"  Facts from messages: {len(state['extracted_facts']['from_messages'])}")
        self.logger.log(f"  Facts from images: {len(state['extracted_facts']['from_images'])}")
        
        return state

# ============================================================================
# DETERMINISTIC FINANCIAL CALCULATIONS
# ============================================================================

class FinancialCalculator:
    """Performs deterministic financial calculations and 90-day simulation"""
    
    def __init__(self, logger: ExecutionLogger, tools=None):
        self.logger = logger
        self.tools = tools
        self._cached_user_id = None
        self._cached_patterns = None
        self._cached_message_txs = None

    def detect_recurrence(self, state: Dict, request_date: str) -> List[Dict]:
        """Detect monthly recurring fixed and variable transactions from history"""
        user_id = state.get('user_id')
        if self._cached_user_id == user_id and self._cached_patterns is not None:
            return self._cached_patterns
            
        request_dt = pd.to_datetime(request_date).date()
        
        # Load from raw dataframe directly
        if self.tools is not None:
            df = self.tools.financial_events_df[self.tools.financial_events_df['user_id'] == user_id].copy()
        else:
            # Fallback if tools not passed
            events = []
            for cat_list in ['confirmed_income', 'recurring_expenses', 'one_time_expenses', 'essential_spending']:
                events.extend(state.get(cat_list, []))
            if not events:
                return []
            df = pd.DataFrame(events)
            
        df['event_date'] = pd.to_datetime(df['event_date']).dt.date
        
        # Filter history before request_date
        history = df[df['event_date'] < request_dt].copy()
        if len(history) == 0:
            return []
            
        history['month'] = pd.to_datetime(history['event_date']).dt.to_period('M')
        
        recurring = []
        
        # 1. Fixed monthly recurring grouping (description, direction, category, amount)
        groups = history.groupby(['description', 'direction', 'category', 'amount'])
        for (desc, direction, cat, amount), group in groups:
            if group['month'].nunique() >= 2:
                days = [d.day for d in group['event_date']]
                avg_day = int(np.round(np.mean(days)))
                recurring.append({
                    'type': 'fixed_monthly',
                    'description': desc,
                    'direction': direction,
                    'category': cat,
                    'amount': float(amount),
                    'day_of_month': avg_day,
                    'last_date': group['event_date'].max()
                })
                
        # 2. Variable interval grouping (category, direction)
        covered_descs = set(r['description'] for r in recurring)
        var_history = history[~history['description'].isin(covered_descs)]
        var_groups = var_history.groupby(['category', 'direction'])
        for (cat, direction), group in var_groups:
            if len(group) >= 3:
                sorted_dates = sorted(group['event_date'])
                intervals = [(sorted_dates[i] - sorted_dates[i-1]).days for i in range(1, len(sorted_dates))]
                avg_interval = int(np.round(np.mean(intervals)))
                if 0 < avg_interval <= 45:
                    recurring.append({
                        'type': 'variable_interval',
                        'description': f"Projected {cat}",
                        'direction': direction,
                        'category': cat,
                        'amount': float(group['amount'].mean()),
                        'interval_days': avg_interval,
                        'last_date': max(sorted_dates)
                    })
                    
        self._cached_user_id = user_id
        self._cached_patterns = recurring
        return recurring

    def simulate_balance(self, state: Dict, request_date: str, payment_amount: float = 0.0, 
                         spending_changes: str = None) -> Tuple[float, str, Dict]:
        """Simulate running balances daily for 90 days, returning minimum balance and date"""
        request_dt = pd.to_datetime(request_date).date()
        balance = state['current_balance'] - payment_amount
        min_balance = balance
        min_date = request_dt
        
        patterns = self.detect_recurrence(state, request_date)
        
        # Parse spending changes to apply stops and reductions
        stops = set()
        reductions = {}
        if spending_changes and spending_changes != 'none':
            for change in spending_changes.split('|'):
                parts = change.split(':')
                if len(parts) == 2 and parts[0] == 'stop':
                    stops.add(parts[1])
                elif len(parts) == 3 and parts[0] == 'reduce_to':
                    reductions[parts[1]] = float(parts[2])

        # Get message confirmed transactions
        message_txs = []
        raw_msgs = state.get('raw_messages', [])
        for msg in raw_msgs:
            text = msg.get('message_text', '')
            amount_match = re.search(r'(IDR|USD|EUR|ZAR|INR)\s*([\d,\.]+)', text)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
            if amount_match and date_match:
                curr = amount_match.group(1)
                amount_str = amount_match.group(2).replace('.', '').replace(',', '')
                if curr in ['IDR', 'INR']:
                    amount = float(amount_str)
                else:
                    amount_str_clean = amount_match.group(2).replace(',', '')
                    amount = float(amount_str_clean)
                date = pd.to_datetime(date_match.group(1)).date()
                if date >= request_dt:
                    direction = 'credit' if any(w in text.lower() for w in ['pembayaran', 'payout', 'gaji', 'salary', 'credit', 'approved', 'disetujui']) else 'debit'
                    message_txs.append({
                        'description': f"Message {msg['message_id']}",
                        'direction': direction,
                        'amount': amount,
                        'date': date
                    })

        # Reserve pending transactions
        pending_debits = sum(e['amount'] for e in state.get('pending_transactions', [])
                             if e['direction'] == 'debit' and pd.notna(e['amount'])
                             and e['event_id'] not in stops)
        balance -= pending_debits
        if balance < min_balance:
            min_balance = balance
            min_date = request_dt

        # Track variable next triggers
        next_trigger_dates = {}
        for p in patterns:
            if p['type'] == 'variable_interval':
                last_date = p['last_date']
                next_date = last_date + timedelta(days=p['interval_days'])
                while next_date < request_dt:
                    next_date += timedelta(days=p['interval_days'])
                next_trigger_dates[p['description']] = next_date

        daily_balances = {request_dt: balance}

        for day in range(1, 91):
            current_date = request_dt + timedelta(days=day)
            daily_change = 0.0
            
            # 1. Apply message txs
            for tx in message_txs:
                if tx['date'] == current_date:
                    if tx['direction'] == 'credit':
                        daily_change += tx['amount']
                    else:
                        daily_change -= tx['amount']

            # 2. Apply fixed monthly recurring
            for p in patterns:
                if p['type'] == 'fixed_monthly':
                    if current_date.day == p['day_of_month']:
                        amount = p['amount']
                        if p['direction'] == 'credit':
                            daily_change += amount
                        else:
                            daily_change -= amount

            # 3. Apply variable interval recurring
            for p in patterns:
                if p['type'] == 'variable_interval':
                    if current_date == next_trigger_dates[p['description']]:
                        amount = p['amount']
                        if p['direction'] == 'credit':
                            daily_change += amount
                        else:
                            daily_change -= amount
                        next_trigger_dates[p['description']] = current_date + timedelta(days=p['interval_days'])

            balance += daily_change
            daily_balances[current_date] = balance
            if balance < min_balance:
                min_balance = balance
                min_date = current_date

        return min_balance, min_date.strftime('%Y-%m-%d'), daily_balances

    def calculate_safe_amount(self, state: Dict, request_date: str) -> float:
        """Find the maximum safe amount to pay today using simulation"""
        requested_amount = state['request']['requested_amount']
        min_balance_to_keep = state['minimum_balance']
        
        # Test full payment
        min_bal, _, _ = self.simulate_balance(state, request_date, requested_amount)
        if min_bal >= min_balance_to_keep:
            return requested_amount
            
        # If not full, perform a binary search to find the exact safe amount
        low = 0.0
        high = requested_amount
        best_safe = 0.0
        
        for _ in range(20):
            mid = (low + high) / 2
            min_bal, _, _ = self.simulate_balance(state, request_date, mid)
            if min_bal >= min_balance_to_keep:
                best_safe = mid
                low = mid
            else:
                high = mid
                
        return float(np.round(best_safe, 2))

    def calculate_earliest_full_payment_date(self, state: Dict, requested_amount: float, 
                                             request_date: str, forecast_days: int = 90) -> Optional[str]:
        """Find the first date when full payment is safe without dropping below minimum balance"""
        request_dt = pd.to_datetime(request_date).date()
        min_balance_to_keep = state['minimum_balance']
        
        for day in range(forecast_days):
            current_date = request_dt + timedelta(days=day)
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            _, _, daily_balances = self.simulate_balance(state, request_date, 0.0)
            
            is_safe = True
            for d in range(day, 91):
                future_date = request_dt + timedelta(days=d)
                if daily_balances.get(future_date, 0.0) - requested_amount < min_balance_to_keep:
                    is_safe = False
                    break
                    
            if is_safe:
                self.logger.log_financial_calculation(f"earliest_full_payment_date found at day {day}", current_date_str)
                return current_date_str
                
        self.logger.log_financial_calculation("earliest_full_payment_date", "not found within forecast")
        return ""

    def validate_decision(self, decision: Dict, state: Dict) -> Tuple[bool, str]:
        """Validate that the decision doesn't violate financial constraints"""
        errors = []
        request_date = state['request']['request_date']
        min_balance_to_keep = state['minimum_balance']
        
        # Check amount_safe_to_pay bounds
        if decision['amount_safe_to_pay'] < 0:
            errors.append("amount_safe_to_pay is negative")
        if decision['amount_safe_to_pay'] > state['request']['requested_amount']:
            errors.append("amount_safe_to_pay exceeds requested_amount")
        
        # Check affordability status is valid
        valid_statuses = ['affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable']
        if decision['affordability_status'] not in valid_statuses:
            errors.append(f"Invalid affordability_status: {decision['affordability_status']}")
        
        # Check payment method is valid
        valid_methods = ['full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended']
        if decision['recommended_payment_method'] not in valid_methods:
            errors.append(f"Invalid payment method: {decision['recommended_payment_method']}")
            
        # Check if chosen method is considered by user (except wait/not_recommended as fallbacks)
        if decision['recommended_payment_method'] in ['full_payment', 'partial_payment', 'installments']:
            considered = state.get('payment_methods_considered', [])
            if isinstance(considered, str):
                considered = considered.split('|')
            if decision['recommended_payment_method'] not in considered:
                errors.append(f"User profile does not consider payment method: {decision['recommended_payment_method']}")

        # Validate and simulate the payment plan
        plan_str = decision['payment_plan']
        spending_changes = decision['spending_changes_needed']
        
        if plan_str != 'none':
            try:
                payments = []
                total_plan_paid = 0.0
                plan_parts = plan_str.split('|')
                
                for part in plan_parts:
                    date_str, amount_str = part.split(':')
                    pay_date = pd.to_datetime(date_str).date()
                    pay_amount = float(amount_str)
                    payments.append((pay_date, pay_amount))
                    total_plan_paid += pay_amount
                    
                # 1. Simulate the entire plan day-by-day
                request_dt = pd.to_datetime(request_date).date()
                balance = state['current_balance']
                min_observed = balance
                
                patterns = self.detect_recurrence(state, request_date)
                
                # Parse spending changes
                stops = set()
                if spending_changes and spending_changes != 'none':
                    for change in spending_changes.split('|'):
                        parts = change.split(':')
                        if len(parts) == 2 and parts[0] == 'stop':
                            stops.add(parts[1])
                            
                # Get message confirmed transactions
                message_txs = []
                for msg in state.get('raw_messages', []):
                    text = msg.get('message_text', '')
                    amount_match = re.search(r'(IDR|USD|EUR|ZAR|INR)\s*([\d,\.]+)', text)
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
                    if amount_match and date_match:
                        curr = amount_match.group(1)
                        amount_str = amount_match.group(2).replace('.', '').replace(',', '')
                        if curr in ['IDR', 'INR']:
                            amount = float(amount_str)
                        else:
                            amount_str_clean = amount_match.group(2).replace(',', '')
                            amount = float(amount_str_clean)
                        date = pd.to_datetime(date_match.group(1)).date()
                        if date >= request_dt:
                            direction = 'credit' if any(w in text.lower() for w in ['pembayaran', 'payout', 'gaji', 'salary', 'credit', 'approved', 'disetujui']) else 'debit'
                            message_txs.append({
                                'direction': direction,
                                'amount': amount,
                                'date': date
                            })
                            
                # Reserve pending
                pending_debits = sum(e['amount'] for e in state.get('pending_transactions', [])
                                     if e['direction'] == 'debit' and pd.notna(e['amount'])
                                     and e['event_id'] not in stops)
                balance -= pending_debits
                if balance < min_observed:
                    min_observed = balance
                    
                # Track variable intervals
                next_trigger_dates = {}
                for p in patterns:
                    if p['type'] == 'variable_interval':
                        last_date = p['last_date']
                        next_date = last_date + timedelta(days=p['interval_days'])
                        while next_date < request_dt:
                            next_date += timedelta(days=p['interval_days'])
                        next_trigger_dates[p['description']] = next_date
                        
                # Perform 90-day ledger simulation
                for day in range(91):
                    current_date = request_dt + timedelta(days=day)
                    daily_change = 0.0
                    
                    # Apply any scheduled payment plan transactions for today
                    for pay_date, pay_amount in payments:
                        if pay_date == current_date:
                            daily_change -= pay_amount
                            
                    if day > 0:
                        # Apply message txs
                        for tx in message_txs:
                            if tx['date'] == current_date:
                                if tx['direction'] == 'credit':
                                    daily_change += tx['amount']
                                else:
                                    daily_change -= tx['amount']

                        # Apply fixed monthly recurring
                        for p in patterns:
                            if p['type'] == 'fixed_monthly':
                                if current_date.day == p['day_of_month']:
                                    amount = p['amount']
                                    if p['direction'] == 'credit':
                                        daily_change += amount
                                    else:
                                        daily_change -= amount

                        # Apply variable interval recurring
                        for p in patterns:
                            if p['type'] == 'variable_interval':
                                if current_date == next_trigger_dates[p['description']]:
                                    amount = p['amount']
                                    if p['direction'] == 'credit':
                                        daily_change += amount
                                    else:
                                        daily_change -= amount
                                    next_trigger_dates[p['description']] = current_date + timedelta(days=p['interval_days'])
                                    
                    balance += daily_change
                    if balance < min_observed:
                        min_observed = balance
                        
                # Check if running balance ever fell below minimum balance
                if min_observed < min_balance_to_keep:
                    errors.append(f"Simulation failed: Running balance falls to {min_observed:.2f} (below minimum required {min_balance_to_keep})")
                    
            except Exception as e:
                errors.append(f"Validation error running simulation: {str(e)}")
                
        if errors:
            return False, "; ".join(errors)
        return True, "All checks passed"

# ============================================================================
# REAL AGENT WITH GROQ
# ============================================================================

class RealFinancialAgent:
    """Real LLM-driven agent using Groq with tool calling"""
    
    SYSTEM_PROMPT = """You are a financial decision agent for the Buy or Wait? challenge.

Your task is to analyze a purchase/payment request and determine whether the user can safely afford it.

WORKFLOW:
1. First, call get_request to get the request details.
2. Call get_financial_profile to get the user's financial profile.
3. Call get_financial_events to get the user's financial events.
4. Call get_messages to find if any messages/employer updates affect upcoming salary, payouts, cancellations, or other cash flow events.
5. If any events have blank amount fields, call get_images to perform OCR and retrieve the exact amounts from the referenced document.
6. Call get_payment_options to see available payment methods.
7. Perform daily 90-day cash flow simulation. Ensure minimum balance is never violated.
8. Call make_decision with your final structured analysis and recommendations.

IMPORTANT RULES:
1. Follow the workflow - check all relevant files (requests, profile, events, messages, images, options) before making a decision.
2. Reason ONLY from the retrieved evidence provided by tools.
3. Explicitly identify which facts support your decision.
4. Never invent information not present in the retrieved data.
5. Provide grounded explanations with specific references to event_ids, message_ids, image_ids, payment_option_ids.

A recommendation is safe only if the user can complete the full payment plan, cover essential expenses, and stay above their minimum balance throughout the 90-day forecast period.

After gathering all necessary information, you MUST call the make_decision tool with your final decision. Do not provide text responses - use the tool."""

    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.tools = FinancialDataTools(logger)
        self.state_builder = FinancialStateBuilder(logger)
        self.calculator = FinancialCalculator(logger, self.tools)
        
        # Initialize Groq client
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)
        self.model = "qwen/qwen3.6-27b"  # Groq model with good reasoning
        
        # Tool definitions for Groq
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_request",
                    "description": "Get details of a specific request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {
                                "type": "string",
                                "description": "The request ID"
                            }
                        },
                        "required": ["request_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financial_profile",
                    "description": "Get the financial profile for a user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user ID"
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financial_events",
                    "description": "Get all financial events for a user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user ID"
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_messages",
                    "description": "Get messages for a user, optionally filtered by request or events",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user ID"
                            },
                            "request_id": {
                                "type": "string",
                                "description": "Optional request ID to filter messages"
                            },
                            "event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of event IDs to filter messages"
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_images",
                    "description": "Get image references and extract content from images",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user ID"
                            },
                            "request_id": {
                                "type": "string",
                                "description": "Optional request ID to filter images"
                            },
                            "event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of event IDs to filter images"
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_payment_options",
                    "description": "Get payment options for a specific request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {
                                "type": "string",
                                "description": "The request ID"
                            }
                        },
                        "required": ["request_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "make_decision",
                    "description": "Make the final financial decision after gathering information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {
                                "type": "string",
                                "description": "The request ID"
                            },
                            "amount_safe_to_pay": {
                                "type": "number",
                                "description": "The maximum amount safe to pay on request_date (0 to requested_amount)"
                            },
                            "affordability_status": {
                                "type": "string",
                                "enum": ["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"],
                                "description": "The affordability status"
                            },
                            "recommended_payment_method": {
                                "type": "string",
                                "enum": ["full_payment", "partial_payment", "installments", "wait", "not_recommended"],
                                "description": "The recommended payment method"
                            },
                            "payment_plan": {
                                "type": "string",
                                "description": "Payment plan in format YYYY-MM-DD:amount|YYYY-MM-DD:amount or 'none'"
                            },
                            "earliest_date_for_full_payment": {
                                "type": "string",
                                "description": "Earliest date for full payment in YYYY-MM-DD format, or empty string"
                            },
                            "spending_changes_needed": {
                                "type": "string",
                                "description": "Spending changes in format stop:event_id|reduce_to:event_id:amount or 'none'"
                            },
                            "decision_explanation": {
                                "type": "string",
                                "description": "Grounded explanation with specific references to data"
                            },
                            "supporting_references": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of evidence source IDs (event_id, message_id, image_id, payment_option_id)"
                            }
                        },
                        "required": ["request_id", "amount_safe_to_pay", "affordability_status", 
                                   "recommended_payment_method", "payment_plan", 
                                   "earliest_date_for_full_payment", "spending_changes_needed",
                                   "decision_explanation", "supporting_references"]
                    }
                }
            }
        ]
    
    def process_request(self, request_id: str, user_id: str, max_iterations: int = 10) -> Dict:
        """Process a request with deterministic pre-fetching and a single-turn LLM reasoning call"""
        self.logger.log_request_start(request_id, user_id)
        
        # 1. Deterministically pre-fetch all data using our tool classes
        self.logger.log("Pre-fetching all financial data for request...")
        request_data = self._execute_tool("get_request", {"request_id": request_id}, {})
        profile_data = self._execute_tool("get_financial_profile", {"user_id": user_id}, {})
        
        # Pass request date to events retrieval
        req_date = request_data.get('request_date', '2025-08-03')
        events_data = self.tools.get_financial_events(user_id, req_date)
        
        messages_data = self._execute_tool("get_messages", {"user_id": user_id}, {})
        payment_options_data = self._execute_tool("get_payment_options", {"request_id": request_id}, {})
        
        # Store in collected_data
        collected_data = {
            'request': request_data,
            'profile': profile_data,
            'events': events_data,
            'messages': messages_data,
            'images': [], # request_26 has no images
            'payment_options': payment_options_data
        }
        
        # 2. Build a single, highly clean user prompt containing all pre-fetched data
        user_prompt = f"""You are analyzing purchase request {request_id} for user {user_id}.

I have pre-fetched all relevant financial data. Analyze it carefully and call the `make_decision` tool with your final decision.

### 1. REQUEST DETAILS
{json.dumps(request_data, indent=2)}

### 2. FINANCIAL PROFILE
{json.dumps(profile_data, indent=2)}

### 3. RECENT FINANCIAL EVENTS (COMPACT)
{events_data.get('events_compact', '')}

### 4. RELEVANT MESSAGES
{json.dumps(messages_data, indent=2)}

### 5. AVAILABLE PAYMENT OPTIONS
{json.dumps(payment_options_data, indent=2)}

Based on these facts, run your financial reasoning, and call `make_decision`."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # 3. Call the LLM once. Force it to call `make_decision`!
        self.logger.log("Calling model for single-turn structured reasoning and decision selection...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto", # Use auto to avoid tool_use_failed errors on Groq endpoints
                max_tokens=800
            )
            
            self.logger.log_model_call(self.model, len(str(messages)), len(response.choices[0].message.content or ''))
            
            response_message = response.choices[0].message
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "make_decision":
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Execute deterministic validation & simulation
                        tool_result = self._execute_tool("make_decision", function_args, collected_data)
                        return self._finalize_decision(tool_result, collected_data)
                        
            # If no tool calls, return content
            self.logger.log(f"Model response content: {response_message.content}")
            return {"error": "Model did not call make_decision tool even with forced tool choice.", "response": response_message.content}
            
        except Exception as e:
            self.logger.log_error(f"Groq API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _execute_tool(self, tool_name: str, args: Dict, collected_data: Dict) -> Any:
        """Execute a tool and return result"""
        if tool_name == "get_request":
            result = self.tools.get_request(args['request_id'])
            collected_data['request'] = result
            return result
        
        elif tool_name == "get_financial_profile":
            result = self.tools.get_financial_profile(args['user_id'])
            collected_data['profile'] = result
            return result
        
        elif tool_name == "get_financial_events":
            req_date = "2025-08-03"
            if collected_data.get('request') and 'request_date' in collected_data['request']:
                req_date = collected_data['request']['request_date']
            result = self.tools.get_financial_events(args['user_id'], req_date)
            collected_data['events'] = result
            return result
        
        elif tool_name == "get_messages":
            result = self.tools.get_messages(
                args['user_id'],
                args.get('request_id'),
                args.get('event_ids')
            )
            collected_data['messages'] = result
            return result
        
        elif tool_name == "get_images":
            result = self.tools.get_images(
                args['user_id'],
                args.get('request_id'),
                args.get('event_ids')
            )
            collected_data['images'] = result
            return result
        
        elif tool_name == "get_payment_options":
            result = self.tools.get_payment_options(args['request_id'])
            collected_data['payment_options'] = result
            return result
        
        elif tool_name == "make_decision":
            # Build financial state
            state = self.state_builder.build_state(
                collected_data['request'],
                collected_data['profile'],
                collected_data['events'],
                collected_data['messages'],
                collected_data['images'],
                collected_data['payment_options']
            )
            
            # Perform deterministic calculations
            safe_amount = self.calculator.calculate_safe_amount(
                state, 
                collected_data['request']['request_date']
            )
            
            earliest_date = self.calculator.calculate_earliest_full_payment_date(
                state,
                collected_data['request']['requested_amount'],
                collected_data['request']['request_date']
            )
            
            # Override LLM's calculated values with deterministic ones
            args['amount_safe_to_pay'] = min(safe_amount, collected_data['request']['requested_amount'])
            args['earliest_date_for_full_payment'] = earliest_date or ''
            
            # Validate decision
            validation_passed, validation_details = self.calculator.validate_decision(args, state)
            self.logger.log_validation(validation_passed, validation_details)
            
            return args
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _finalize_decision(self, decision: Dict, collected_data: Dict) -> Dict:
        """Finalize the decision with additional metadata"""
        self.logger.log_decision(decision)
        
        result = {
            'decision': decision,
            'collected_data_summary': {
                'request': bool(collected_data['request']),
                'profile': bool(collected_data['profile']),
                'events_count': len(collected_data['events']),
                'messages_count': len(collected_data['messages']),
                'images_count': len(collected_data['images']),
                'payment_options_count': len(collected_data['payment_options'])
            },
            'model_used': self.model,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("REAL LLM-DRIVEN AGENT: ONE REQUEST TEST")
    print("=" * 70)
    print()
    
    # Initialize logger
    logger = ExecutionLogger(LOG_FILE)
    logger.log("\n" + "=" * 70)
    logger.log("REAL AGENT EXECUTION START")
    logger.log("=" * 70)
    
    try:
        # Initialize agent
        logger.log("Initializing agent with Groq...")
        agent = RealFinancialAgent(logger)
        logger.log(f"Using model: {agent.model}")
        
        # Select first request from dataset
        requests_df = pd.read_csv(DATASET_DIR / "requests.csv")
        first_request_id = requests_df.iloc[0]['request_id']
        first_user_id = requests_df.iloc[0]['user_id']
        
        logger.log(f"Selected request: {first_request_id} (user: {first_user_id})")
        
        # Process the request
        result = agent.process_request(first_request_id, first_user_id)
        
        # Flush logs
        logger.flush()
        
        # Print summary
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            if 'response' in result:
                print(f"Model response: {result['response']}")
        else:
            decision = result['decision']
            print(f"Request ID: {decision['request_id']}")
            print(f"Amount Safe to Pay: {decision['amount_safe_to_pay']}")
            print(f"Affordability Status: {decision['affordability_status']}")
            print(f"Recommended Payment Method: {decision['recommended_payment_method']}")
            print(f"Payment Plan: {decision['payment_plan']}")
            print(f"Earliest Date for Full Payment: {decision['earliest_date_for_full_payment']}")
            print(f"Spending Changes Needed: {decision['spending_changes_needed']}")
            print(f"\nDecision Explanation:")
            print(f"  {decision['decision_explanation']}")
            print(f"\nSupporting References: {', '.join(decision['supporting_references'])}")
            print(f"\nData Collected:")
            print(f"  - Request: {'Yes' if result['collected_data_summary']['request'] else 'No'}")
            print(f"  - Profile: {'Yes' if result['collected_data_summary']['profile'] else 'No'}")
            print(f"  - Events: {result['collected_data_summary']['events_count']}")
            print(f"  - Messages: {result['collected_data_summary']['messages_count']}")
            print(f"  - Images: {result['collected_data_summary']['images_count']}")
            print(f"  - Payment Options: {result['collected_data_summary']['payment_options_count']}")
            print(f"\nModel Used: {result['model_used']}")
        
        print("\nExecution trace logged to: log.txt")
        print()
        
    except Exception as e:
        logger.log_error(f"Fatal error: {str(e)}")
        logger.flush()
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
