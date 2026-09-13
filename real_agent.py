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
    
    def get_financial_events(self, user_id: str) -> Dict[str, Any]:
        """Tool: Retrieve financial events for a user (summarized if many)"""
        events = self.financial_events_df[self.financial_events_df['user_id'] == user_id]
        all_events = events.to_dict('records')
        
        # If too many events, return summary instead of full data
        if len(all_events) > 20:
            summary = {
                'total_events': len(all_events),
                'summary': 'Too many events to return all. Here are key statistics:',
                'by_type': events['event_type'].value_counts().to_dict(),
                'by_status': events['status'].value_counts().to_dict(),
                'by_direction': events['direction'].value_counts().to_dict(),
                'sample_events': all_events[:10]  # Return first 10 as sample
            }
            self.logger.log_tool_call("get_financial_events", {"user_id": user_id}, f"{len(all_events)} events (summarized)")
            return summary
        else:
            self.logger.log_tool_call("get_financial_events", {"user_id": user_id}, f"{len(all_events)} events retrieved")
            return {'events': all_events, 'total_events': len(all_events)}
    
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
            'evidence_sources': []
        }
        
        # Process events
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
    """Performs deterministic financial calculations"""
    
    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
    
    def calculate_safe_amount(self, state: Dict, request_date: str) -> float:
        """Calculate the maximum safe amount to pay on request_date"""
        balance = state['current_balance']
        minimum = state['minimum_balance']
        
        # Reserve pending debits
        pending_debits = sum(e['amount'] for e in state['pending_transactions'] 
                           if e['direction'] == 'debit' and pd.notna(e['amount']))
        
        # Reserve essential spending near request date
        request_dt = pd.to_datetime(request_date)
        essential_near = sum(e['amount'] for e in state['essential_spending']
                            if pd.notna(e['amount']) and 
                            abs((pd.to_datetime(e['event_date']) - request_dt).days) <= 7)
        
        safe_amount = balance - minimum - pending_debits - essential_near
        safe_amount = max(0, safe_amount)
        
        self.logger.log_financial_calculation(
            f"safe_amount = {balance} - {minimum} - {pending_debits} - {essential_near}",
            safe_amount
        )
        
        return safe_amount
    
    def calculate_earliest_full_payment_date(self, state: Dict, requested_amount: float, 
                                            request_date: str, forecast_days: int = 90) -> Optional[str]:
        """Calculate the earliest date when full payment is safe"""
        balance = state['current_balance']
        minimum = state['minimum_balance']
        request_dt = pd.to_datetime(request_date)
        
        # Simulate daily balance over forecast period
        for day in range(forecast_days):
            current_date = request_dt + timedelta(days=day)
            
            # Add confirmed income on settlement date
            daily_balance = balance
            for income in state['confirmed_income']:
                if pd.to_datetime(income['settlement_date']) == current_date:
                    daily_balance += income['amount']
            
            # Subtract pending debits on settlement date
            for debit in state['pending_transactions']:
                if pd.to_datetime(debit['settlement_date']) == current_date and debit['direction'] == 'debit':
                    daily_balance -= debit['amount']
            
            # Check if we can afford the payment
            if daily_balance - minimum >= requested_amount:
                result_date = current_date.strftime('%Y-%m-%d')
                self.logger.log_financial_calculation(
                    f"earliest_full_payment_date found at day {day}",
                    result_date
                )
                return result_date
        
        self.logger.log_financial_calculation("earliest_full_payment_date", "not found within forecast")
        return None
    
    def validate_decision(self, decision: Dict, state: Dict) -> Tuple[bool, str]:
        """Validate that the decision doesn't violate financial constraints"""
        errors = []
        
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
        
        # Check payment plan format if provided
        if decision['payment_plan'] != 'none':
            try:
                payments = decision['payment_plan'].split('|')
                for payment in payments:
                    date, amount = payment.split(':')
                    # Validate date format
                    pd.to_datetime(date)
                    # Validate amount is numeric
                    float(amount)
            except Exception as e:
                errors.append(f"Invalid payment_plan format: {str(e)}")
        
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
1. First, call get_request to get the request details
2. Then call get_financial_profile to get the user's financial profile
3. Then call get_financial_events to get the user's financial events
4. Then call get_payment_options to see available payment methods
5. Finally, call make_decision with your analysis

IMPORTANT RULES:
1. Follow the workflow above - call tools in order
2. Reason ONLY from the retrieved evidence provided by tools
3. Explicitly identify which facts support your decision
4. Never invent information not present in the retrieved data
5. Provide grounded explanations with specific references to event_ids, message_ids, image_ids

A recommendation is safe only if the user can complete the full payment plan, cover essential expenses, and stay above their minimum balance throughout the forecast period.

After gathering all necessary information (request, profile, events, payment options), you MUST call the make_decision tool with your final decision. Do not provide text responses - use the tool."""

    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.tools = FinancialDataTools(logger)
        self.state_builder = FinancialStateBuilder(logger)
        self.calculator = FinancialCalculator(logger)
        
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
    
    def process_request(self, request_id: str, max_iterations: int = 10) -> Dict:
        """Process a request with LLM-driven tool calling"""
        self.logger.log_request_start(request_id, "unknown")
        
        # Initial user message
        user_message = f"Please analyze request {request_id} and determine whether the user can safely afford it. Use the available tools to gather the necessary information."
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        collected_data = {
            'request': None,
            'profile': None,
            'events': [],
            'messages': [],
            'images': [],
            'payment_options': []
        }
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            self.logger.log(f"Iteration {iteration}")
            
            try:
                # Call Groq with tools
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    max_tokens=500  # Limit output to stay within rate limits
                )
                
                self.logger.log_model_call(self.model, len(str(messages)), len(response.choices[0].message.content or ''))
                
                response_message = response.choices[0].message
                
                # Check if model wants to call tools
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Execute tool
                        tool_result = self._execute_tool(function_name, function_args, collected_data)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(tool_result)
                        })
                        
                        # Check if this is the final decision
                        if function_name == "make_decision":
                            return self._finalize_decision(tool_result, collected_data)
                else:
                    # Model provided a direct response without tools
                    self.logger.log("Model provided direct response (no tool calls)")
                    self.logger.log(f"Response: {response_message.content}")
                    return {"error": "Model did not call make_decision tool", "response": response_message.content}
                    
            except Exception as e:
                self.logger.log_error(f"Groq API call failed: {str(e)}")
                return {"error": str(e)}
        
        return {"error": "Max iterations exceeded without decision"}
    
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
            result = self.tools.get_financial_events(args['user_id'])
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
        result = agent.process_request(first_request_id)
        
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
