#!/usr/bin/env python3
"""
Agent-style prototype for Buy or Wait? challenge.
Processes ONE request to demonstrate agent workflow with tool orchestration.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional

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
    
    def log_tool_call(self, tool_name: str, inputs: Dict, result_count: int = None):
        """Log a tool invocation"""
        self.log(f"TOOL CALL: {tool_name}")
        self.log(f"  Inputs: {json.dumps(inputs, default=str)}")
        if result_count is not None:
            self.log(f"  Results: {result_count} record(s) retrieved")
    
    def log_model_call(self, prompt: str, response: str = None):
        """Log a model invocation"""
        self.log("MODEL CALL")
        self.log(f"  Prompt length: {len(prompt)} characters")
        if response:
            self.log(f"  Response length: {len(response)} characters")
    
    def log_decision(self, decision: Dict):
        """Log final decision"""
        self.log("FINAL DECISION")
        self.log(f"  {json.dumps(decision, default=str, indent=2)}")
    
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
    """Tools for retrieving financial data"""
    
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
    
    def get_financial_profile(self, user_id: str) -> Dict[str, Any]:
        """Tool: Retrieve financial profile for a user"""
        self.logger.log_tool_call("get_financial_profile", {"user_id": user_id})
        
        profile = self.financial_profiles_df[self.financial_profiles_df['user_id'] == user_id]
        
        if len(profile) == 0:
            self.logger.log("  WARNING: No profile found")
            return {}
        
        result = profile.iloc[0].to_dict()
        self.logger.log_tool_call("get_financial_profile", {"user_id": user_id}, result_count=1)
        return result
    
    def get_financial_events(self, user_id: str) -> List[Dict[str, Any]]:
        """Tool: Retrieve financial events for a user"""
        self.logger.log_tool_call("get_financial_events", {"user_id": user_id})
        
        events = self.financial_events_df[self.financial_events_df['user_id'] == user_id]
        result = events.to_dict('records')
        
        self.logger.log_tool_call("get_financial_events", {"user_id": user_id}, result_count=len(result))
        return result
    
    def get_messages(self, user_id: str, request_id: str = None, event_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Tool: Retrieve messages for a user/request/events"""
        inputs = {"user_id": user_id}
        if request_id:
            inputs["request_id"] = request_id
        if event_ids:
            inputs["event_ids"] = event_ids
        
        self.logger.log_tool_call("get_messages", inputs)
        
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
        self.logger.log_tool_call("get_messages", inputs, result_count=len(result))
        return result
    
    def get_images(self, user_id: str, request_id: str = None, event_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Tool: Retrieve image references for a user/request/events"""
        inputs = {"user_id": user_id}
        if request_id:
            inputs["request_id"] = request_id
        if event_ids:
            inputs["event_ids"] = event_ids
        
        self.logger.log_tool_call("get_images", inputs)
        
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
        
        # Add file paths
        for img in result:
            img['file_path'] = str(DATASET_DIR / "media" / "images" / f"{img['image_id']}.png")
        
        self.logger.log_tool_call("get_images", inputs, result_count=len(result))
        return result
    
    def get_payment_options(self, request_id: str) -> List[Dict[str, Any]]:
        """Tool: Retrieve payment options for a request"""
        self.logger.log_tool_call("get_payment_options", {"request_id": request_id})
        
        options = self.payment_options_df[self.payment_options_df['request_id'] == request_id]
        result = options.to_dict('records')
        
        self.logger.log_tool_call("get_payment_options", {"request_id": request_id}, result_count=len(result))
        return result
    
    def get_request(self, request_id: str) -> Dict[str, Any]:
        """Tool: Retrieve a specific request"""
        self.logger.log_tool_call("get_request", {"request_id": request_id})
        
        request = self.requests_df[self.requests_df['request_id'] == request_id]
        
        if len(request) == 0:
            self.logger.log("  WARNING: No request found")
            return {}
        
        result = request.iloc[0].to_dict()
        self.logger.log_tool_call("get_request", {"request_id": request_id}, result_count=1)
        return result

# ============================================================================
# MODEL INTERFACE (ISOLATED)
# ============================================================================

class ModelInterface:
    """Isolated interface for model calls - can be configured with any provider"""
    
    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.api_key = os.environ.get('MODEL_API_KEY')
        self.model_provider = os.environ.get('MODEL_PROVIDER', 'not_configured')
    
    def is_configured(self) -> bool:
        """Check if a model is properly configured"""
        return self.api_key is not None and self.model_provider != 'not_configured'
    
    def call_model(self, system_prompt: str, user_prompt: str, context: Dict) -> str:
        """
        Call the model with system prompt, user prompt, and context.
        Returns the model's response.
        
        This is a placeholder - implement actual API call based on provider.
        """
        self.logger.log_model_call(system_prompt + "\n\n" + user_prompt)
        
        if not self.is_configured():
            self.logger.log("  WARNING: No model configured - returning placeholder")
            return "MODEL_NOT_CONFIGURED: Please set MODEL_API_KEY and MODEL_PROVIDER environment variables"
        
        # TODO: Implement actual API call based on provider
        # Example for OpenAI:
        # import openai
        # client = openai.OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(...)
        # return response.choices[0].message.content
        
        self.logger.log("  ERROR: Model interface not implemented for this provider")
        return "MODEL_NOT_IMPLEMENTED"

# ============================================================================
# AGENT
# ============================================================================

class FinancialAgent:
    """Agent that orchestrates tools and model for financial decisions"""
    
    SYSTEM_PROMPT = """You are a financial decision agent for the Buy or Wait? challenge.

Your task is to analyze a purchase/payment request and determine whether the user can safely afford it.

You must:
1. Reason ONLY from the retrieved evidence provided in the context
2. Explicitly identify which facts support your decision
3. Never invent information not present in the retrieved data
4. If information is missing, identify what is missing rather than guessing
5. Provide grounded explanations with specific references to the data

The user's financial safety depends on:
- Current available balance
- Minimum balance they want to maintain
- Recurring expenses and commitments
- Pending payments
- Confirmed income
- Payment options available
- Their financial priorities and preferences

A recommendation is safe only if the user can complete the full payment plan, cover essential expenses, and stay above their minimum balance throughout the forecast period."""

    def __init__(self, logger: ExecutionLogger):
        self.logger = logger
        self.tools = FinancialDataTools(logger)
        self.model = ModelInterface(logger)
    
    def process_request(self, request_id: str) -> Dict[str, Any]:
        """Process a single request through the agent workflow"""
        self.logger.log("=" * 70)
        self.logger.log(f"PROCESSING REQUEST: {request_id}")
        self.logger.log("=" * 70)
        
        # Step 1: Get the request
        self.logger.log("\n[STEP 1] Retrieve request details")
        request = self.tools.get_request(request_id)
        if not request:
            return {"error": "Request not found"}
        user_id = request['user_id']
        self.logger.log(f"  User: {user_id}")
        self.logger.log(f"  Request type: {request['request_type']}")
        self.logger.log(f"  Requested amount: {request['requested_amount']}")
        
        # Step 2: Get financial profile
        self.logger.log("\n[STEP 2] Retrieve financial profile")
        profile = self.tools.get_financial_profile(user_id)
        
        # Step 3: Get financial events
        self.logger.log("\n[STEP 3] Retrieve financial events")
        events = self.tools.get_financial_events(user_id)
        event_ids = [e['event_id'] for e in events]
        
        # Step 4: Get messages
        self.logger.log("\n[STEP 4] Retrieve messages")
        messages = self.tools.get_messages(user_id, request_id, event_ids)
        
        # Step 5: Get images
        self.logger.log("\n[STEP 5] Retrieve image references")
        images = self.tools.get_images(user_id, request_id, event_ids)
        
        # Step 6: Get payment options
        self.logger.log("\n[STEP 6] Retrieve payment options")
        payment_options = self.tools.get_payment_options(request_id)
        
        # Step 7: Prepare context for model
        self.logger.log("\n[STEP 7] Prepare context for model decision")
        context = {
            "request": request,
            "profile": profile,
            "events_count": len(events),
            "messages_count": len(messages),
            "images_count": len(images),
            "payment_options_count": len(payment_options),
            "events": events[:10],  # Include first 10 events as sample
            "payment_options": payment_options
        }
        
        # Step 8: Model decision point
        self.logger.log("\n[STEP 8] Model decision point")
        user_prompt = f"""Analyze this request and provide a financial decision.

Request:
- ID: {request['request_id']}
- Type: {request['request_type']}
- Amount: {request['requested_amount']}
- Date: {request['request_date']}
- Deadline: {request['desired_completion_date']}
- Allows partial payment: {request['allows_partial_payment']}
- Description: {request['request_text']}

User Profile:
- Home currency: {profile.get('home_currency')}
- Current balance: {profile.get('current_available_balance')}
- Minimum balance to keep: {profile.get('minimum_balance_to_keep')}
- Priorities: {profile.get('financial_priorities')}
- Protected categories: {profile.get('expense_categories_to_protect')}
- Payment methods considered: {profile.get('payment_methods_user_will_consider')}

Context:
- Financial events: {len(events)} records
- Messages: {len(messages)} records
- Images: {len(images)} references
- Payment options: {len(payment_options)} options

Please provide:
1. A decision (affordable_now, affordable_with_plan, affordable_later, not_affordable)
2. Recommended payment method
3. Amount safe to pay
4. A grounded explanation with specific references to the data
5. Any missing information needed for a complete decision"""

        model_response = self.model.call_model(self.SYSTEM_PROMPT, user_prompt, context)
        
        # Step 9: Compile result
        result = {
            "request_id": request_id,
            "user_id": user_id,
            "tools_invoked": [
                "get_request",
                "get_financial_profile",
                "get_financial_events",
                "get_messages",
                "get_images",
                "get_payment_options"
            ],
            "information_retrieved": {
                "profile": bool(profile),
                "events_count": len(events),
                "messages_count": len(messages),
                "images_count": len(images),
                "payment_options_count": len(payment_options)
            },
            "model_configured": self.model.is_configured(),
            "model_response": model_response,
            "supporting_references": {
                "profile_keys": list(profile.keys()) if profile else [],
                "sample_event_ids": event_ids[:5],
                "message_ids": [m['message_id'] for m in messages[:3]],
                "image_ids": [img['image_id'] for img in images],
                "payment_option_ids": [opt['payment_option_id'] for opt in payment_options]
            }
        }
        
        self.logger.log("\n[STEP 9] Compile result")
        self.logger.log_decision(result)
        
        return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("AGENT PROTOTYPE: ONE REQUEST DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize logger
    logger = ExecutionLogger(LOG_FILE)
    logger.log("\n" + "=" * 70)
    logger.log("AGENT PROTOTYPE EXECUTION START")
    logger.log("=" * 70)
    
    # Initialize agent
    logger.log("\nInitializing agent...")
    agent = FinancialAgent(logger)
    
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
    print(f"User ID: {result['user_id']}")
    print(f"Request ID: {result['request_id']}")
    print(f"\nTools Invoked: {', '.join(result['tools_invoked'])}")
    print(f"\nInformation Retrieved:")
    print(f"  - Profile: {'Yes' if result['information_retrieved']['profile'] else 'No'}")
    print(f"  - Events: {result['information_retrieved']['events_count']}")
    print(f"  - Messages: {result['information_retrieved']['messages_count']}")
    print(f"  - Images: {result['information_retrieved']['images_count']}")
    print(f"  - Payment Options: {result['information_retrieved']['payment_options_count']}")
    print(f"\nModel Configured: {result['model_configured']}")
    if not result['model_configured']:
        print("  ⚠️  No model API configured - set MODEL_API_KEY and MODEL_PROVIDER environment variables")
    print(f"\nSupporting References:")
    print(f"  - Profile keys: {result['supporting_references']['profile_keys']}")
    print(f"  - Sample event IDs: {result['supporting_references']['sample_event_ids']}")
    print(f"  - Message IDs: {result['supporting_references']['message_ids']}")
    print(f"  - Image IDs: {result['supporting_references']['image_ids']}")
    print(f"  - Payment option IDs: {result['supporting_references']['payment_option_ids']}")
    print()
    print("Execution trace logged to: log.txt")
    print()

if __name__ == "__main__":
    main()
