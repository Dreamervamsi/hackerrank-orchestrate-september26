import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time

# Load env variables
load_dotenv()

# Import local modules
from code.data_manager import DataManager
from code.simulator import FinancialSimulator
from code.agent import StructuredFinancialAgent

# Paths
DATASET_DIR = Path("dataset")
LOG_FILE = Path("log.txt")
OUTPUT_FILE = Path("dataset/output.csv")
REPORT_FILE = Path("code/evaluation/usage_report.md")

class BatchPipeline:
    """Production orchestration batch pipeline for all 250 requests."""
    
    def __init__(self):
        self.total_model_calls = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_duration_sec = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.validation_passed_count = 0
        
    def log(self, message: str):
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        entry = f"[{timestamp}] {message}"
        print(entry)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(entry + '\n')

    def run_preflight_checks(self):
        self.log("=" * 70)
        self.log("RUNNING PRE-FLIGHT CHECKS")
        self.log("=" * 70)
        
        # 1. Verify dataset files exist
        required_files = [
            "requests.csv", "financial_profiles.csv", "financial_events.csv",
            "messages.csv", "images.csv", "request_payment_options.csv",
            "exchange_rates.csv"
        ]
        for f in required_files:
            p = DATASET_DIR / f
            if not p.exists():
                raise FileNotFoundError(f"Critical pre-flight check failed: {f} not found in dataset folder!")
            self.log(f"  Verified dataset file exists: {f}")
            
        # 2. Check requests count
        requests_df = pd.read_csv(DATASET_DIR / "requests.csv")
        req_count = len(requests_df)
        self.log(f"  Checked request count: {req_count} requests")
        if req_count != 250:
            self.log(f"Warning: requests.csv contains {req_count} rows instead of expected 250!")
            
        # 3. Check Groq API Key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Critical pre-flight check failed: GROQ_API_KEY environment variable is missing!")
        self.log("  Checked Groq API key exists")
        
        self.log("All pre-flight checks passed successfully.")
        print()

    def process_all_requests(self):
        # 1. Initialize our modules
        data_manager = DataManager()
        simulator = FinancialSimulator(data_manager)
        agent = StructuredFinancialAgent(data_manager, simulator)
        
        requests_df = data_manager.requests_df
        results = []
        
        # Clear existing output file or verify we can write to it
        if OUTPUT_FILE.exists():
            self.log(f"Removing old output file: {OUTPUT_FILE}")
            OUTPUT_FILE.unlink()
            
        self.log("=" * 70)
        self.log("STARTING BATCH PROCESSING LOOP")
        self.log("=" * 70)
        
        start_time = time.time()
        
        for idx, row in requests_df.iterrows():
            request_id = row['request_id']
            user_id = row['user_id']
            
            self.log(f"Processing request: {request_id} ({idx+1}/{len(requests_df)}) - User: {user_id}")
            request_start_time = time.time()
            
            try:
                # Fresh request-level agent context
                res = agent.process_request(request_id, user_id)
                decision = res['decision']
                validation = res['validation']
                token_stats = res['token_stats']
                
                # Accumulate metrics
                self.total_model_calls += 1
                self.total_prompt_tokens += token_stats['prompt_tokens']
                self.total_completion_tokens += token_stats['completion_tokens']
                self.total_tokens += token_stats['total_tokens']
                
                duration = time.time() - request_start_time
                self.success_count += 1
                
                # Log success and stats
                self.log(f"  [OK] Decision: {decision['recommended_payment_method']} | Amount safe: {decision['amount_safe_to_pay']}")
                self.log(f"  [OK] Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
                if validation['passed']:
                    self.validation_passed_count += 1
                else:
                    self.log(f"    Validation Errors: {validation['details']}")
                    
                self.log(f"  [OK] Tokens: {token_stats['total_tokens']} | Duration: {duration:.2f}s")
                
                # Format output columns
                results.append({
                    'request_id': decision['request_id'],
                    'amount_safe_to_pay': decision['amount_safe_to_pay'],
                    'affordability_status': decision['affordability_status'],
                    'recommended_payment_method': decision['recommended_payment_method'],
                    'payment_plan': decision['payment_plan'],
                    'earliest_date_for_full_payment': decision['earliest_date_for_full_payment'],
                    'spending_changes_needed': decision['spending_changes_needed'],
                    'decision_explanation': decision['decision_explanation']
                })
                
            except Exception as e:
                self.failure_count += 1
                self.log(f"  [ERROR] Failed to process request: {request_id} - Error: {str(e)}")
                # Fill with standard failure fallback row
                results.append({
                    'request_id': request_id,
                    'amount_safe_to_pay': 0.0,
                    'affordability_status': 'not_affordable',
                    'recommended_payment_method': 'not_recommended',
                    'payment_plan': 'none',
                    'earliest_date_for_full_payment': '',
                    'spending_changes_needed': 'none',
                    'decision_explanation': f"System processing error: {str(e)}"
                })
                
            # Pause between requests to stay comfortably below rate limits
            time.sleep(1.0)
            print()
            
        self.total_duration_sec = time.time() - start_time
        
        # 2. Save final predictions to output.csv
        output_df = pd.DataFrame(results)
        output_df.to_csv(OUTPUT_FILE, index=False)
        self.log(f"  Saved final predictions of {len(output_df)} requests to: {OUTPUT_FILE}")
        
        # 3. Generate usage report
        self.generate_usage_report()
        
        # 4. Print final summary
        self.print_final_summary()

    def generate_usage_report(self):
        """Generates code/evaluation/usage_report.md containing final token and cost analysis."""
        avg_tokens = self.total_tokens / max(1, self.success_count)
        # Cost estimate on Groq on-demand tier for Qwen/Qwen3 is roughly $0.05 per 1M prompt and $0.08 per 1M output tokens
        estimated_cost = (self.total_prompt_tokens * 0.00000005) + (self.total_completion_tokens * 0.00000008)
        
        report_content = f"""# HackerRank Orchestrate — Buy or Wait? Token Usage Report

This file summarizes the final full-dataset run's token metrics and cost analysis.

## 1. Executive Summary

* **Model Provider:** Groq Cloud
* **Model Name:** `qwen/qwen3.6-27b`
* **Total Requests Processed:** {self.success_count + self.failure_count}
* **Successful Requests:** {self.success_count}
* **Failed Requests:** {self.failure_count}
* **Validation Passed:** {self.validation_passed_count}

## 2. Token Statistics

* **Total Model Calls:** {self.total_model_calls}
* **Total Prompt Tokens:** {self.total_prompt_tokens}
* **Total Completion Tokens:** {self.total_completion_tokens}
* **Total Tokens Combined:** {self.total_tokens}
* **Average Tokens Per Request:** {avg_tokens:.2f}

## 3. Cost Analysis

* **Estimated Cost (Prompt Tokens):** ${self.total_prompt_tokens * 0.00000005:.6f}
* **Estimated Cost (Completion Tokens):** ${self.total_completion_tokens * 0.00000008:.6f}
* **Total Estimated API Cost:** ${estimated_cost:.6f}
* **Average Cost Per Request:** ${estimated_cost / max(1, self.success_count):.6f}

## 4. Run Metadata

* **Total Duration:** {self.total_duration_sec / 60.0:.2f} minutes
* **Average Latency Per Request:** {self.total_duration_sec / max(1, self.success_count):.2f} seconds
* **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}
"""
        # Ensure evaluation folder exists
        Report_Dir = REPORT_FILE.parent
        Report_Dir.mkdir(parents=True, exist_ok=True)
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report_content)
        self.log(f"  Saved final evaluation usage report to: {REPORT_FILE}")

    def print_final_summary(self):
        print()
        print("=" * 70)
        print("FINAL PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total Requests: {self.success_count + self.failure_count}")
        print(f"Successful: {self.success_count}")
        print(f"Failed: {self.failure_count}")
        print(f"Validation Passed: {self.validation_passed_count}")
        print(f"Total Model Calls: {self.total_model_calls}")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"Total Run Duration: {self.total_duration_sec / 60.0:.2f} minutes")
        print("=" * 70)
        print()

def main():
    pipeline = BatchPipeline()
    try:
        pipeline.run_preflight_checks()
        pipeline.process_all_requests()
    except Exception as e:
        pipeline.log(f"FATAL PIPELINE ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
