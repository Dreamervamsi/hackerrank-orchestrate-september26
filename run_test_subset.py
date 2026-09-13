import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add root folder to python path
sys.path.insert(0, str(Path(__file__).parent))

from code.data_manager import DataManager
from code.simulator import FinancialSimulator
from code.agent import StructuredFinancialAgent

def main():
    print("=" * 70)
    print("CONTROLLED TEST: SUBSET PIPELINE RUN (5 REQUESTS)")
    print("=" * 70)
    print()
    
    # 1. Initialize
    data_manager = DataManager()
    simulator = FinancialSimulator(data_manager)
    agent = StructuredFinancialAgent(data_manager, simulator)
    
    # Take first 5 requests from Requests CSV
    test_requests = data_manager.requests_df.head(5)
    
    results = []
    
    for idx, row in test_requests.iterrows():
        request_id = row['request_id']
        user_id = row['user_id']
        print(f"[{idx+1}/5] Processing: {request_id} (user: {user_id})...")
        
        try:
            res = agent.process_request(request_id, user_id)
            decision = res['decision']
            validation = res['validation']
            token_stats = res['token_stats']
            
            print(f"  [OK] Recommended Method: {decision['recommended_payment_method']}")
            print(f"  [OK] Amount Safe to Pay: {decision['amount_safe_to_pay']}")
            print(f"  [OK] Affordability Status: {decision['affordability_status']}")
            print(f"  [OK] Payment Plan: {decision['payment_plan']}")
            print(f"  [OK] Validation Result: {'PASSED' if validation['passed'] else 'FAILED'}")
            if not validation['passed']:
                print(f"    Errors: {validation['details']}")
            print()
            
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
            print(f"  [ERROR] Failed: {str(e)}")
            print()
            
    # Save test results
    test_output_file = Path("dataset/test_output.csv")
    pd.DataFrame(results).to_csv(test_output_file, index=False)
    print(f"Saved test output to: {test_output_file}")
    
    print("\n--- SAMPLE TEST OUTPUT DATAFRAME ---")
    print(pd.read_csv(test_output_file).to_string())
    print()

if __name__ == "__main__":
    main()
