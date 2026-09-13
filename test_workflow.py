#!/usr/bin/env python3
"""
Test workflow script to verify data relationships for one user from requests.csv
This script demonstrates the data workflow without making financial decisions.
"""

import pandas as pd
from pathlib import Path

# Paths
DATASET_DIR = Path("dataset")

def main():
    print("=" * 70)
    print("TEST WORKFLOW: DATA RELATIONSHIP VERIFICATION")
    print("=" * 70)
    print()

    # Load all datasets
    print("Loading datasets...")
    requests_df = pd.read_csv(DATASET_DIR / "requests.csv")
    financial_profiles_df = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
    financial_events_df = pd.read_csv(DATASET_DIR / "financial_events.csv")
    messages_df = pd.read_csv(DATASET_DIR / "messages.csv")
    images_df = pd.read_csv(DATASET_DIR / "images.csv")
    payment_options_df = pd.read_csv(DATASET_DIR / "request_payment_options.csv")
    print(f"✓ Loaded {len(requests_df)} requests")
    print(f"✓ Loaded {len(financial_profiles_df)} financial profiles")
    print(f"✓ Loaded {len(financial_events_df)} financial events")
    print(f"✓ Loaded {len(messages_df)} messages")
    print(f"✓ Loaded {len(images_df)} image references")
    print(f"✓ Loaded {len(payment_options_df)} payment options")
    print()

    # Select first user from requests.csv
    first_user_id = requests_df.iloc[0]['user_id']
    print(f"SELECTED USER: {first_user_id}")
    print()

    # [1] REQUESTS
    print("[1] REQUESTS")
    print("-" * 70)
    user_requests = requests_df[requests_df['user_id'] == first_user_id]
    print(f"Found {len(user_requests)} request(s) for {first_user_id}")
    print()
    for idx, row in user_requests.iterrows():
        print(f"  request_id: {row['request_id']}")
        print(f"  request_date: {row['request_date']}")
        print(f"  request_type: {row['request_type']}")
        print(f"  requested_amount: {row['requested_amount']} (currency from profile)")
        print(f"  desired_completion_date: {row['desired_completion_date']}")
        print(f"  allows_partial_payment: {row['allows_partial_payment']}")
        print(f"  request_text: {row['request_text'][:100]}..." if len(row['request_text']) > 100 else f"  request_text: {row['request_text']}")
        print()
    print()

    # [2] FINANCIAL PROFILE
    print("[2] FINANCIAL PROFILE")
    print("-" * 70)
    user_profile = financial_profiles_df[financial_profiles_df['user_id'] == first_user_id]
    if len(user_profile) > 0:
        profile = user_profile.iloc[0]
        print(f"Found profile for {first_user_id}")
        print(f"  home_currency: {profile['home_currency']}")
        print(f"  current_available_balance: {profile['current_available_balance']}")
        print(f"  minimum_balance_to_keep: {profile['minimum_balance_to_keep']}")
        print(f"  financial_priorities: {profile['financial_priorities']}")
        print(f"  expense_categories_to_protect: {profile['expense_categories_to_protect']}")
        print(f"  expense_categories_user_is_willing_to_reduce: {profile['expense_categories_user_is_willing_to_reduce']}")
        print(f"  expense_categories_user_is_willing_to_stop: {profile['expense_categories_user_is_willing_to_stop']}")
        print(f"  payment_methods_user_will_consider: {profile['payment_methods_user_will_consider']}")
        print(f"  max_installment_months: {profile['max_installment_months'] if pd.notna(profile['max_installment_months']) else 'N/A'}")
    else:
        print(f"WARNING: No profile found for {first_user_id}")
    print()
    print()

    # [3] FINANCIAL EVENTS
    print("[3] FINANCIAL EVENTS")
    print("-" * 70)
    user_events = financial_events_df[financial_events_df['user_id'] == first_user_id]
    print(f"Found {len(user_events)} financial event(s) for {first_user_id}")
    print(f"  Relationship: user_id in financial_events.csv matches user_id in requests.csv")
    print()
    if len(user_events) > 0:
        print("Sample events (first 5):")
        for idx, row in user_events.head(5).iterrows():
            print(f"  event_id: {row['event_id']}")
            print(f"  event_type: {row['event_type']}")
            print(f"  description: {row['description']}")
            print(f"  category: {row['category']}")
            print(f"  direction: {row['direction']}")
            print(f"  amount: {row['amount'] if pd.notna(row['amount']) else 'BLANK (may be in image)'}")
            print(f"  currency: {row['currency']}")
            print(f"  event_date: {row['event_date']}")
            print(f"  settlement_date: {row['settlement_date']}")
            print(f"  status: {row['status']}")
            print(f"  linked_event_id: {row['linked_event_id'] if pd.notna(row['linked_event_id']) else 'N/A'}")
            print(f"  flexibility: {row['flexibility']}")
            print()
        if len(user_events) > 5:
            print(f"  ... and {len(user_events) - 5} more events")
    print()

    # [4] MESSAGES
    print("[4] MESSAGES")
    print("-" * 70)
    # Messages can be linked by user_id, request_id, or related_event_id
    user_messages_by_user = messages_df[messages_df['user_id'] == first_user_id]
    print(f"Messages linked by user_id: {len(user_messages_by_user)}")
    
    # Also check messages linked by request_id for this user's requests
    user_request_ids = user_requests['request_id'].tolist()
    user_messages_by_request = messages_df[messages_df['request_id'].isin(user_request_ids)]
    print(f"Messages linked by request_id: {len(user_messages_by_request)}")
    
    # Check messages linked by related_event_id to this user's events
    user_event_ids = user_events['event_id'].tolist()
    user_messages_by_event = messages_df[messages_df['related_event_id'].isin(user_event_ids)]
    print(f"Messages linked by related_event_id: {len(user_messages_by_event)}")
    
    all_user_messages = pd.concat([user_messages_by_user, user_messages_by_request, user_messages_by_event]).drop_duplicates()
    print(f"Total unique messages for {first_user_id}: {len(all_user_messages)}")
    print()
    if len(all_user_messages) > 0:
        print("Sample messages:")
        for idx, row in all_user_messages.head(3).iterrows():
            print(f"  message_id: {row['message_id']}")
            print(f"  user_id: {row['user_id']}")
            print(f"  request_id: {row['request_id'] if pd.notna(row['request_id']) else 'N/A'}")
            print(f"  related_event_id: {row['related_event_id'] if pd.notna(row['related_event_id']) else 'N/A'}")
            print(f"  sent_at: {row['sent_at']}")
            print(f"  source_type: {row['source_type']}")
            print(f"  message_text: {row['message_text'][:100]}..." if len(row['message_text']) > 100 else f"  message_text: {row['message_text']}")
            print()
    print()

    # [5] IMAGES
    print("[5] IMAGES")
    print("-" * 70)
    # Images can be linked by user_id, request_id, or related_event_id
    user_images_by_user = images_df[images_df['user_id'] == first_user_id]
    print(f"Images linked by user_id: {len(user_images_by_user)}")
    
    user_images_by_request = images_df[images_df['request_id'].isin(user_request_ids)]
    print(f"Images linked by request_id: {len(user_images_by_request)}")
    
    user_images_by_event = images_df[images_df['related_event_id'].isin(user_event_ids)]
    print(f"Images linked by related_event_id: {len(user_images_by_event)}")
    
    all_user_images = pd.concat([user_images_by_user, user_images_by_request, user_images_by_event]).drop_duplicates()
    print(f"Total unique image references for {first_user_id}: {len(all_user_images)}")
    print()
    if len(all_user_images) > 0:
        print("Image references:")
        for idx, row in all_user_images.iterrows():
            print(f"  image_id: {row['image_id']}")
            print(f"  user_id: {row['user_id']}")
            print(f"  request_id: {row['request_id']}")
            print(f"  related_event_id: {row['related_event_id']}")
            print(f"  File path: dataset/media/images/{row['image_id']}.png")
            print()
    else:
        print("No images found for this user")
    print()

    # [6] PAYMENT OPTIONS
    print("[6] PAYMENT OPTIONS")
    print("-" * 70)
    user_payment_options = payment_options_df[payment_options_df['request_id'].isin(user_request_ids)]
    print(f"Found {len(user_payment_options)} payment option(s) for {first_user_id}'s request(s)")
    print(f"  Relationship: request_id in request_payment_options.csv matches request_id in requests.csv")
    print()
    if len(user_payment_options) > 0:
        # Group by request_id
        for request_id in user_request_ids:
            request_options = user_payment_options[user_payment_options['request_id'] == request_id]
            if len(request_options) > 0:
                print(f"  Payment options for {request_id}:")
                for idx, row in request_options.iterrows():
                    print(f"    payment_option_id: {row['payment_option_id']}")
                    print(f"    payment_method: {row['payment_method']}")
                    print(f"    payment_amount: {row['payment_amount']}")
                    print(f"    number_of_payments: {row['number_of_payments']}")
                    print(f"    first_payment_date: {row['first_payment_date']}")
                    print(f"    payment_frequency_days: {row['payment_frequency_days'] if pd.notna(row['payment_frequency_days']) else 'N/A'}")
                    print(f"    financing_fee: {row['financing_fee']}")
                    print(f"    total_payable_amount: {row['total_payable_amount']}")
                    print()
    print()

    # SUMMARY
    print("=" * 70)
    print("SUMMARY OF DATA RELATIONSHIPS")
    print("=" * 70)
    print()
    print("Identified relationships:")
    print("  1. requests.csv → financial_profiles.csv via user_id")
    print("  2. requests.csv → financial_events.csv via user_id")
    print("  3. requests.csv → request_payment_options.csv via request_id")
    print("  4. financial_events.csv → messages.csv via related_event_id (when populated)")
    print("  5. financial_events.csv → images.csv via related_event_id")
    print("  6. messages.csv can also link via user_id and request_id")
    print("  7. images.csv can also link via user_id and request_id")
    print()
    print("Note: When financial_events.amount is blank, check images.csv using event_id")
    print()

if __name__ == "__main__":
    main()
