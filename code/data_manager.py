import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

class DataManager:
    """Loads all CSV datasets once on startup and provides fast indexed lookups."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.load_datasets()
        self.build_indexes()
        
    def load_datasets(self):
        """Load all raw datasets from disk."""
        print("Loading datasets...")
        self.requests_df = pd.read_csv(self.dataset_dir / "requests.csv")
        self.profiles_df = pd.read_csv(self.dataset_dir / "financial_profiles.csv")
        self.events_df = pd.read_csv(self.dataset_dir / "financial_events.csv")
        self.messages_df = pd.read_csv(self.dataset_dir / "messages.csv")
        self.images_df = pd.read_csv(self.dataset_dir / "images.csv")
        self.payment_options_df = pd.read_csv(self.dataset_dir / "request_payment_options.csv")
        self.exchange_rates_df = pd.read_csv(self.dataset_dir / "exchange_rates.csv")
        
        print(f"* Loaded {len(self.requests_df)} requests")
        print(f"* Loaded {len(self.profiles_df)} profiles")
        print(f"* Loaded {len(self.events_df)} events")
        print(f"* Loaded {len(self.messages_df)} messages")
        print(f"* Loaded {len(self.images_df)} image references")
        print(f"* Loaded {len(self.payment_options_df)} payment options")
        print(f"* Loaded {len(self.exchange_rates_df)} exchange rates")

    def build_indexes(self):
        """Index datasets for instant O(1) request-specific lookups."""
        print("Indexing datasets...")
        # 1. Profiles indexed by user_id
        self.profiles_index = self.profiles_df.set_index('user_id').to_dict('index')
        
        # 2. Events grouped by user_id
        self.events_index = {}
        for user_id, group in self.events_df.groupby('user_id'):
            self.events_index[user_id] = group
            
        # 3. Messages grouped by user_id
        self.messages_index = {}
        for user_id, group in self.messages_df.groupby('user_id'):
            self.messages_index[user_id] = group.to_dict('records')
            
        # 4. Images grouped by user_id
        self.images_index = {}
        for user_id, group in self.images_df.groupby('user_id'):
            self.images_index[user_id] = group.to_dict('records')
            
        # 5. Payment options grouped by request_id
        self.payment_options_index = {}
        for request_id, group in self.payment_options_df.groupby('request_id'):
            self.payment_options_index[request_id] = group.to_dict('records')
            
        # 6. Exchange rates indexed by (rate_date, from_currency, to_currency)
        self.exchange_rates_index = {}
        for idx, row in self.exchange_rates_df.iterrows():
            key = (row['rate_date'], row['from_currency'], row['to_currency'])
            self.exchange_rates_index[key] = float(row['rate'])
            
        print("* Dataset indexing complete")

    def get_request(self, request_id: str) -> Dict[str, Any]:
        """Lookup request details."""
        req_row = self.requests_df[self.requests_df['request_id'] == request_id]
        if len(req_row) == 0:
            return {}
        return req_row.iloc[0].to_dict()

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Lookup financial profile for a user."""
        return self.profiles_index.get(user_id, {})

    def get_events(self, user_id: str) -> pd.DataFrame:
        """Lookup raw financial events dataframe for a user."""
        return self.events_index.get(user_id, pd.DataFrame())

    def get_messages(self, user_id: str) -> List[Dict[str, Any]]:
        """Lookup messages for a user."""
        return self.images_index.get(user_id, [])

    def get_images(self, user_id: str) -> List[Dict[str, Any]]:
        """Lookup images for a user."""
        return self.images_index.get(user_id, [])

    def get_payment_options(self, request_id: str) -> List[Dict[str, Any]]:
        """Lookup payment options for a request."""
        return self.payment_options_index.get(request_id, [])

    def get_exchange_rate(self, rate_date: str, from_curr: str, to_curr: str) -> float:
        """Get exchange rate between two currencies."""
        if from_curr == to_curr:
            return 1.0
        # Try exact date rate
        rate = self.exchange_rates_index.get((rate_date, from_curr, to_curr))
        if rate is not None:
            return rate
            
        # Fallback to closest available rate for the currency pair
        matching_rates = self.exchange_rates_df[
            (self.exchange_rates_df['from_currency'] == from_curr) &
            (self.exchange_rates_df['to_currency'] == to_curr)
        ]
        if len(matching_rates) > 0:
            return float(matching_rates.iloc[0]['rate'])
            
        return 1.0 # fallback
