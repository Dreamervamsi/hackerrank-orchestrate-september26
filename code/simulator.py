import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Tuple, Optional

class FinancialSimulator:
    """Performs deterministic 90-day daily balance projections and policy validations."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def detect_recurrence(self, user_id: str, request_date: str) -> List[Dict]:
        """Detect recurring monthly fixed or interval-based debits and credits from history."""
        request_dt = pd.to_datetime(request_date).date()
        df = self.data_manager.get_events(user_id)
        if len(df) == 0:
            return []
            
        df = df.copy()
        df['event_date'] = pd.to_datetime(df['event_date']).dt.date
        
        # Filter history before request_date
        history = df[df['event_date'] < request_dt].copy()
        if len(history) == 0:
            return []
            
        history['month'] = pd.to_datetime(history['event_date']).dt.to_period('M')
        
        recurring = []
        
        # 1. Fixed monthly grouping (description, direction, category, amount)
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
                    
        return recurring

    def simulate_balance(self, user_id: str, request_date: str, payment_amount: float = 0.0, 
                         spending_changes: str = None) -> Tuple[float, str, Dict]:
        """Simulate running balances daily for 90 days, returning minimum balance and date."""
        request_dt = pd.to_datetime(request_date).date()
        profile = self.data_manager.get_profile(user_id)
        balance = float(profile.get('current_available_balance', 0.0)) - payment_amount
        min_balance = balance
        min_date = request_dt
        
        patterns = self.detect_recurrence(user_id, request_date)
        
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
        raw_msgs = self.data_manager.messages_df[self.data_manager.messages_df['user_id'] == user_id]
        for idx, row in raw_msgs.iterrows():
            text = str(row.get('message_text', ''))
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
                        'description': f"Message {row['message_id']}",
                        'direction': direction,
                        'amount': amount,
                        'date': date
                    })

        # Reserve pending transactions
        # Status 'pending' debits from state are applied on day 0/1
        df = self.data_manager.get_events(user_id)
        if len(df) > 0:
            pending_debits = sum(e['amount'] for idx, e in df[df['status'] == 'pending'].iterrows()
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

    def calculate_safe_amount(self, user_id: str, request_date: str, requested_amount: float, min_balance_to_keep: float) -> float:
        """Find the maximum safe amount to pay today using simulation."""
        # Test full payment
        min_bal, _, _ = self.simulate_balance(user_id, request_date, requested_amount)
        if min_bal >= min_balance_to_keep:
            return requested_amount
            
        # Binary search for precise safe amount
        low = 0.0
        high = requested_amount
        best_safe = 0.0
        
        for _ in range(20):
            mid = (low + high) / 2
            min_bal, _, _ = self.simulate_balance(user_id, request_date, mid)
            if min_bal >= min_balance_to_keep:
                best_safe = mid
                low = mid
            else:
                high = mid
                
        return float(np.round(best_safe, 2))

    def calculate_earliest_full_payment_date(self, user_id: str, requested_amount: float, 
                                             request_date: str, min_balance_to_keep: float, forecast_days: int = 90) -> str:
        """Find the first date when full payment is safe without dropping below minimum balance."""
        request_dt = pd.to_datetime(request_date).date()
        
        # Calculate baseline daily balances ONCE outside the loop to achieve a 90x execution speedup!
        _, _, daily_balances = self.simulate_balance(user_id, request_date, 0.0)
        
        for day in range(forecast_days):
            current_date = request_dt + timedelta(days=day)
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            is_safe = True
            for d in range(day, 91):
                future_date = request_dt + timedelta(days=d)
                if daily_balances.get(future_date, 0.0) - requested_amount < min_balance_to_keep:
                    is_safe = False
                    break
                    
            if is_safe:
                return current_date_str
                
        return ""

    def validate_decision(self, decision: Dict, user_id: str, request_date: str, requested_amount: float, min_balance_to_keep: float) -> Tuple[bool, str]:
        """Validate that the decision doesn't violate financial constraints."""
        errors = []
        
        # Check amount_safe_to_pay bounds
        if decision['amount_safe_to_pay'] < 0:
            errors.append("amount_safe_to_pay is negative")
        if decision['amount_safe_to_pay'] > requested_amount:
            errors.append("amount_safe_to_pay exceeds requested_amount")
        
        # Check affordability status is valid
        valid_statuses = ['affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable']
        if decision['affordability_status'] not in valid_statuses:
            errors.append(f"Invalid affordability_status: {decision['affordability_status']}")
        
        # Check payment method is valid
        valid_methods = ['full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended']
        if decision['recommended_payment_method'] not in valid_methods:
            errors.append(f"Invalid payment method: {decision['recommended_payment_method']}")

        # Validate and simulate the payment plan
        plan_str = decision['payment_plan']
        spending_changes = decision['spending_changes_needed']
        
        if plan_str != 'none':
            try:
                payments = []
                plan_parts = plan_str.split('|')
                
                for part in plan_parts:
                    date_str, amount_str = part.split(':')
                    pay_date = pd.to_datetime(date_str).date()
                    pay_amount = float(amount_str)
                    payments.append((pay_date, pay_amount))
                    
                # Simulate the entire plan day-by-day
                profile = self.data_manager.get_profile(user_id)
                balance = float(profile.get('current_available_balance', 0.0))
                min_observed = balance
                request_dt = pd.to_datetime(request_date).date()
                
                patterns = self.detect_recurrence(user_id, request_date)
                
                stops = set()
                if spending_changes and spending_changes != 'none':
                    for change in spending_changes.split('|'):
                        parts = change.split(':')
                        if len(parts) == 2 and parts[0] == 'stop':
                            stops.add(parts[1])
                            
                # Get message txs
                message_txs = []
                raw_msgs = self.data_manager.messages_df[self.data_manager.messages_df['user_id'] == user_id]
                for idx, row in raw_msgs.iterrows():
                    text = str(row.get('message_text', ''))
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
                            
                df = self.data_manager.get_events(user_id)
                if len(df) > 0:
                    pending_debits = sum(e['amount'] for idx, e in df[df['status'] == 'pending'].iterrows()
                                         if e['direction'] == 'debit' and pd.notna(e['amount'])
                                         and e['event_id'] not in stops)
                    balance -= pending_debits
                    if balance < min_observed:
                        min_observed = balance
                    
                next_trigger_dates = {}
                for p in patterns:
                    if p['type'] == 'variable_interval':
                        last_date = p['last_date']
                        next_date = last_date + timedelta(days=p['interval_days'])
                        while next_date < request_dt:
                            next_date += timedelta(days=p['interval_days'])
                        next_trigger_dates[p['description']] = next_date
                        
                for day in range(91):
                    current_date = request_dt + timedelta(days=day)
                    daily_change = 0.0
                    
                    for pay_date, pay_amount in payments:
                        if pay_date == current_date:
                            daily_change -= pay_amount
                            
                    if day > 0:
                        for tx in message_txs:
                            if tx['date'] == current_date:
                                if tx['direction'] == 'credit':
                                    daily_change += tx['amount']
                                else:
                                    daily_change -= tx['amount']

                        for p in patterns:
                            if p['type'] == 'fixed_monthly':
                                if current_date.day == p['day_of_month']:
                                    amount = p['amount']
                                    if p['direction'] == 'credit':
                                        daily_change += amount
                                    else:
                                        daily_change -= amount

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
                        
                if min_observed < min_balance_to_keep:
                    errors.append(f"Simulation failed: Running balance falls to {min_observed:.2f} (below minimum required {min_balance_to_keep})")
                    
            except Exception as e:
                errors.append(f"Validation error running simulation: {str(e)}")
                
        if errors:
            return False, "; ".join(errors)
        return True, "All checks passed"
