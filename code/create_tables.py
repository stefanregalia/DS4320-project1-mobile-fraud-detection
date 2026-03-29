# create_tables.py
# Creates relational tables from raw PaySim dataset: 
# Transactions, accounts, transaction_types, time_steps (CSV and parquet)

import pandas as pd
import logging
import os

# Configure logging to track progress and errors during table creation
logging.basicConfig(
    filename='../.logs/create_tables.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_raw_data(filepath):
    """Load raw PaySim CSV file"""
    try:
        logging.info(f"Loading raw data from {filepath}")
        df = pd.read_csv(filepath, on_bad_lines='skip')
        logging.info(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def drop_leaky_columns(df):
    """Drop balance columns that cause data leakage"""
    try:
        leaky_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        df = df.drop(columns=leaky_cols)
        logging.info(f"Dropped leaky columns: {leaky_cols}")
        return df
    except Exception as e:
        logging.error(f"Failed to drop leaky columns: {e}")
        raise

def engineer_features(df):
    """Engineer is_merchant, hour_of_day, and day_of_month features"""
    try:
        # Deriving recipient type from account ID prefix (M = merchant, C = customer)
        df['is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        # Deriving time features from simulation step (1 step = 1 hour)
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = df['step'] // 24
        logging.info("Engineered features: is_merchant, hour_of_day, day_of_month")
        return df
    except Exception as e:
        logging.error(f"Failed to engineer features: {e}")
        raise

def create_transaction_types(df):
    """Create transaction_types lookup table"""
    try:
        types = df['type'].unique()
        transaction_types = pd.DataFrame({
            'type_id': range(1, len(types) + 1),
            'type_name': types
        })
        logging.info(f"Created transaction_types table with {len(transaction_types)} rows")
        return transaction_types
    except Exception as e:
        logging.error(f"Failed to create transaction_types table: {e}")
        raise

def create_accounts(df):
    """Create accounts table from unique sender and recipient IDs"""
    try:
        senders = pd.DataFrame({'account_id': df['nameOrig'].unique()})
        recipients = pd.DataFrame({'account_id': df['nameDest'].unique()})
        # Combine sender and recipient IDs, removing duplicates
        accounts = pd.concat([senders, recipients]).drop_duplicates()
        # Remove corrupted rows that do not follow the C/M prefix format
        accounts = accounts[accounts['account_id'].str.match(r'^[CM]\d+$')]
        accounts['account_type'] = accounts['account_id'].apply(
            lambda x: 'merchant' if x.startswith('M') else 'customer'
        )
        logging.info(f"Created accounts table with {len(accounts)} rows")
        return accounts
    except Exception as e:
        logging.error(f"Failed to create accounts table: {e}")
        raise

def create_time_steps(df):
    """Create time_steps table from unique step values"""
    try:
        time_steps = df[['step']].drop_duplicates().copy()
        time_steps['hour_of_day'] = time_steps['step'] % 24
        time_steps['day_of_month'] = time_steps['step'] // 24
        time_steps = time_steps.sort_values('step').reset_index(drop=True)
        logging.info(f"Created time_steps table with {len(time_steps)} rows")
        return time_steps
    except Exception as e:
        logging.error(f"Failed to create time_steps table: {e}")
        raise

def create_transactions(df, transaction_types):
    """Create core transactions table"""
    try:
        # Mapping transaction type names to integer IDs for normalization
        type_map = dict(zip(transaction_types['type_name'], transaction_types['type_id']))
        transactions = pd.DataFrame({
            'transaction_id': range(1, len(df) + 1),
            'sender_id': df['nameOrig'].values,
            'recipient_id': df['nameDest'].values,
            'type_id': df['type'].map(type_map).values,
            'step': df['step'].values,
            'amount': df['amount'].values,
            'isFraud': df['isFraud'].values
        })
        logging.info(f"Created transactions table with {len(transactions)} rows")
        return transactions
    except Exception as e:
        logging.error(f"Failed to create transactions table: {e}")
        raise

def save_table(df, name, output_dir):
    """Save table as both CSV and parquet"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{name}.csv")
        parquet_path = os.path.join(output_dir, f"{name}.parquet")
        # Save as CSV for readability and parquet for efficient storage
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Saved {name} to {csv_path} and {parquet_path}")
    except Exception as e:
        logging.error(f"Failed to save table {name}: {e}")
        raise

def main():
    raw_path = 'data/raw/PS_20174392719_1491204439457_log 2.csv'
    output_dir = 'data/relational'

    df = load_raw_data(raw_path)
    df = drop_leaky_columns(df)
    df = engineer_features(df)

    # Create all relational tables
    transaction_types = create_transaction_types(df)
    accounts = create_accounts(df)
    time_steps = create_time_steps(df)
    transactions = create_transactions(df, transaction_types)

    # Save all tables to output directory
    save_table(transactions, 'transactions', output_dir)
    save_table(accounts, 'accounts', output_dir)
    save_table(transaction_types, 'transaction_types', output_dir)
    save_table(time_steps, 'time_steps', output_dir)

    logging.info("All tables created successfully")
    print("All tables saved to", output_dir)

if __name__ == '__main__':
    main()