import subprocess
from datetime import datetime
import pandas as pd
from feast import FeatureStore
import os

def run_our_demo_final():
    repo_path = r"C:\MLOps\otus-feature-store\feature_store\feature_repo"
    store = FeatureStore(repo_path=repo_path)
    print("=" * 50)
    
    print("\nHistorical (обучение модели)")
    fetch_historical_features_our(store)
    
    print("\nOnline (реал-тайм предсказания)")
    fetch_online_features_our(store)
    
    print("\nПолная проверка НАШИХ VIEW")
    full_test(store)
    

    print("Historical: данные для обучения ✓")
    
def fetch_historical_features_our(store):
    """Historical НАШИ данные 2024"""
    entity_df = pd.DataFrame({
        "driver_id": [1001, 1005],
        "event_timestamp": [
            datetime(2024, 10, 17, 12, 7, 8),
            datetime(2024, 10, 2, 11, 0, 0),
        ],
    })
    
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_average_check_base:avg_daily_trips",
            "driver_average_check_base:conv_rate",
            "driver_average_check_premium:acc_rate",
        ],
    ).to_df()
    
    print(training_df[['driver_id', 'avg_daily_trips', 'conv_rate', 'acc_rate']].round(2))

def fetch_online_features_our(store):
    """Online НАШИ фичи"""
    returned_features = store.get_online_features(
        features=[
            "driver_average_check_base:avg_daily_trips",
            "driver_average_check_base:conv_rate",
            "driver_average_check_premium:acc_rate",
        ],
        entity_rows=[{"driver_id": 1001}, {"driver_id": 1005}],
    ).to_dict()
    
    print("Driver 1001,1005:")
    for key, value in sorted(returned_features.items()):
        print(f"  {key}: {value}")

def full_test(store):
    """Тест каждого VIEW отдельно"""
    print("driver_average_check_base:")
    base_features = store.get_online_features(
        features=["driver_average_check_base:avg_daily_trips"],
        entity_rows=[{"driver_id": 1001}]
    ).to_dict()
    print(f"  avg_daily_trips: {base_features['avg_daily_trips'][0]}")
    
    print("\ndriver_average_check_premium:")
    premium_features = store.get_online_features(
        features=["driver_average_check_premium:acc_rate"],
        entity_rows=[{"driver_id": 1001}]
    ).to_dict()
    print(f"  acc_rate: {premium_features['acc_rate'][0]}")

if __name__ == "__main__":
    run_our_demo_final()
