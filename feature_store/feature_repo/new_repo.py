import os
from datetime import timedelta
import pandas as pd
import numpy as np 

from feast import (
    Entity, FeatureView, FeatureService, Field, FileSource, RequestSource
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_PATH, "data")

driver = Entity(name="driver", join_keys=["driver_id"])

driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path=r"C:/MLOps/otus-feature-store/feature_store/feature_repo/data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# FV #1
average_check_base_fv = FeatureView(
    name="driver_average_check_base",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_daily_trips", dtype=Int64),
        Field(name="conv_rate", dtype=Float32),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "revenue", "type": "base"},
)

# FV #2
average_check_premium_fv = FeatureView(
    name="driver_average_check_premium", 
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="acc_rate", dtype=Float32),
        Field(name="conv_rate", dtype=Float32),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "revenue", "type": "premium"},
)

# RequestSource
current_order_request = RequestSource(
    name="current_order",
    schema=[
        Field(name="current_trip_value", dtype=Float64),
        Field(name="is_peak_hour", dtype=Int64),
    ],
)

# On-Demand
@on_demand_feature_view(
    sources=[average_check_base_fv, average_check_premium_fv, current_order_request],
    schema=[
        Field(name="predicted_check", dtype=Float64),
        Field(name="revenue_potential", dtype=Float64),
    ],
)
def driver_check_prediction(inputs: pd.DataFrame):
    
    df = pd.DataFrame(index=inputs.index)
    
    # Качество
    quality_factor = inputs["acc_rate"] * inputs["conv_rate"]
    
    # Базовый чек
    base_daily_check = inputs["avg_daily_trips"] * 500 * quality_factor
    
    # Предсказанный чек
    df["predicted_check"] = base_daily_check + inputs["current_trip_value"]
    
    # Пиковый коэффициент
    peak_multiplier = np.where(inputs["is_peak_hour"] == 1, 1.3, 1.0)
    df["revenue_potential"] = df["predicted_check"] * peak_multiplier
    
    return df

# FeatureService
driver_revenue_service = FeatureService(
    name="driver_revenue_v1",
    features=[
        average_check_base_fv,
        average_check_premium_fv, 
        driver_check_prediction
    ]
)
