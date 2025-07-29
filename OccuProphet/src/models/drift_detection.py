import sys
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error
from retrain_model import retrain_prophet_model

def detect_drift(df_pre, df_post, state_name):
    df_pre['state'] = df_pre['state'].str.upper()
    df_post['state'] = df_post['state'].str.upper()
    state_name = state_name.upper()
    df_pre_state = df_pre[df_pre['state'] == state_name]
    df_post_state = df_post[df_post['state'] == state_name]
    if df_pre_state.empty or df_post_state.empty:
        return {"error": f"No data for {state_name}"}
    pre_mean = df_pre_state['occupancy_rate'].mean()
    predictions = [pre_mean] * len(df_post_state)
    pre_mae = mean_absolute_error(df_pre_state['occupancy_rate'], [pre_mean] * len(df_pre_state))
    current_mae = mean_absolute_error(df_post_state['occupancy_rate'], predictions)
    mae_threshold = pre_mae * 1.5
    drift_detected = current_mae > mae_threshold
    result = {
        "pre_mae": pre_mae,
        "post_mae": current_mae,
        "threshold": mae_threshold,
        "drift_detected": drift_detected
    }
    if drift_detected:
        model = retrain_prophet_model(df_pre, df_post, state_name)
        result["model_retrained"] = bool(model)
    return result

if __name__ == "__main__":
    state = sys.argv[1]
    df_pre = pd.read_csv("pre_covid_occupancy.csv")
    df_post = pd.read_csv("post_covid_occupancy.csv")
    result = detect_drift(df_pre, df_post, state)
    print(json.dumps(result))