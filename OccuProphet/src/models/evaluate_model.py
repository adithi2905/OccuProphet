import sys
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
from train_model import train_prophet_model

def evaluate_model_on_post_covid(model, file, state_name):
    df_post = pd.read_csv(file)
    df_post['state'] = df_post['state'].str.upper()
    state_name = state_name.upper()
    state_df = df_post[df_post['state'] == state_name]
    if state_df.empty or 'date' not in state_df.columns:
        return None, None
    state_df = state_df[['date', 'occupancy_rate']].copy()
    state_df.rename(columns={'date': 'ds', 'occupancy_rate': 'y'}, inplace=True)
    state_df['ds'] = pd.to_datetime(state_df['ds'])
    forecast = model.predict(state_df[['ds']])
    return state_df, forecast

if __name__ == "__main__":
    state = sys.argv[1]
    model, _ = train_prophet_model(pd.read_csv("pre_covid_occupancy.csv"), state)
    if not model:
        print(json.dumps({"error": "Model training failed"})); exit(1)
    state_df, forecast = evaluate_model_on_post_covid(model, "post_covid_occupancy.csv", state)
    if state_df is None:
        print(json.dumps({"error": "Evaluation failed"})); exit(1)
    mae = mean_absolute_error(state_df['y'], forecast['yhat'])
    print(json.dumps({"mae": mae}))