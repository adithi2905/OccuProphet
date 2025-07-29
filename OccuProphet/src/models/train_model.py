import sys
import pandas as pd
from prophet import Prophet

def train_prophet_model(df, state_name):
    df['state'] = df['state'].str.upper()
    state_name = state_name.upper()
    state_df = df[df['state'] == state_name]
    if state_df.empty or 'date' not in state_df.columns:
        return None, None
    state_df = state_df[['date', 'occupancy_rate']].copy()
    state_df.rename(columns={'date': 'ds', 'occupancy_rate': 'y'}, inplace=True)
    state_df['ds'] = pd.to_datetime(state_df['ds'])
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(state_df)
    return model, state_df

if __name__ == "__main__":
    state = sys.argv[1]
    df = pd.read_csv("pre_covid_occupancy.csv")
    model, _ = train_prophet_model(df, state)
    print(f"Model trained for {state}" if model else f"Training failed for {state}")