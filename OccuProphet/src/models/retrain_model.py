import pickle
import pandas as pd
from prophet import Prophet

def retrain_prophet_model(pre_covid_df, post_covid_df, state_name):
    pre_covid_df['state'] = pre_covid_df['state'].str.upper()
    post_covid_df['state'] = post_covid_df['state'].str.upper()
    state_name = state_name.upper()
    combined_df = pd.concat([pre_covid_df, post_covid_df], ignore_index=True)
    state_df = combined_df[combined_df['state'] == state_name]
    if state_df.empty or 'date' not in state_df.columns or 'occupancy_rate' not in state_df.columns:
        return None
    state_df = state_df[['date', 'occupancy_rate']].copy()
    state_df.rename(columns={'date': 'ds', 'occupancy_rate': 'y'}, inplace=True)
    state_df['ds'] = pd.to_datetime(state_df['ds'])
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(state_df)
    filename = f"{state_name.lower()}_prophet_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return model