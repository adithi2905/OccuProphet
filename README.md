
# OccuProphet: Drift-Aware Hospital Occupancy Forecasting

## Project Description

OccuProphet is a time series forecasting system designed to model and predict hospital bed occupancy trends. It trains on pre-COVID data to understand normal patterns and evaluates on post-COVID data to uncover how pandemic shocks cause existing models to fail.

This project demonstrates how concept drift impacts forecasting systems and explores lightweight solutions for detecting these shifts before moving to complex MLOps pipelines.

---

## Data Modeling

### Pre-COVID Data

To model pre-pandemic hospital occupancy trends, I merged two Kaggle datasets:

1. *Hospitals and Beds in India (2019)* – provided state-wise hospital and bed counts.
2. *India State-wise Population Dataset* – offered population breakdowns.

These datasets provided **static information** for each state. To convert this static data into a **monthly time series**, I:

* Assigned **month-wise occupancy rates** arbitrarily but logically:

  * Urban states: Higher baseline occupancy (60–70%) due to denser populations and greater demand.
  * Rural states: Lower baseline occupancy (35–50%) reflecting limited access to healthcare.
  * Added seasonal variation: Flu season spikes (November–February) and minor dips during summer (May–June).
* Generated a monthly time series for 2018–2019 with columns like:

  * `state`, `date`, `total_beds`, `occupied_beds`, `occupancy_rate`.

This synthetic time series represents “normal” healthcare dynamics before COVID disruptions.

---

### Post-COVID Data

For post-pandemic data, I simulated the COVID-19 impact on hospital capacity (2020–2021):

* Pandemic waves caused sharp occupancy surges, peaking at 100–120% during crisis months (for example, April–June 2020).
* Off-peak months saw occupancy settle at 60–75%, slightly elevated compared to pre-COVID levels.

This dataset reflects the stress and volatility that COVID imposed on healthcare infrastructure.

---

## Why Prophet Over LSTM

Prophet was chosen over LSTM because:

* Data Characteristics: Hospital occupancy data is low-volume and monthly, making LSTM’s complex architecture unnecessary.
* Interpretability: Prophet provides trend, seasonality, and event decomposition, making results more explainable to healthcare professionals.
* Faster Experimentation: Prophet is quick to train and tune, ideal for rapid prototyping.
* Domain Alignment: Prophet’s seasonality handling directly suits healthcare demand patterns like flu seasons.

---

## Why Not Advanced Drift Detection

Instead of using EvidentlyAI, KS-tests, or p-values, I applied a custom drift detection logic:

* Feature Drift: Compared mean occupancy pre- and post-COVID.
* Concept Drift: Compared pre-COVID MAE to post-COVID MAE, triggering drift alerts if post-COVID errors exceeded 1.5× baseline MAE.

This lightweight approach worked for prototyping and visual storytelling. In future iterations, I plan to integrate robust statistical tests and tools like Evidently for production-scale drift monitoring.

---

## Next Steps

* Experiment with LSTM or Transformer models for comparison.
* Add automated drift detection metrics (PSI, KS-test, p-values).
* Deploy as a REST API for hospital dashboards.

