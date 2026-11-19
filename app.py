"""Inference app for the MIS 545 NFL ATS project."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='NFL ATS Explorer', page_icon='🏈', layout='centered')
st.title('NFL ATS Explorer')
st.write('Explore whether a team and betting situation will finish a season above a 52.38% ATS cover rate.')
st.caption('ATS means against the spread. Historical inputs must come from seasons before the season being predicted.')


@st.cache_resource
def load_model(modified_time):
    import joblib
    return joblib.load(ROOT / 'best_model.joblib')


if not (ROOT / 'best_model.joblib').exists():
    st.error('The saved model is missing. Run python train_model.py in this folder first.')
    st.stop()

metadata = json.loads((ROOT / 'model_metadata.json').read_text())
model = load_model((ROOT / 'best_model.joblib').stat().st_mtime_ns)
examples = pd.read_csv(ROOT / 'example_inputs.csv')
mode = st.radio('Choose an input method', ['Historical example', 'Enter historical statistics'])

with st.form('prediction'):
    if mode == 'Historical example':
        choices = examples.apply(lambda row: f"{int(row['Season'])} · {row['Team']} · {row['Situation']}", axis=1)
        chosen = st.selectbox('Choose a team and situation', range(len(examples)),
                              format_func=lambda i: choices.iloc[i])
        inputs = examples.iloc[[chosen]][metadata['features']].copy()
        st.caption('These example inputs use only earlier seasons. 2026 outcomes are excluded from the model inputs.')
        st.dataframe(inputs, hide_index=True)
    else:
        team = st.selectbox('Team', metadata['teams'])
        situation = st.selectbox('Betting situation', metadata['situations'])
        st.caption('Enter rates as percentages. Bet counts include wins and losses and exclude pushes.')
        previous_rate = st.number_input('Previous season cover rate (%)', min_value=0.0, max_value=100.0,
                                       value=50.0, step=0.1)
        previous_count = st.number_input('Previous season bet count', min_value=1, value=16, step=1)
        rolling_rate = st.number_input('Pooled cover rate over previous three seasons (%)',
                                      min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        rolling_count = st.number_input('Total bet count over previous three seasons',
                                       min_value=1, value=48, step=1)
        inputs = pd.DataFrame([{'Team': team, 'Situation': situation,
            'Previous_Season_Cover_Rate': previous_rate / 100,
            'Previous_Season_Bet_Count': previous_count,
            'Rolling_3Y_Cover_Rate': rolling_rate / 100,
            'Rolling_3Y_Bet_Count': rolling_count}])[metadata['features']]
    submitted = st.form_submit_button('Generate prediction', type='primary')

if submitted:
    previous_count = float(inputs.iloc[0]['Previous_Season_Bet_Count'])
    rolling_count = float(inputs.iloc[0]['Rolling_3Y_Bet_Count'])
    if pd.notna(previous_count) and rolling_count < previous_count:
        st.error('The three-year bet count must include the previous season bet count.')
    else:
        probability = float(model.predict_proba(inputs)[0, list(model.classes_).index(1)])
        st.metric('Probability of season cover rate above 52.38%', f'{probability:.1%}')
        st.write('**Predicted class:** ' + ('Above 52.38%' if probability > 0.5 else 'At or below 52.38%'))
        st.caption('The classification threshold is 50% probability. The 52.38% threshold defines the season-level target at standard −110 odds. This output describes a team–situation season, rather than the chance of covering a single game.')
        st.dataframe(inputs, hide_index=True)
        st.download_button('Download prediction', inputs.assign(
            predicted_probability=probability, predicted_class=int(probability > .5)).to_csv(index=False),
            file_name='nfl_ats_prediction.csv', mime='text/csv')

with st.expander('Model and evaluation'):
    metrics = metadata['metrics']
    st.write('Random Forest with 3,000 trees, trained on 2019–2024 records with at least five betting decisions.')
    st.write(f"2025 accuracy: {metrics['accuracy']:.1%}. Always predict the negative class: {metrics['baseline_accuracy']:.1%}.")
    st.write(f"2025 ROC AUC: {metrics['auc']:.3f}. Brier score: {metrics['brier']:.3f}.")
    st.write('Selected by highest accuracy and lowest Brier score among the notebook classifiers. The baseline performs better. Selection used 2025 test results, so these displayed results are descriptive after selection.')
    st.write('The saved pipeline includes median imputation, standardization, one-hot encoding, and the fitted classifier.')

