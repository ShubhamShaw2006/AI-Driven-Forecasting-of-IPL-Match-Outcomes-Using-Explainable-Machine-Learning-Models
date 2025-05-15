import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from src.data_loader import load_data, create_features
from src.model import train_model
from src.explainability import generate_shap_explainer, plot_shap_summary

def encode_features(X, y):
    X_encoded = X.copy()
    le_dict = {}
    for col in ['team1', 'team2', 'toss_winner', 'venue', 'city']:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        le_dict[col] = le
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y)
    return X_encoded, y_encoded, le_dict, y_le

def main():
    st.title("🏏 IPL Match Outcome Predictor")
    match_info, _ = load_data()
    X, y = create_features(match_info)
    X_encoded, y_encoded, encoders, winner_encoder = encode_features(X, y)
    model, acc, report = train_model(X_encoded, y_encoded)
    st.subheader("🔍 SHAP Feature Importance Summary")
    try:
        sample_X = X_encoded.sample(100, random_state=42)
        explainer, shap_values = generate_shap_explainer(model, sample_X)
        shap_fig = plot_shap_summary(shap_values, sample_X)
        st.pyplot(shap_fig)
    except Exception as e:
        st.warning(f"SHAP couldn't run: {e}")
    st.subheader("📈 Model Performance")
    st.text(f"Accuracy: {acc:.2f}")
    st.text(report)
    team_names = sorted(X['team1'].dropna().unique())
    venue_names = sorted(X['venue'].dropna().unique())
    toss_decisions = ['bat', 'field']
    st.subheader("🤖 Make a Prediction")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", team_names)
        team2 = st.selectbox("Team 2", team_names)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", toss_decisions)
    with col2:
        venue = st.selectbox("Venue", venue_names)
    default_match_year = int(X['match_year'].median())
    default_match_month = int(X['match_month'].median())
    default_match_day = int(X['match_day'].median())
    default_season = int(X['season'].median())
    most_common_city = X['city'].mode()[0]
    default_city_code = encoders['city'].transform([most_common_city])[0]
    try:
        team1_encoded = encoders['team1'].transform([team1])[0]
        team2_encoded = encoders['team2'].transform([team2])[0]
        toss_winner_encoded = encoders['toss_winner'].transform([toss_winner])[0]
        toss_decision_encoded = 0 if toss_decision == 'bat' else 1
        venue_encoded = encoders['venue'].transform([venue])[0]
        input_df = pd.DataFrame([{
            'team1': team1_encoded,
            'team2': team2_encoded,
            'toss_winner': toss_winner_encoded,
            'toss_decision_encoded': toss_decision_encoded,
            'venue': venue_encoded,
            'city': default_city_code,
            'match_year': default_match_year,
            'match_month': default_match_month,
            'match_day': default_match_day,
            'season': default_season
        }])
        prediction_encoded = model.predict(input_df)[0]
        predicted_winner = winner_encoder.inverse_transform([prediction_encoded])[0]
        st.success(f"🏆 Predicted Winner: {predicted_winner}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()