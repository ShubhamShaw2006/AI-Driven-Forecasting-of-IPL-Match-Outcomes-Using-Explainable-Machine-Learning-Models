import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def preprocess_match_info(df):
    all_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
    team_map = {team: idx for idx, team in enumerate(all_teams)}
    city_map = {city: idx for idx, city in enumerate(sorted(df['city'].dropna().unique()))}
    venue_map = {venue: idx for idx, venue in enumerate(sorted(df['venue'].dropna().unique()))}
    df['team1_code'] = df['team1'].map(team_map)
    df['team2_code'] = df['team2'].map(team_map)
    df['toss_winner_code'] = df['toss_winner'].map(team_map)
    df['city_code'] = df['city'].map(city_map)
    df['venue_code'] = df['venue'].map(venue_map)
    df['toss_decision_encoded'] = df['toss_decision'].map({'bat': 0, 'field': 1})
    if 'date' in df.columns:
        df['match_day'] = pd.to_datetime(df['date']).dt.day
        df['match_month'] = pd.to_datetime(df['date']).dt.month
        df['match_year'] = pd.to_datetime(df['date']).dt.year
    features = ['team1_code', 'team2_code', 'toss_winner_code', 'toss_decision_encoded',
                'city_code', 'venue_code', 'match_day', 'match_month', 'match_year']
    target = 'winner_code'
    df['winner_code'] = df['winner'].map(team_map)
    df = df.dropna(subset=features + [target])
    return df, features, target, team_map, city_map, venue_map

def train_model(df, features, target):
    X = df[features]
    y = df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    df = pd.read_csv("data/ipl_match_info_data.csv")
    df, features, target, team_map, city_map, venue_map = preprocess_match_info(df)
    model = train_model(df, features, target)
    joblib.dump(model, "model.pkl")
    joblib.dump(features, "features.pkl")
    joblib.dump(team_map, "team_map.pkl")
    joblib.dump(city_map, "city_map.pkl")
    joblib.dump(venue_map, "venue_map.pkl")
    print("Model training completed and saved!")

if __name__ == "__main__":
    main()