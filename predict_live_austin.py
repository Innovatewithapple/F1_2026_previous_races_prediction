"""
FORMULA 1 LIVE INFERENCE ENGINE — AUSTIN 2026
Developed for standalone live race forecasting.
"""

import pandas as pd
import numpy as np
import joblib
import ast
from pathlib import Path

# =====================================================================
# 1. DEFINE PATH TARGETS
# =====================================================================
MODEL_PATH = Path("f1_regressor_multi_season_2.65.pkl")
LIVE_CSV_PATH = Path("content/Austin.csv")

if not MODEL_PATH.exists():
    print(f"❌ Critical Error: Could not locate '{MODEL_PATH}' inside this folder.")
    exit(1)

if not LIVE_CSV_PATH.exists():
    print(f"❌ Critical Error: Could not locate data sheet at '{LIVE_CSV_PATH}'.")
    exit(1)

# Ingest our fine-tuned 2.65 model brain
model = joblib.load(MODEL_PATH)

def extract_furthest_gap(row, gap_column_name, default_flag=99.0):
    """Safely handles openF1 nested single-lap qualifying time list configurations."""
    if gap_column_name not in row.index or pd.isna(row[gap_column_name]): 
        return default_flag
    val = row[gap_column_name]
    try:
        if isinstance(val, (int, float)): 
            return float(val)
        gap_list = ast.literal_eval(str(val))
        valid_gaps = [float(x) for x in gap_list if x is not None]
        if len(valid_gaps) > 0: 
            return valid_gaps[-1]
    except: 
        pass
    return default_flag

# =====================================================================
# 2. DATA INGESTION & DEFENSIVE EXTRACTION
# =====================================================================
df_incoming = pd.read_csv(LIVE_CSV_PATH)
df_live = pd.DataFrame()

# Clean and secure structural driver configurations
df_live["driver_number"] = pd.to_numeric(df_incoming["driver_number"], errors="coerce")

if 'position' in df_incoming.columns and 'quali_position' not in df_incoming.columns:
    df_live['quali_position'] = pd.to_numeric(df_incoming['position'], errors='coerce')
else:
    df_live['quali_position'] = pd.to_numeric(df_incoming.get('quali_position', 20), errors='coerce')

if 'gap_to_leader' in df_incoming.columns:
    df_live['final_quali_gap'] = df_incoming.apply(lambda r: extract_furthest_gap(r, 'gap_to_leader'), axis=1)
else:
    df_live['final_quali_gap'] = 99.0

# --- THE ABSOLUTE PILLAR SAFETY SHIELD FOR SPRINT METRICS ---
# Defensively builds data Series objects instead of raw integers to prevent attribute crashes
if 'sprint_quali_position' in df_incoming.columns:
    df_live['sprint_quali_position'] = pd.to_numeric(df_incoming['sprint_quali_position'], errors='coerce').fillna(-99)
else:
    df_live['sprint_quali_position'] = pd.Series(-99, index=df_live.index)

if 'sprint_position' in df_incoming.columns:
    df_live['sprint_position'] = pd.to_numeric(df_incoming['sprint_position'], errors='coerce').fillna(-99)
else:
    df_live['sprint_position'] = pd.Series(-99, index=df_live.index)

if 'final_sprint_quali_gap' in df_incoming.columns:
    df_live['final_sprint_quali_gap'] = df_incoming.apply(lambda r: extract_furthest_gap(r, 'final_sprint_quali_gap'), axis=1)
else:
    df_live['final_sprint_quali_gap'] = pd.Series(99.0, index=df_live.index)

# Compute engineered long-run tyre craft features
df_live['sprint_position_delta'] = df_live['sprint_quali_position'] - df_live['sprint_position']

if 'team_name' in df_incoming.columns:
    df_live['team_name'] = df_incoming['team_name']
else:
    df_live['team_name'] = "Unknown"

# =====================================================================
# 3. MATRIX PROCESSING & INFERENCE
# =====================================================================
features = ['quali_position', 'final_quali_gap', 'sprint_quali_position', 'final_sprint_quali_gap', 'sprint_position', 'sprint_position_delta']
X_numeric = df_live[features].fillna(99.0).copy()
X_categorical = pd.get_dummies(df_live['team_name'], drop_first=True)
X_live = pd.concat([X_numeric, X_categorical], axis=1)

# Reinforce trained shape perfectly, padding missing elements with 0
X_live = X_live.reindex(columns=model.feature_names_in_, fill_value=0)

# Run forecast calculations
df_live['predicted_score'] = model.predict(X_live)
sorted_podium = df_live.sort_values(by='predicted_score').reset_index(drop=True)

# Map driver numbers to names for visual output check
driver_map = {
    1: "Max Verstappen", 
    4: "Lando Norris", 
    63: "George Russell", 
    12: "Kimi Antonelli", 
    16: "Charles Leclerc", 
    44: "Lewis Hamilton",
    11: "Sergio Perez", 
    55: "Carlos Sainz", 
    81: "Oscar Piastri",
    3: "Liam Lawson"        # 🎯 Added Liam Lawson's name profile string
}

p1_num = int(sorted_podium.loc[0, 'driver_number'])
p2_num = int(sorted_podium.loc[1, 'driver_number'])
p3_num = int(sorted_podium.loc[2, 'driver_number'])

p1_name = driver_map.get(p1_num, f"Driver {p1_num}")
p2_name = driver_map.get(p2_num, f"Driver {p2_num}")
p3_name = driver_map.get(p3_num, f"Driver {p3_num}")

p1_team = sorted_podium.loc[0, 'team_name']
p2_team = sorted_podium.loc[1, 'team_name']
p3_team = sorted_podium.loc[2, 'team_name']

# =====================================================================
# 4. PRINT OUT THE GORGEOUS INFERENCE REPORT
# =====================================================================
print("\n" + "="*65)
print("   🏆  PREDICTED LIVE PODIUM — 2026 UNITED STATES GRAND PRIX")
print("="*65)
print(f"  🥇  P1: {p1_name} ({p1_team})")
print(f"  🥈  P2: {p2_name} ({p2_team})")
print(f"  🥉  P3: {p3_name} ({p3_team})")
print("="*65 + "\n")
