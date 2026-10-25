"""
FORMULA 1 LOCAL FULL-SEASON TEST ENGINE (AUTOMATED EXTRACTION)
Developed for local workspace validation.

This script automatically unpacks the complete 2026 zip dataset, scans
through all available race folders, builds blind features matrices,
and prints the final P1 leaderboard winner accuracy scorecard.
"""

import os
import zipfile
import pandas as pd
import numpy as np
import joblib
import ast
from pathlib import Path

# =====================================================================
# 1. SETUP PATH CONSTRAINTS & VERIFICATION
# =====================================================================
ZIP_FILE_PATH = Path("f1_complete_2026_dataset.zip")
MODEL_PATH = Path("f1_regressor_multi_season_2.65.pkl")

# Verify the required model and data zip files are inside the folder
if not MODEL_PATH.exists():
    print(f"❌ Critical Error: Could not locate '{MODEL_PATH}' in this workspace.")
    print("Please make sure your model binary file sits right in this active directory.")
    exit(1)

if not ZIP_FILE_PATH.exists():
    print(f"❌ Critical Error: Could not locate '{ZIP_FILE_PATH}' in this workspace.")
    print("Please make sure you downloaded the zip file from Colab into this project folder.")
    exit(1)

# =====================================================================
# 2. AUTOMATED DATA EXTRACTION ZONE
# =====================================================================
print(f"📦 Unpacking '{ZIP_FILE_PATH}' into your workspace... Please wait...")
try:
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ Extraction complete!")
except Exception as e:
    print(f"❌ Extraction Error: Failed to unzip dataset file: {e}")
    exit(1)

# --- THE PATH FIX ---
# Check if the zip file extracted a 'content' folder wrapper, and adjust our path automatically
if Path("content/f1_data").exists():
    DATA_DIR = Path("content/f1_data")
elif Path("f1_data").exists():
    DATA_DIR = Path("f1_data")
else:
    DATA_DIR = Path(".")

print(f"📂 Actively scanning data directory: '{DATA_DIR}'")

# Ingest our fine-tuned 2.65 champion brain model file
model = joblib.load(MODEL_PATH)
print("🎉 Saved 2.65 multi-season model successfully loaded into memory!")

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

print("\n" + "=" * 60)
print("🔮 LOCAL FORECAST REPORT: BLIND 2026 SEASON EVALUATION 🔮")
print("=" * 60)

perfect_p1_hits = 0
races_processed = 0

# =====================================================================
# 3. CHRONOLOGICAL FOLDER TRAVERSAL PIPELINE
# =====================================================================
all_2026_folders = sorted(list(DATA_DIR.glob("2026_*")))

for country_folder in all_2026_folders:
    if not country_folder.is_dir(): 
        continue
        
    grand_prix_name = country_folder.name
    
    # Locate directory structures dynamically to bypass naming syntax drops from the API loops
    race_subfolders = list(country_folder.glob("*Race*"))
    quali_subfolders = list(country_folder.glob("*Quali*"))
    
    # Assure both primary weekend pillars are fully present inside this folder path
    if not race_subfolders or not quali_subfolders:
        continue
        
    race_path = race_subfolders[0] / "session_result.csv"
    drivers_path = race_subfolders[0] / "drivers.csv"
    
    # Isolate main qualification configurations vs sprint variations dynamically
    main_quali_folder = [f for f in quali_subfolders if "Sprint" not in f.name]
    sprint_quali_folder = [f for f in quali_subfolders if "Sprint" in f.name]
    sprint_race_folder = [f for f in race_subfolders if "Sprint" in f.name]
    
    if main_quali_folder:
        quali_path = main_quali_folder[0] / "session_result.csv"
    else:
        quali_path = quali_subfolders[0] / "session_result.csv"
    
    # Pass if the target CSV tracking records themselves are missing inside
    if not race_path.exists() or not quali_path.exists():
        continue

    # --- A. PARSE INPUT PACE SIGNALS (BLIND TO SUNDAY TARGET VALUES) ---
    df_quali = pd.read_csv(quali_path)
    df_live = pd.DataFrame()
    df_live["driver_number"] = pd.to_numeric(df_quali["driver_number"], errors="coerce")
    df_live["quali_position"] = pd.to_numeric(df_quali["position"], errors="coerce")
    df_live["final_quali_gap"] = df_live.apply(lambda r: extract_furthest_gap(df_quali.loc[r.name], "gap_to_leader"), axis=1)
    
    # Process Sprint data dynamically if active on this targeted weekend layout
    if sprint_quali_folder and sprint_race_folder:
        sq_p = sprint_quali_folder[0] / "session_result.csv"
        s_p = sprint_race_folder[0] / "session_result.csv"
        
        if sq_p.exists() and s_p.exists():
            df_sq = pd.read_csv(sq_p)
            df_sq["sprint_quali_position"] = pd.to_numeric(df_sq["position"], errors="coerce")
            df_sq["final_sprint_quali_gap"] = df_sq.apply(lambda r: extract_furthest_gap(df_sq.loc[r.name], "gap_to_leader"), axis=1)
            df_live = df_live.merge(df_sq[["driver_number", "sprint_quali_position", "final_sprint_quali_gap"]], on="driver_number", how="left")
            
            df_s = pd.read_csv(s_p)
            df_s["sprint_position"] = pd.to_numeric(df_s["position"], errors="coerce")
            df_live = df_live.merge(df_s[["driver_number", "sprint_position"]], on="driver_number", how="left")
            
    # Pad structural indicator defaults seamlessly if it was a standard race footprint
    if 'sprint_quali_position' not in df_live.columns:
        df_live['sprint_quali_position'] = -99
        df_live['sprint_position'] = -99
        df_live['final_sprint_quali_gap'] = 99.0
        
    df_live['sprint_position_delta'] = df_live['sprint_quali_position'] - df_live['sprint_position']
    
    if drivers_path.exists():
        df_drivers = pd.read_csv(drivers_path).drop_duplicates("driver_number")
        if 'team_name' in df_drivers.columns:
            df_live = df_live.merge(df_drivers[["driver_number", "team_name"]], on="driver_number", how="left")
    if 'team_name' not in df_live.columns: 
        df_live['team_name'] = "Unknown"

    # --- B. ENFORCE UNIFIED FEATURE STRUCTURE MAPS ---
    features = ['quali_position', 'final_quali_gap', 'sprint_quali_position', 'final_sprint_quali_gap', 'sprint_position', 'sprint_position_delta']
    X_numeric = df_live[features].fillna(99.0).copy()
    X_categorical = pd.get_dummies(df_live['team_name'], drop_first=True)
    X_live = pd.concat([X_numeric, X_categorical], axis=1)
    
    # Force the local matrix shapes to match the .pkl columns configuration footprint perfectly
    X_live = X_live.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # --- C. EXECUTE FORECAST ENGINE PREDICTIONS ---
    df_live['predicted_score'] = model.predict(X_live)
    sorted_podium = df_live.sort_values(by='predicted_score').reset_index(drop=True)
    
    # --- D. REVEAL ACTUAL TRUTH AT THE VERY END ONLY FOR EVALUATION MARKS ---
    actual_p1, actual_p2, actual_p3 = "Unknown", "Unknown", "Unknown"
    df_race = pd.read_csv(race_path)
    df_race = df_race[df_race["position"].notna()]
    df_race["driver_number"] = pd.to_numeric(df_race["driver_number"], errors="coerce")
    df_race["position"] = pd.to_numeric(df_race["position"], errors="coerce")
    
    p1_row = df_race[df_race["position"] == 1]
    p2_row = df_race[df_race["position"] == 2]
    p3_row = df_race[df_race["position"] == 3]
    
    if not p1_row.empty: actual_p1 = int(p1_row.iloc[0]["driver_number"])
    if not p2_row.empty: actual_p2 = int(p2_row.iloc[0]["driver_number"])
    if not p3_row.empty: actual_p3 = int(p3_row.iloc[0]["driver_number"])
        
    races_processed += 1
    match_status = "🎯 MATCH" if sorted_podium.loc[0, 'driver_number'] == actual_p1 else "❌ MISSED"
    if sorted_podium.loc[0, 'driver_number'] == actual_p1:  
        perfect_p1_hits += 1
        
    print(f"\n🌍 Grand Prix Weekend: {grand_prix_name}")
    print(f"  🥇 Predicted P1: Driver {int(sorted_podium.loc[0, 'driver_number'])} | Actual Winner: Driver {actual_p1} [{match_status}]")
    print(f"  🥈 Predicted P2: Driver {int(sorted_podium.loc[1, 'driver_number'])} | Actual P2: Driver {actual_p2}")
    print(f"  🥉 Predicted P3: Driver {int(sorted_podium.loc[2, 'driver_number'])} | Actual P3: Driver {actual_p3}")
    print("-" * 50)

# =====================================================================
# 4. PRINT COMPREHENSIVE PERFORMANCE SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("🏁 LOCAL WORKSPACE ACCURACY SUMMARY")
print("=" * 60)
print(f"The 2.65 model correctly forecast the absolute winner in {perfect_p1_hits} out of {races_processed} live dynamic 2026 races!")
print("=" * 60)

