import requests
import pandas as pd
from pathlib import Path

API_URL = "https://api.openf1.org/v1"
YEAR = 2026
COUNTRY = "United States"

print(f"🔎 Searching OpenF1 API for live {YEAR} {COUNTRY} Grand Prix sessions...")

# 1. Look up the session keys for the Austin weekend
session_res = requests.get(f"{API_URL}/sessions", params={"year": YEAR, "country_name": COUNTRY}, timeout=30)
session_res.raise_for_status()
sessions_df = pd.DataFrame(session_res.json())

# Isolate the main Qualifying session key for the Sunday grid setup
quali_session = sessions_df[sessions_df["session_name"].str.contains("Qualifying", case=False, na=False) & 
                            ~sessions_df["session_name"].str.contains("Sprint", case=False, na=False)]

if quali_session.empty:
    print("❌ Error: Main Qualifying session not found in the API database yet.")
else:
    session_key = int(quali_session.iloc[0]["session_key"])
    print(f"✅ Found Qualifying Session! Key: {session_key}. Fetching Saturday grids...")
    
    # 2. Fetch the live qualifying metrics (input signals)
    result_res = requests.get(f"{API_URL}/session_result", params={"session_key": session_key}, timeout=45)
    drivers_res = requests.get(f"{API_URL}/drivers", params={"session_key": session_key}, timeout=45)
    
    df_res = pd.DataFrame(result_res.json())
    df_drv = pd.DataFrame(drivers_res.json()).drop_duplicates("driver_number")
    
    # Merge the team names into the qualifying dataframe cleanly
    if not df_res.empty and 'team_name' in df_drv.columns:
        df_final = df_res.merge(df_drv[['driver_number', 'team_name']], on='driver_number', how='left')
    else:
        df_final = df_res
        
    # Save the real live data sheet out to your workspace drive
    output_filename = "/Users/himanshuvyas/Desktop/F1_2026_previous_races_prediction/content/Austin.csv"
    df_final.to_csv(output_filename, index=False)
    
    print("\n" + "="*60)
    print(f"🎉 SUCCESS! Real Austin Qualifying data downloaded as: '{output_filename}'")
    print("="*60)
    print(df_final[['position', 'driver_number', 'team_name']].head(5))
