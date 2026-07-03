import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Page Config für die Web-App
st.set_page_config(page_title="Goldpreis Prognose", layout="wide")

# Titel der App
st.title("📈 Goldpreis Prognose-Dashboard (Live & Interaktiv)")
st.write("Dieses Dashboard nutzt Machine Learning (XGBoost & SVR), um den Goldpreis von morgen vorherzusagen.")

# ==========================================
# ==========================================
# 1. DATEN LADEN (LOKAL & LIVE)
# ==========================================
@st.cache_data
def lade_lokale_daten():
    # Lädt die hochgeladene CSV-Datei
    df = pd.read_csv("gold_price_cleaned.csv")
    # 'Datum' statt 'Date' verwenden, da dies so in deiner CSV steht
    df['Datum'] = pd.to_datetime(df['Datum'])
    df = df.sort_values('Datum')
    return df

try:
    df_lokal = lade_lokale_daten()
    # 'GLD_Goldpreis' statt 'Close' verwenden, da deine Spalte so heißt
    letzter_preis_lokal = df_lokal['GLD_Goldpreis'].iloc[-1]
    st.sidebar.success("Lokale CSV-Daten erfolgreich geladen!")
except Exception as e:
    st.sidebar.error(f"Fehler beim Laden der CSV: {e}")
    df_lokal = None


# Live-Daten von Yahoo Finance abrufen (GC=F ist Gold)
live_daten_geladen = False
try:
    gold_ticker = yf.Ticker("GC=F")
    historie = gold_ticker.history(period="1mo")
    if not historie.empty:
        aktueller_preis_usd = historie['Close'].iloc[-1]
        live_daten_geladen = True
        st.sidebar.success(f"Live-Kurs geladen: ${aktueller_preis_usd:.2f} USD")
    else:
        aktueller_preis_usd = 2300.0 # Fallback-Wert
except:
    st.sidebar.warning("Live-Daten konnten nicht geladen werden. Nutze Fallback-Werte.")
    aktueller_preis_usd = 2300.0

# ==========================================
# 2. MODELLE TRAINIEREN & VORHERSAGE
# ==========================================
# Vereinfachtes Training für die Live-Demo
if df_lokal is not None:
    # Feature Engineering (wir nutzen die letzten 3 Tage als Features)
    df_lokal['Lag_1'] = df_lokal['Close'].shift(1)
    df_lokal['Lag_2'] = df_lokal['Close'].shift(2)
    df_lokal['Lag_3'] = df_lokal['Close'].shift(3)
    df_train = df_lokal.dropna()
    
    X = df_train[['Lag_1', 'Lag_2', 'Lag_3']].values
    y = df_train['Close'].values
    
    # Modelle definieren und trainieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVR
    model_svr = SVR(kernel='rbf', C=100, gamma=0.1)
    model_svr.fit(X_scaled, y)
    
    # XGBoost
    model_xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model_xgb.fit(X_scaled, y)
    
    # Letzte Werte für die Prognose von morgen vorbereiten
    letzte_werte = df_lokal['Close'].iloc[-3:].values.reshape(1, -1)
    letzte_werte_scaled = scaler.transform(letzte_werte)
    
    pred_svr = model_svr.predict(letzte_werte_scaled)[0]
    pred_xgb = model_xgb.predict(letzte_werte_scaled)[0]
    
    # Durchschnitt als finale Prognose
    prognose_morgen = (pred_svr + pred_xgb) / 2
else:
    # Fallback, falls keine CSV da ist
    prognose_morgen = aktueller_preis_usd * 1.002
    pred_svr = aktueller_preis_usd * 0.999
    pred_xgb = aktueller_preis_usd * 1.005

# ==========================================
# 3. STATISTIKEN ANZEIGEN (KPIs)
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Letzter bekannter Preis (USD)", value=f"${aktueller_preis_usd:.2f}")
with col2:
    st.metric(label="Prognose morgen (SVR Model)", value=f"${pred_svr:.2f}")
with col3:
    st.metric(label="Prognose morgen (XGBoost)", value=f"${pred_xgb:.2f}")

st.info(f"💡 **Kombinierte Prognose für den nächsten Handelstag:** **${prognose_morgen:.2f} USD**")

# ==========================================
# 4. CHART ZEICHNEN
# ==========================================
st.subheader("📊 Kursverlauf & Prognose")
fig, ax = plt.subplots(figsize=(10, 4))
if df_lokal is not None:
    # Die letzten 30 Tage plotten
    daten_letzte_30 = df_lokal.tail(30)
    ax.plot(daten_letzte_30['Date'], daten_letzte_30['Close'], label="Historischer Verlauf", color="gold", linewidth=2)
    
    # Prognosepunkt hinzufügen
morgen_datum = pd.Timestamp.now() + pd.Timedelta(days=1)
ax.scatter(morgen_datum, prognose_morgen, color="red", label="Prognose Morgen", s=100, zorder=5)
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
