import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

# ==============================================================================
# STREAMLIT SEITEN-KONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Goldpreis Prognose", layout="wide", initial_sidebar_state="expanded")

st.title("📈 Goldpreis Prognose-Dashboard (Live & Interaktiv)")
st.write("Dieses Dashboard nutzt Machine Learning (XGBoost & SVR), um den Goldpreis von morgen vorherzusagen.")

# ==========================================
# ==========================================
# 1. DATEN LADEN (LOKAL & LIVE)
# ==========================================
@st.cache_data
def lade_lokale_daten():
    skript_ordner = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    file_path = os.path.join(skript_ordner, "gold_price_cleaned.csv")
    df_temp = pd.read_csv(file_path)
    
    # Datum konvertieren
    if 'Datum' in df_temp.columns:
        df_temp['Datum'] = pd.to_datetime(df_temp['Datum'])
        df_temp.set_index('Datum', inplace=True)
    elif 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp.set_index('Date', inplace=True)
        
    df_temp = df_temp.sort_index()
    return df_temp

try:
    df = lade_lokale_daten()
    letzter_preis_lokal = df['GLD_Goldpreis'].iloc[-1]
    st.sidebar.success("Lokale CSV-Daten erfolgreich geladen!")
except Exception as e:
    st.sidebar.error(f"Fehler beim Laden der CSV: {e}")
    st.stop()

# ==========================================
# 2. ZIELVARIABLE ERZEUGEN & LEERE WERTE ENTFERNEN
# ==========================================
letzter_bekannter_tag_features = df.drop(columns=['Close', 'Adj. Close'], errors='ignore').tail(1).copy()

df_model = df.copy()
df_model['Gold_Price_Tomorrow'] = df_model['GLD_Goldpreis'].shift(-1)
df_model.dropna(inplace=True)

# ==========================================
# 3. FEATURES (X) UND TARGET (y) FESTLEGEN
# ==========================================
drop_cols = ['Gold_Price_Tomorrow', 'Close', 'Adj. Close']
X = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns], errors='ignore')
y = df_model['Gold_Price_Tomorrow']

# ==========================================
# 4. ZEITBASIERTER TRAIN-TEST-SPLIT (80/20)
# ==========================================
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

# ==========================================
# 5. FEATURE-SKALIERUNG
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. VOLATILITÄTS-INDIKATOR BERECHNEN (Schwellenwert)
# ==========================================
train_returns = df_model['GLD_Goldpreis'].iloc[:split_idx].pct_change().dropna()
train_volatility = train_returns.rolling(window=5).std().dropna()
threshold = train_volatility.quantile(0.75)

# ==========================================
# 7. BASIS-MODELLE TRAINIEREN & BEWERTEN
# ==========================================
model_svr = SVR(kernel='linear', C=1.0)
model_svr.fit(X_train_scaled, y_train)

model_xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)

# Vorhersagen für Auswertung
pred_svr = model_svr.predict(X_test_scaled)
pred_xgb = model_xgb.predict(X_test)

all_returns = df_model['GLD_Goldpreis'].pct_change()
all_volatility = all_returns.rolling(window=5).std()
test_volatility = all_volatility.loc[y_test.index].bfill()

dynamic_predictions = []
for idx, current_vol in zip(y_test.index, test_volatility):
    pos = y_test.index.get_loc(idx)
    if current_vol > threshold:
        w_xgb, w_svr = 0.8, 0.2
    else:
        w_xgb, w_svr = 0.2, 0.8
    final_pred = (w_svr * pred_svr[pos]) + (w_xgb * pred_xgb[pos])
    dynamic_predictions.append(final_pred)

dynamic_predictions = np.array(dynamic_predictions)
mae_dynamic = mean_absolute_error(y_test, dynamic_predictions)

# ==============================================================================
# 8. LIVE-DATEN LADEN ODER FALLBACK AUS CSV
# ==============================================================================
gld_data = pd.DataFrame()
forex_data = pd.DataFrame()
eur_usd_kurs = 1.10  # Standard-Wechselkurs als Sicherheitsnetz

if yf is not None:
    try:
        gld_data = yf.download("GLD", period="1mo", timeout=5, progress=False)
        forex_data = yf.download("EURUSD=X", period="1d", timeout=5, progress=False)
    except Exception as e:
        st.sidebar.warning(f"Live-Verbindung fehlgeschlagen ({e}). Nutze Fallback.")

# Wechselkurs bestimmen
if not forex_data.empty:
    try:
        eur_usd_kurs = forex_data['Close'].values[-1][0] if isinstance(forex_data['Close'].values[-1], (list, np.ndarray)) else forex_data['Close'].values[-1]
    except Exception:
        eur_usd_kurs = 1.10

# Datenquelle wählen (Live-Verbindung oder Lokaler CSV-Fallback)
using_live = False
if not gld_data.empty and len(gld_data) > 5:
    if isinstance(gld_data.columns, pd.MultiIndex):
        gld_data.columns = gld_data.columns.get_level_values(0)

    if 'Close' in gld_data.columns:
        plot_data = gld_data['Close'].tail(15)
        using_live = True

if not using_live:
    plot_data = df['GLD_Goldpreis'].tail(15)

# Daten für Anzeige aufbereiten
daten_str = [d.strftime('%d.%m.') for d in plot_data.index]
preise_etf = plot_data.values.flatten().tolist()
letzter_preis_etf = preise_etf[-1]

if using_live:
    st.sidebar.success(f"Live-Kurs geladen: ${letzter_preis_etf:.2f} USD")
else:
    st.sidebar.info(f"Nutze CSV-Fallback-Kurs: ${letzter_preis_etf:.2f} USD")

# ==============================================================================
# 9. PROGNOSE-BERECHNUNG
# ==============================================================================
gld_heute_csv = letzter_bekannter_tag_features['GLD_Goldpreis'].values[0]
X_live = letzter_bekannter_tag_features.drop(columns=['Gold_Price_Tomorrow'], errors='ignore')
X_live = X_live[X_train.columns]
X_live_scaled = scaler.transform(X_live)

live_pred_svr = model_svr.predict(X_live_scaled)[0]
live_pred_xgb = model_xgb.predict(X_live)[0]

live_vol = all_volatility.iloc[-1]
if pd.isna(live_vol):
    live_vol = train_volatility.iloc[-1]

if live_vol > threshold:
    w_xgb, w_svr = 0.8, 0.2
    modus = "Volatiler Markt (Fokus: XGBoost)"
else:
    w_xgb, w_svr = 0.2, 0.8
    modus = "Stabiler Markt (Fokus: SVR)"

historische_prognose = (w_svr * live_pred_svr) + (w_xgb * live_pred_xgb)
prozentuale_veraenderung = (historische_prognose - gld_heute_csv) / gld_heute_csv

# Finale Vorhersage für den ETF-Preis
morgige_vorhersage_etf = letzter_preis_etf * (1 + prozentuale_veraenderung)

# --------------------------------------------------------------------------
# UMRECHNUNG IN EURO (€) UND LEGIERUNGEN
# --------------------------------------------------------------------------
gld_unzen_faktor = 0.091906
umrechnung_g_usd = lambda etf: (etf / gld_unzen_faktor) / 31.1035
umrechnung_g_eur = lambda etf: umrechnung_g_usd(etf) / eur_usd_kurs

# Historische Euro-Preise für die Grafik
preise_gramm_eur = [umrechnung_g_eur(p) for p in preise_etf]

# Gold-Legierungen kalkulieren
preis_morgen_24k = umrechnung_g_eur(morgige_vorhersage_etf)
preis_morgen_22k = preis_morgen_24k * 0.916
preis_morgen_18k = preis_morgen_24k * 0.750
preis_morgen_14k = preis_morgen_24k * 0.585
preis_morgen_8k  = preis_morgen_24k * 0.333
ladenpreis_morgen = preis_morgen_24k * 1.12

# Zeitsteuerung für Anzeige
heute_echt = datetime.now()
heute_wochentag = heute_echt.weekday()

if heute_wochentag in [4, 5, 6]:
    tage_bis_montag = (7 - heute_wochentag) % 7
    if tage_bis_montag == 0:
        tage_bis_montag = 1
    elif heute_wochentag == 4:
        tage_bis_montag = 3
    naechster_tag = heute_echt + timedelta(days=tage_bis_montag)
else:
    naechster_tag = heute_echt + timedelta(days=1)

morgen_datum_str = naechster_tag.strftime('%d.%m.%Y')
wochentage_de = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
wochentag_de = wochentage_de[naechster_tag.weekday()]

# ==============================================================================
# STREAMLIT UI ANZEIGE
# ==============================================================================
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.metric(label="Morgiger Zieltag", value=f"{wochentag_de}, {morgen_datum_str}")
with col_stat2:
    st.metric(label="Prognostizierter 24K Feingoldpreis", value=f"{preis_morgen_24k:.2f} EUR/g", delta=f"{prozentuale_veraenderung*100:+.2f}%")
with col_stat3:
    st.metric(label="Modell-Genauigkeit (MAE)", value=f"{mae_dynamic:.4f} USD")

# Info Boxen
st.info(f"**Marktmodus:** {modus} | **Wechselkurs:** 1 EUR = {eur_usd_kurs:.4f} USD | **Datenquelle:** {'Live (Yahoo Finance)' if using_live else 'Lokale CSV'}")

# Legierungen Tabelle
st.subheader("Goldwert nach Legierung für morgen:")
df_legierungen = pd.DataFrame({
    "Legierung": ["Feingold (24K / 999)", "Gold (22K / 916)", "Gold (18K / 750)", "Gold (14K / 585)", "Gold (8K / 333)", "Richtpreis Barren (inkl. Aufschlag)"],
    "Preis pro Gramm (€)": [
        f"{preis_morgen_24k:.2f} €",
        f"{preis_morgen_22k:.2f} €",
        f"{preis_morgen_18k:.2f} €",
        f"{preis_morgen_14k:.2f} €",
        f"{preis_morgen_8k:.2f} €",
        f"ca. {ladenpreis_morgen:.2f} €"
    ]
})
st.table(df_legierungen)

# ==============================================================================
# DIAGRAMM ERZEUGEN UND DIREKT IN STREAMLIT RENDERN
# ==============================================================================
st.subheader("Visualisierung des Kursverlaufs & der morgigen Prognose")

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(14, 6), facecolor='#121212')
ax1.set_facecolor('#0d0d0d')

color_etf = '#2980b9'      
color_gramm = '#2ecc71'    

# Linke Y-Achse: ETF
ax1.plot(daten_str, preise_etf, color=color_etf, marker='o', linewidth=2.5, alpha=0.8, label="Kursverlauf (GLD ETF)")
ax1.set_ylabel("ETF-Preis in USD (GLD)", color=color_etf, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_etf)

# Rechte Y-Achse: Grammpreis
ax2 = ax1.twinx()
ax2.plot(daten_str, preise_gramm_eur, color=color_gramm, alpha=0.0) 
ax2.set_ylabel("Goldpreis pro Gramm (24K) in EUR (€)", color=color_gramm, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_gramm)

# Labels
for i, (datum, preis_g_eur) in enumerate(zip(daten_str, preise_gramm_eur)):
    ax1.annotate(
        f"{preis_g_eur:.2f} €/g", 
        xy=(datum, preise_etf[i]), 
        xytext=(0, 10), 
        textcoords="offset points", 
        ha='center', 
        va='bottom', 
        color=color_gramm, 
        fontsize=8, 
        fontweight='semibold',
        bbox=dict(boxstyle="round,pad=0.2", fc='#121212', ec=color_gramm, lw=0.5, alpha=0.75)
    )

# Prognosepunkt
plot_daten_str = daten_str.copy()
ziel_label = f"{naechster_tag.strftime('%d.%m.')}"
plot_daten_str.append(ziel_label)
preise_etf_neu = preise_etf.copy()
preise_etf_neu.append(morgige_vorhersage_etf)

ax1.scatter(plot_daten_str[-2], letzter_preis_etf, color='#e74c3c', s=130, marker='D', zorder=5, 
            label=f"Letzter Tag ({plot_daten_str[-2]})")

ax1.scatter(plot_daten_str[-1], morgige_vorhersage_etf, color='#f1c40f', s=220, marker='*', zorder=6, 
            label=f"Prognose ({ziel_label}): {morgige_vorhersage_etf:.2f} USD")

ax1.plot([plot_daten_str[-2], plot_daten_str[-1]], [letzter_preis_etf, morgige_vorhersage_etf], 
         color='#f1c40f', linestyle='--', linewidth=2, alpha=0.8)

ax1.set_xlabel("Datum", fontsize=11, fontweight='bold', color='white')
ax1.set_xticks(range(len(plot_daten_str)))
ax1.set_xticklabels(plot_daten_str, rotation=45, ha='right', color='white')
ax1.grid(True, linestyle=':', color='#444444', alpha=0.5)

lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='lower left', frameon=True, facecolor='#121212', edgecolor='#444444')

# Rendere das Diagramm in der Streamlit-App
st.pyplot(fig)
