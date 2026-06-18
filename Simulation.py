import math
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.graph_objects as go


# ----------------------------
# Hilfsfunktionen
# ----------------------------
def to_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def normalize_curve(curve_rows):
    df = pd.DataFrame(curve_rows if curve_rows else [])
    if df.empty:
        df = pd.DataFrame({
            "laufzeit_j": [0.5, 1, 2, 3, 5, 7, 10],
            "zins_pct": [2.0, 2.1, 2.25, 2.35, 2.5, 2.6, 2.7]
        })

    if "laufzeit_j" not in df.columns:
        df["laufzeit_j"] = np.nan
    if "zins_pct" not in df.columns:
        df["zins_pct"] = np.nan

    df["laufzeit_j"] = df["laufzeit_j"].apply(to_float)
    df["zins_pct"] = df["zins_pct"].apply(to_float)
    df = df.dropna(subset=["laufzeit_j", "zins_pct"])
    df = df[df["laufzeit_j"] > 0]
    df = df.sort_values("laufzeit_j").drop_duplicates(subset=["laufzeit_j"], keep="last")

    if df.empty:
        df = pd.DataFrame({
            "laufzeit_j": [1, 2, 5, 10],
            "zins_pct": [2.0, 2.2, 2.5, 2.8]
        })

    return df.reset_index(drop=True)


def build_rate_interpolator(curve_df):
    x = curve_df["laufzeit_j"].values.astype(float)
    y = curve_df["zins_pct"].values.astype(float)

    def rate_at(t):
        t = max(1e-6, float(t))
        return float(np.interp(t, x, y, left=y[0], right=y[-1]))

    return rate_at


def discount_factor(rate_pct, t):
    r = rate_pct / 100.0
    return 1.0 / ((1.0 + r) ** float(t))


def pv_single_deal(row, rate_func):
    bezeichnung = str(row.get("bezeichnung", "")).strip()
    typ = str(row.get("typ", "")).strip().lower()

    vol_mio = to_float(row.get("vol_mio"), default=np.nan)
    coupon_pct = to_float(row.get("zins_pct"), default=np.nan)
    maturity = to_float(row.get("restlaufzeit_j"), default=np.nan)

    if np.isnan(vol_mio) or np.isnan(coupon_pct) or np.isnan(maturity) or maturity <= 0:
        return {
            "bezeichnung": bezeichnung,
            "pv": np.nan,
            "valid": False,
            "reason": "Ungültige Eingaben"
        }

    sign = 1.0
    if typ in ["passiv", "liability", "verbindlichkeit"]:
        sign = -1.0

    notional = vol_mio * 1_000_000.0
    coupon_rate = coupon_pct / 100.0

    full_years = int(math.floor(maturity))
    frac = maturity - full_years

    pv = 0.0

    for t in range(1, full_years + 1):
        cf = notional * coupon_rate
        pv += cf * discount_factor(rate_func(t), t)

    if frac > 1e-12:
        t = maturity
        cf_last = notional * coupon_rate * frac + notional
        pv += cf_last * discount_factor(rate_func(t), t)
    else:
        t = full_years
        pv += notional * discount_factor(rate_func(t), t)

    pv_signed = sign * pv

    return {
        "bezeichnung": bezeichnung if bezeichnung else "(ohne Bezeichnung)",
        "pv": pv_signed,
        "valid": True,
        "reason": ""
    }


def calculate_eve(deals_rows, curve_df, shock_bp=0.0):
    rate_base = build_rate_interpolator(curve_df)

    def rate_shocked(t):
        return rate_base(t) + shock_bp / 100.0

    deal_results = []
    total_pv = 0.0

    for r in (deals_rows or []):
        res = pv_single_deal(r, rate_shocked)
        deal_results.append(res)
        if res["valid"]:
            total_pv += res["pv"]

    return total_pv, deal_results


def format_eur(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.2f} EUR".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:,.4f} %".replace(",", "X").replace(".", ",").replace("X", ".")


# ----------------------------
# App Setup
# ----------------------------
app = Dash(__name__)
app.title = "IRRBB EVE Simulator"

# 🚀 SERVER-ANPASSUNG 1: Den Flask-Server für Gunicorn freilegen
server = app.server

app.layout = html.Div(
    style={
        "maxWidth": "1300px",
        "margin": "0 auto",
        "padding": "20px",
        "fontFamily": "Segoe UI, Tahoma, sans-serif"
    },
    children=[
        html.H1("IRRBB Simulationsmöglichkeit (EVE, +200bp, Delta Quote Neugeschäft)"),
        html.P("Formel Delta Quote Neugeschäft = (EVE +200bp - EVE Basis) / Kernkapital"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                    children=[
                        html.H3("1) Kernkapital"),
                        dcc.Input(
                            id="kernkapital",
                            type="number",
                            value=500_000_000,
                            step=1_000_000,
                            style={"width": "100%", "padding": "8px", "fontSize": "16px"}
                        ),
                        html.Div("Eingabe in EUR", style={"marginTop": "8px", "color": "#555"})
                    ]
                ),
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                    children=[
                        html.H3("2) Ergebnis"),
                        html.Div(id="kpi_eve_base", style={"marginBottom": "6px"}),
                        html.Div(id="kpi_eve_shock", style={"marginBottom": "6px"}),
                        html.Div(id="kpi_delta_eve", style={"marginBottom": "6px"}),
                        html.Div(id="kpi_delta_quote", style={"fontWeight": "bold", "marginBottom": "12px"}),

                        html.H4("Rechenweg (sichtbar)"),
                        html.Div(
                            id="rechenweg_output",
                            style={
                                "whiteSpace": "pre-line",
                                "background": "#f7f7f7",
                                "padding": "10px",
                                "borderRadius": "8px",
                                "fontSize": "14px"
                            }
                        )
                    ]
                )
            ]
        ),

        html.Hr(),

        html.H3("3) Zinskurve (Stützpunkte)"),
        dash_table.DataTable(
            id="curve_table",
            columns=[
                {"name": "Laufzeit (Jahre)", "id": "laufzeit_j", "type": "numeric"},
                {"name": "Zins (%)", "id": "zins_pct", "type": "numeric"}
            ],
            data=[
                {"laufzeit_j": 0.5, "zins_pct": 2.0},
                {"laufzeit_j": 1, "zins_pct": 2.1},
                {"laufzeit_j": 2, "zins_pct": 2.25},
                {"laufzeit_j": 3, "zins_pct": 2.35},
                {"laufzeit_j": 5, "zins_pct": 2.5},
                {"laufzeit_j": 7, "zins_pct": 2.6},
                {"laufzeit_j": 10, "zins_pct": 2.7},
            ],
            editable=True,
            row_deletable=True,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        ),

        html.Br(),

        dcc.Graph(id="curve_plot"),

        html.Hr(),

        html.H3("4) Geschäfte"),
        dash_table.DataTable(
            id="deals_table",
            columns=[
                {"name": "Bezeichnung", "id": "bezeichnung", "type": "text"},
                {
                    "name": "Typ",
                    "id": "typ",
                    "presentation": "dropdown"
                },
                {"name": "Vol. (Mio.)", "id": "vol_mio", "type": "numeric"},
                {"name": "Zins (%)", "id": "zins_pct", "type": "numeric"},
                {"name": "Restlz. (J.)", "id": "restlaufzeit_j", "type": "numeric"},
            ],
            dropdown={
                "typ": {
                    "options": [
                        {"label": "Aktiv", "value": "Aktiv"},
                        {"label": "Passiv", "value": "Passiv"}
                    ]
                }
            },
            data=[
                {"bezeichnung": "Kredit A", "typ": "Aktiv", "vol_mio": 50, "zins_pct": 3.2, "restlaufzeit_j": 4.0},
                {"bezeichnung": "Einlage B", "typ": "Passiv", "vol_mio": 30, "zins_pct": 1.8, "restlaufzeit_j": 2.5},
            ],
            editable=True,
            row_deletable=True,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        ),

        html.Br(),

        html.H3("5) Detailansicht je Geschäft"),
        dash_table.DataTable(
            id="detail_table",
            columns=[
                {"name": "Bezeichnung", "id": "bezeichnung"},
                {"name": "PV Basis (EUR)", "id": "pv_base"},
                {"name": "PV +200bp (EUR)", "id": "pv_shock"},
                {"name": "Delta (EUR)", "id": "delta"},
                {"name": "Hinweis", "id": "hinweis"},
            ],
            data=[],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        ),

        html.Div(
            "Hinweis: Dies ist ein vereinfachtes DCF-Modell für Simulationszwecke. Für regulatorische Produktivnutzung bitte Methoden und Konventionen fachlich validieren.",
            style={"marginTop": "12px", "fontSize": "13px", "color": "#666"}
        )
    ]
)


@app.callback(
    Output("kpi_eve_base", "children"),
    Output("kpi_eve_shock", "children"),
    Output("kpi_delta_eve", "children"),
    Output("kpi_delta_quote", "children"),
    Output("rechenweg_output", "children"),
    Output("curve_plot", "figure"),
    Output("detail_table", "data"),
    Input("kernkapital", "value"),
    Input("curve_table", "data"),
    Input("deals_table", "data"),
)
def recalc(kernkapital, curve_rows, deals_rows):
    curve_df = normalize_curve(curve_rows)

    eve_base, detail_base = calculate_eve(deals_rows, curve_df, shock_bp=0.0)
    eve_shock, detail_shock = calculate_eve(deals_rows, curve_df, shock_bp=200.0)

    delta_eve = eve_shock - eve_base

    kk = to_float(kernkapital, default=np.nan)
    if np.isnan(kk) or kk == 0:
        delta_quote = np.nan
        quote_formel_wert = "nicht definiert (Kernkapital = 0 oder ungültig)"
    else:
        delta_quote = delta_eve / kk
        quote_formel_wert = f"{delta_eve:,.2f} / {kk:,.2f} = {delta_quote:.8f}"

    base_txt = f"EVE Basis: {format_eur(eve_base)}"
    shock_txt = f"EVE +200bp: {format_eur(eve_shock)}"
    delta_txt = f"Delta EVE (+200bp - Basis): {format_eur(delta_eve)}"
    quote_txt = f"Delta Quote Neugeschäft: {format_pct(delta_quote)}"

    rechenweg_text = (
        "1) Baseline-Zinssatz je Laufzeit t: r_base(t) aus linearer Interpolation der Stützpunkte\n"
        "2) Shift-Zinssatz: r_shift(t) = r_base(t) + 2.00 Prozentpunkte\n"
        "3) Diskontfaktor: DF(t,r) = 1 / (1 + r/100)^t\n"
        "4) PV je Geschäft i (Baseline): PV_i_base = sign_i * Summe[CF_i,t * DF(t, r_base)]\n"
        "5) PV je Geschäft i (Shift):    PV_i_shift = sign_i * Summe[CF_i,t * DF(t, r_shift)]\n"
        "6) EVE Basis: EVE_base = Summe_i PV_i_base = " + format_eur(eve_base) + "\n"
        "7) EVE +200bp: EVE_shift = Summe_i PV_i_shift = " + format_eur(eve_shock) + "\n"
        "8) Delta EVE: Delta = EVE_shift - EVE_base = " + format_eur(delta_eve) + "\n"
        "9) Delta Quote Neugeschäft = Delta EVE / Kernkapital = " + quote_formel_wert
    )

    # Curve plot
    x = curve_df["laufzeit_j"].values
    y_base = curve_df["zins_pct"].values
    y_shock = y_base + 2.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_base, mode="lines+markers", name="Basis-Zinskurve"))
    fig.add_trace(go.Scatter(x=x, y=y_shock, mode="lines+markers", name="+200bp-Zinskurve"))
    fig.update_layout(
        template="plotly_white",
        title="Zinskurve: Basis vs. +200bp",
        xaxis_title="Laufzeit (Jahre)",
        yaxis_title="Zins (%)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Detail table
    detail_data = []
    for b, s in zip(detail_base, detail_shock):
        if not b["valid"] or not s["valid"]:
            detail_data.append({
                "bezeichnung": b["bezeichnung"],
                "pv_base": "-",
                "pv_shock": "-",
                "delta": "-",
                "hinweis": b["reason"] if b["reason"] else "Ungültige Eingaben"
            })
        else:
            d = s["pv"] - b["pv"]
            detail_data.append({
                "bezeichnung": b["bezeichnung"],
                "pv_base": format_eur(b["pv"]),
                "pv_shock": format_eur(s["pv"]),
                "delta": format_eur(d),
                "hinweis": ""
            })

    return base_txt, shock_txt, delta_txt, quote_txt, rechenweg_text, fig, detail_data


# 🚀 SERVER-ANPASSUNG 2: Start-Konfiguration für die Cloud
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8050)