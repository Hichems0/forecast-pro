import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

# =========================
# Fonctions utilitaires
# =========================

def prepare_daily_df(df,
                     col_article="Description article",
                     col_date="Date de livraison",
                     col_qte="Quantite"):
    """Prépare un DataFrame avec 1 ligne par (article, date) et quantités = 0 si absence."""
    df[col_date] = pd.to_datetime(df[col_date], dayfirst=True, errors="coerce")

    df[col_qte] = (
        df[col_qte]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .astype(float)
    )

    grouped = (
        df.groupby([col_article, col_date], as_index=False)[col_qte]
          .sum()
          .rename(columns={col_qte: "Quantité_totale"})
    )

    all_dates = pd.date_range(
        start=grouped[col_date].min(),
        end=grouped[col_date].max(),
        freq="D",
    )

    all_articles = grouped[col_article].unique()

    full_index = pd.MultiIndex.from_product(
        [all_articles, all_dates],
        names=[col_article, col_date],
    )

    result = (
        grouped
        .set_index([col_article, col_date])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return result


def aggregate_quantities(df_daily, freq="D"):
    """
    Agrège les quantités par article sur la fréquence donnée.

    freq:
      - 'D'     : journalier
      - 'W-MON' : hebdo lundi->dimanche
      - 'M'     : mensuel
    """
    if freq == "D":
        out = df_daily.copy()
        # Le df_daily a déjà les bonnes colonnes (Description article, Date de livraison, Quantité_totale)
        out = out.rename(columns={"Date de livraison": "Période"})
        return out

    agg = (
        df_daily
        .groupby(
            [
                "Description article",
                pd.Grouper(key="Date de livraison", freq=freq),
            ]
        )["Quantité_totale"]
        .sum()
        .reset_index()
        .rename(columns={"Date de livraison": "Période"})
    )
    return agg


def keep_business_day(df_agg):
    """Garde uniquement les dates avec quantité > 0"""
    # 1. Calculer la quantité totale par date
    somme_par_date = df_agg.groupby("Période")["Quantité_totale"].sum()

    # 2. Garder uniquement les dates dont la somme est > 0
    dates_valides = somme_par_date[somme_par_date > 0].index

    # 3. Filtrer le DataFrame final
    df_filtre = df_agg[df_agg["Période"].isin(dates_valides)].copy()
    return df_filtre


# =========================
# Fonctions Market View
# =========================

def setup_logger(
    name: str = "ProductSignals",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def _parse_month_yyyy_mm(s: str) -> pd.Timestamp:
    return pd.to_datetime(f"{s}-01", errors="coerce")


def _month_start(yyyy_mm: str) -> pd.Timestamp:
    return pd.Period(yyyy_mm, freq="M").to_timestamp(how="start")


def _month_end(yyyy_mm: str) -> pd.Timestamp:
    return (pd.Period(yyyy_mm, freq="M") + 1).to_timestamp(how="start")


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _linear_reg_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _pct_slope_per_day(slope: float, baseline: float) -> float:
    if baseline is None or np.isnan(baseline) or baseline == 0:
        return np.nan
    return float(slope / baseline)


def _sparkline_fig_sober(
    ts: pd.Series,
    height: int = 70,
    line_width: float = 1.2,
) -> go.Figure:
    ts = ts.dropna()
    fig = go.Figure()

    if ts.empty:
        fig.update_layout(
            template="plotly_white",
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False, fixedrange=True),
            yaxis=dict(visible=False, fixedrange=True),
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts.values,
            mode="lines",
            line=dict(width=line_width, color="black"),
            line_shape="linear",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[ts.index[-1]],
            y=[ts.values[-1]],
            mode="markers",
            marker=dict(size=5, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
    )
    return fig


@dataclass
class ProductSignalsConfig:
    col_article: str = "Description article"
    col_date: str = "Période"
    col_qty: str = "Quantité_totale"
    top_universe_n: int = 50
    min_points: int = 5


def compute_product_signals_calendar_months(
    df_daily: pd.DataFrame,
    lookback_start: str,
    lookback_end: str,
    cfg: Optional[ProductSignalsConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = ProductSignalsConfig()
    if logger is None:
        logger = setup_logger("ProductSignals")

    required = [cfg.col_article, cfg.col_date, cfg.col_qty]
    missing = [c for c in required if c not in df_daily.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df = df_daily.copy()
    df[cfg.col_date] = _safe_to_datetime(df[cfg.col_date])
    df = df.dropna(subset=[cfg.col_article, cfg.col_date])
    df[cfg.col_qty] = pd.to_numeric(df[cfg.col_qty], errors="coerce").fillna(0.0)

    start_dt = _month_start(lookback_start)
    end_excl = _month_end(lookback_end)

    df_before = df[df[cfg.col_date] < start_dt]
    df_lb = df[(df[cfg.col_date] >= start_dt) & (df[cfg.col_date] < end_excl)]

    if df_lb.empty:
        raise ValueError("Fenêtre lookback vide")

    universe = (
        df_lb.groupby(cfg.col_article, as_index=False)[cfg.col_qty]
            .sum()
            .sort_values(cfg.col_qty, ascending=False)
            .head(cfg.top_universe_n)[cfg.col_article]
            .tolist()
    )

    lb_groups = {k: g for k, g in df_lb[df_lb[cfg.col_article].isin(universe)].groupby(cfg.col_article)}
    before_groups = {k: g for k, g in df_before[df_before[cfg.col_article].isin(universe)].groupby(cfg.col_article)}

    out_rows: List[Dict] = []
    n = len(universe)

    for i, article in enumerate(universe, start=1):
        g_lb = lb_groups.get(article)
        if g_lb is None or g_lb.empty:
            continue

        g_lb = g_lb.sort_values(cfg.col_date)
        y = g_lb[cfg.col_qty].astype(float).values
        dates = g_lb[cfg.col_date].values
        n_pts = len(y)

        vol_level = float(np.std(y, ddof=1)) if n_pts >= 2 else np.nan

        if n_pts >= 3:
            prev = y[:-1].copy()
            prev[prev == 0] = 1.0
            rets = (y[1:] - y[:-1]) / prev
            vol_ret = float(np.std(rets, ddof=1)) if len(rets) >= 2 else np.nan
        else:
            vol_ret = np.nan

        if n_pts >= 2:
            t0 = pd.Timestamp(g_lb[cfg.col_date].iloc[0])
            t = (pd.to_datetime(g_lb[cfg.col_date]) - t0).dt.days.values.astype(float)
            slope_abs = _linear_reg_slope(t, y)
            baseline = float(np.mean(y)) if np.mean(y) != 0 else float(np.median(y))
            slope_pct = _pct_slope_per_day(slope_abs, baseline)
        else:
            slope_abs = np.nan
            slope_pct = np.nan

        max_lb = float(np.max(y)) if n_pts > 0 else np.nan
        min_lb = float(np.min(y)) if n_pts > 0 else np.nan

        g_b = before_groups.get(article)
        if g_b is not None and not g_b.empty:
            y_b = g_b[cfg.col_qty].astype(float).values
            max_b = float(np.max(y_b)) if len(y_b) > 0 else np.nan
            min_b = float(np.min(y_b)) if len(y_b) > 0 else np.nan
        else:
            max_b = np.nan
            min_b = np.nan

        new_high = bool(np.isfinite(max_lb) and np.isfinite(max_b) and (max_lb > max_b))
        new_low = bool(np.isfinite(min_lb) and np.isfinite(min_b) and (min_lb < min_b))

        out_rows.append({
            "article": article,
            "lookback_start": lookback_start,
            "lookback_end": lookback_end,
            "n_points_lookback": n_pts,
            "sum_qty_lookback": float(np.sum(y)),
            "mean_qty_lookback": float(np.mean(y)) if n_pts > 0 else np.nan,
            "slope_abs_per_day": slope_abs,
            "slope_pct_per_day": slope_pct,
            "volatility_level_std": vol_level,
            "volatility_return_std": vol_ret,
            "max_lookback": max_lb,
            "max_before": max_b,
            "min_lookback": min_lb,
            "min_before": min_b,
            "new_high": new_high,
            "new_low": new_low,
        })

    res = pd.DataFrame(out_rows)
    res["gainer_score"] = res["slope_pct_per_day"].replace([np.inf, -np.inf], np.nan)
    if res["gainer_score"].isna().all():
        res["gainer_score"] = res["slope_abs_per_day"]

    return res.sort_values("sum_qty_lookback", ascending=False).reset_index(drop=True)


def build_dashboard_table_solution_A(df_sig: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    df = df_sig.copy()

    df["gainer_score"] = df["gainer_score"].replace([np.inf, -np.inf], np.nan)
    gainers = df.sort_values("gainer_score", ascending=False).head(top_k)
    losers = df.sort_values("gainer_score", ascending=True).head(top_k)

    vol = df.copy()
    vol["volatility_level_std"] = pd.to_numeric(vol["volatility_level_std"], errors="coerce")
    vol["volatility_return_std"] = pd.to_numeric(vol["volatility_return_std"], errors="coerce")

    mx_lvl = float(vol["volatility_level_std"].max()) if vol["volatility_level_std"].notna().any() else 0.0
    mx_ret = float(vol["volatility_return_std"].max()) if vol["volatility_return_std"].notna().any() else 0.0

    vol["vol_level_norm"] = (vol["volatility_level_std"] / mx_lvl) if mx_lvl > 0 else 0.0
    vol["vol_ret_norm"] = (vol["volatility_return_std"] / mx_ret) if mx_ret > 0 else 0.0
    vol["vol_score"] = vol[["vol_level_norm", "vol_ret_norm"]].max(axis=1).fillna(0.0)
    most_vol = vol.sort_values("vol_score", ascending=False).head(top_k)

    nh = df[df["new_high"] == True].copy()
    nh["delta_high"] = nh["max_lookback"] - nh["max_before"]
    nh = nh.sort_values("delta_high", ascending=False).head(top_k)

    nl = df[df["new_low"] == True].copy()
    nl["delta_low"] = nl["min_before"] - nl["min_lookback"]
    nl = nl.sort_values("delta_low", ascending=False).head(top_k)

    def _pack(block_name: str, sub: pd.DataFrame, cols: list) -> pd.DataFrame:
        out = sub[[c for c in cols if c in sub.columns]].copy()
        out.insert(0, "block", block_name)
        return out

    cols_common = [
        "article",
        "sum_qty_lookback",
        "gainer_score",
        "volatility_level_std",
        "volatility_return_std",
        "new_high",
        "new_low",
    ]
    cols_vol = cols_common + ["vol_score"]

    table = pd.concat(
        [
            _pack("Top Gainers (slope)", gainers, cols_common),
            _pack("Top Losers (slope)", losers, cols_common),
            _pack("Most Volatile", most_vol, cols_vol),
            _pack("New High", nh, cols_common + ["delta_high", "max_lookback", "max_before"]),
            _pack("New Low", nl, cols_common + ["delta_low", "min_lookback", "min_before"]),
        ],
        axis=0,
        ignore_index=True,
    )

    return table


# Logger global
logger = setup_logger("MonthlyAnalysis")


# =========================
# Configuration Streamlit
# =========================

st.set_page_config(
    page_title="Analyse Mensuelle Q10/Q90",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analyse Mensuelle - Quantiles Q10/Q90")
st.markdown("""
Analysez la volatilité de la demande par mois pour détecter les périodes à forte ou faible demande.
Les quantiles Q10 et Q90 vous aident à comprendre la distribution de la demande.
""")

# Upload de fichier
uploaded_file = st.file_uploader("Sélectionner un fichier de données", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lecture du fichier
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file, sep=";")
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success(f"✅ Fichier chargé: {uploaded_file.name}")

    # Préparation des données
    col_article = "Description article"
    col_date = "Date de livraison"
    col_qte = "Quantite"

    # Créer un DataFrame complet avec toutes les dates pour tous les articles (rempli avec 0)
    df_daily = prepare_daily_df(df_raw, col_article=col_article, col_date=col_date, col_qte=col_qte)

    # ========================================
    # SECTION 1 : TOP PRODUITS MENSUEL (TEMPOREL)
    # ========================================
    st.markdown("---")
    st.header("📊 Top Produits par Mois")
    st.markdown("Classement des meilleurs produits pour un ou plusieurs mois spécifiques")

    # Préparer les mois disponibles (df_daily a "Date de livraison", pas encore "Période")
    df_daily_temp = df_daily.copy()
    df_daily_temp["Mois"] = pd.to_datetime(df_daily_temp["Date de livraison"]).dt.to_period("M")
    available_months_ranking = sorted(df_daily_temp["Mois"].unique(), reverse=True)  # Plus récent en premier
    month_options_ranking = [str(m) for m in available_months_ranking]

    # Sélection des mois
    col_month1, col_month2 = st.columns([2, 1])

    with col_month1:
        selected_months_ranking = st.multiselect(
            "📅 Sélectionner le(s) mois",
            month_options_ranking,
            default=[month_options_ranking[0]],  # Dernier mois par défaut
            help="Sélectionnez un ou plusieurs mois pour voir le top produits",
            key="months_ranking"
        )

    with col_month2:
        top_n_products = st.number_input(
            "Nombre de produits",
            min_value=5,
            max_value=50,
            value=30,
            step=5,
            help="Nombre de produits à afficher dans le top",
            key="top_n_products"
        )

    if not selected_months_ranking:
        st.warning("⚠️ Sélectionnez au moins un mois")
        st.stop()

    # Convertir les mois sélectionnés en Period
    selected_periods = [pd.Period(m, freq="M") for m in selected_months_ranking]

    # Filtrer les données pour les mois sélectionnés
    df_selected_months = df_daily_temp[df_daily_temp["Mois"].isin(selected_periods)].copy()

    # Calculer le top N produits pour les mois sélectionnés
    top_products = (
        df_selected_months.groupby("Description article")["Quantité_totale"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_products)
        .reset_index()
    )

    # Créer le graphique horizontal
    fig_top = go.Figure()

    fig_top.add_trace(go.Bar(
        x=top_products["Quantité_totale"],
        y=top_products["Description article"],
        orientation='h',
        marker=dict(
            color=top_products["Quantité_totale"],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Quantité")
        ),
        text=top_products["Quantité_totale"],
        texttemplate='%{text:.0f}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Quantité: %{x:.0f}<extra></extra>'
    ))

    # Titre du graphique
    if len(selected_months_ranking) == 1:
        title_months = selected_months_ranking[0]
    else:
        title_months = f"{len(selected_months_ranking)} mois sélectionnés"

    fig_top.update_layout(
        template="plotly_white",
        height=max(600, top_n_products * 20),  # Hauteur dynamique
        xaxis_title="Quantité totale vendue",
        yaxis_title="",
        title=f"Top {top_n_products} Produits - {title_months}",
        yaxis=dict(autorange='reversed'),  # Meilleur produit en haut
        showlegend=False
    )

    st.plotly_chart(fig_top, use_container_width=True)

    # Métriques clés
    col_metric1, col_metric2, col_metric3 = st.columns(3)

    with col_metric1:
        total_qty = top_products["Quantité_totale"].sum()
        st.metric(
            f"📦 Total Top {top_n_products}",
            f"{total_qty:,.0f} unités"
        )

    with col_metric2:
        avg_qty = top_products["Quantité_totale"].mean()
        st.metric(
            "📊 Moyenne",
            f"{avg_qty:,.0f} unités/produit"
        )

    with col_metric3:
        top_1_product = top_products.iloc[0]
        st.metric(
            "🥇 Meilleur produit",
            f"{top_1_product['Quantité_totale']:,.0f} unités",
            delta=f"{top_1_product['Description article'][:30]}..."
        )

    # Afficher les mois sélectionnés
    months_display = ", ".join(selected_months_ranking)
    st.caption(f"📅 Mois sélectionné(s): {months_display}")

    # Liste des articles triés par quantité totale (pour les sections suivantes)
    ranking = (
        df_daily.groupby("Description article")["Quantité_totale"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    articles_sorted = ranking["Description article"].tolist()

    # ========================================
    # SECTION 2 : ANALYSE MENSUELLE DÉTAILLÉE
    # ========================================
    st.markdown("---")
    st.header("📅 Analyse Mensuelle Détaillée")
    st.markdown("Analyse approfondie des quantiles Q10/Q90 par mois")

    # ÉTAPE 1: Sélection du MOIS
    st.markdown("---")
    st.markdown("### 📅 Étape 1: Sélectionner le mois à analyser")

    # Préparer les données pour obtenir les mois disponibles
    freq = "D"  # Agrégation journalière par défaut
    df_agg = aggregate_quantities(df_daily, freq=freq)
    df_agg = keep_business_day(df_agg)
    df_agg["Mois"] = pd.to_datetime(df_agg["Période"]).dt.to_period("M")

    # Sélection du mois
    available_months = sorted(df_agg["Mois"].unique())
    month_options = [str(m) for m in available_months]

    if not month_options:
        st.warning("Aucun mois disponible dans les données.")
        st.stop()

    selected_month_str = st.selectbox(
        "📅 Choisir un mois",
        month_options,
        index=len(month_options) - 1,  # Dernier mois par défaut
        key="selected_month"
    )

    selected_month = pd.Period(selected_month_str, freq="M")

    # ÉTAPE 2: Filtrer les données pour le mois choisi
    df_month_all = df_agg[df_agg["Mois"] == selected_month].copy()

    if df_month_all.empty:
        st.warning(f"Aucune donnée pour le mois {selected_month_str}")
        st.stop()

    # Obtenir tous les articles ayant des données pour ce mois
    articles_in_month = df_month_all["Description article"].unique().tolist()

    # ÉTAPE 3: Optionnel - Filtrer les articles
    st.markdown("---")
    st.markdown("### 🔍 Étape 2 (Optionnel): Filtrer les articles")

    col_filter1, col_filter2 = st.columns([2, 1])

    with col_filter1:
        search_text = st.text_input(
            "Rechercher des articles spécifiques",
            value="",
            placeholder="Laissez vide pour voir tous les articles...",
            key="search_monthly"
        )

    with col_filter2:
        top_n = st.number_input(
            "Top N articles",
            min_value=5,
            max_value=len(articles_in_month),
            value=min(20, len(articles_in_month)),
            step=5,
            help="Nombre d'articles à afficher (triés par volume)"
        )

    # Filtrage par recherche
    if search_text:
        filtered_articles_month = [a for a in articles_in_month if search_text.lower() in a.lower()]
    else:
        # Trier par volume total et prendre les top N
        volume_ranking = (
            df_month_all.groupby("Description article")["Quantité_totale"]
            .sum()
            .sort_values(ascending=False)
        )
        filtered_articles_month = volume_ranking.head(top_n).index.tolist()

    if not filtered_articles_month:
        st.warning("Aucun article ne correspond à votre recherche.")
        st.stop()

    st.caption(f"**{len(filtered_articles_month)}** article(s) à analyser pour {selected_month_str}")

    # Filtrer le DataFrame pour les articles sélectionnés
    df_month = df_month_all[df_month_all["Description article"].isin(filtered_articles_month)].copy()

    if df_month.empty:
        st.warning(f"Aucune donnée pour le mois {selected_month_str}")
        st.stop()

    # Calculer les métriques par article
    st.markdown("---")
    st.subheader(f"📊 Métriques pour {selected_month_str}")

    monthly_stats = []

    for article in filtered_articles_month:
        df_art_month = df_month[df_month["Description article"] == article].copy()

        if df_art_month.empty:
            continue

        quantities = df_art_month["Quantité_totale"]

        stats = {
            "Article": article,
            "Moyenne": quantities.mean(),
            "Médiane": quantities.median(),
            "Q10 (10ème percentile)": quantities.quantile(0.10),
            "Q90 (90ème percentile)": quantities.quantile(0.90),
            "Écart Q90-Q10": quantities.quantile(0.90) - quantities.quantile(0.10),
            "Min": quantities.min(),
            "Max": quantities.max(),
            "Total du mois": quantities.sum(),
            "Nombre de jours": len(quantities),
        }

        monthly_stats.append(stats)

    if not monthly_stats:
        st.warning("Aucune statistique calculée pour les articles sélectionnés.")
        st.stop()

    # Créer le DataFrame des statistiques
    monthly_stats_df = pd.DataFrame(monthly_stats)

    # Afficher le tableau
    st.dataframe(
        monthly_stats_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Article": st.column_config.TextColumn("Article", width="large"),
            "Moyenne": st.column_config.NumberColumn("Moyenne", format="%.1f"),
            "Médiane": st.column_config.NumberColumn("Médiane", format="%.1f"),
            "Q10 (10ème percentile)": st.column_config.NumberColumn("Q10", format="%.1f"),
            "Q90 (90ème percentile)": st.column_config.NumberColumn("Q90", format="%.1f"),
            "Écart Q90-Q10": st.column_config.NumberColumn("Écart Q90-Q10", format="%.1f"),
            "Min": st.column_config.NumberColumn("Min", format="%.0f"),
            "Max": st.column_config.NumberColumn("Max", format="%.0f"),
            "Total du mois": st.column_config.NumberColumn("Total", format="%.0f"),
            "Nombre de jours": st.column_config.NumberColumn("Jours", format="%d"),
        }
    )

    # Interprétation
    st.markdown("---")
    st.subheader("💡 Interprétation des quantiles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Q10 (10ème percentile)**
        - 10% des jours ont une demande inférieure à cette valeur
        - Utile pour identifier les périodes creuses
        """)

    with col2:
        st.markdown("""
        **Q90 (90ème percentile)**
        - 90% des jours ont une demande inférieure à cette valeur
        - 10% des jours dépassent ce seuil (pics de demande)
        - Utile pour dimensionner les stocks de sécurité
        """)

    with col3:
        st.markdown("""
        **Écart Q90-Q10**
        - Écart faible → Demande stable et prévisible
        - Écart élevé → Forte volatilité, risque de rupture ou surstock
        """)

    # Visualisation de la distribution
    st.markdown("---")
    st.subheader("📊 Visualisation de la distribution")

    if len(filtered_articles_month) > 0:
        viz_article = st.selectbox(
            "Sélectionner un article pour voir sa distribution",
            filtered_articles_month,
            key="viz_monthly_article"
        )

        df_viz = df_month[df_month["Description article"] == viz_article].copy()

        if not df_viz.empty:
            quantities_viz = df_viz["Quantité_totale"]

            # Calculer les quantiles
            q10_val = quantities_viz.quantile(0.10)
            q90_val = quantities_viz.quantile(0.90)
            median_val = quantities_viz.median()

            # Créer l'histogramme avec bins nombreux
            # Herve: 15-20 intervalles minimum, viser 30-40 pour granularité fine
            n = len(quantities_viz)

            # Calculer explicitement la taille des bins pour forcer le nombre
            target_bins = max(40, n)  # Au moins 40 bins
            data_min = quantities_viz.min()
            data_max = quantities_viz.max()
            data_range = data_max - data_min
            bin_size = data_range / target_bins if data_range > 0 else 1

            fig_dist = go.Figure()

            fig_dist.add_trace(
                go.Histogram(
                    x=quantities_viz,
                    xbins=dict(
                        start=data_min,
                        end=data_max,
                        size=bin_size
                    ),
                    name="Distribution",
                    marker=dict(color="rgba(52, 152, 219, 0.7)", line=dict(color="black", width=1)),
                )
            )

            # Ajouter les lignes de quantiles
            fig_dist.add_vline(
                x=q10_val,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Q10 = {q10_val:.1f}",
                annotation_position="top"
            )

            fig_dist.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Médiane = {median_val:.1f}",
                annotation_position="top"
            )

            fig_dist.add_vline(
                x=q90_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Q90 = {q90_val:.1f}",
                annotation_position="top"
            )

            fig_dist.update_layout(
                template="plotly_white",
                height=400,
                xaxis_title="Quantité",
                yaxis_title="Fréquence",
                title=f"Distribution de la demande - {viz_article} ({selected_month_str})",
                showlegend=True
            )

            st.plotly_chart(fig_dist, use_container_width=True)

    # Heatmap - Patterns de demande (calendrier multi-mois)
    st.markdown("---")
    st.subheader("🔥 Heatmap - Calendrier agrégé")

    # Sélection des mois pour le heatmap
    st.markdown("### Sélectionner le(s) mois à visualiser")

    available_months_heatmap = sorted(df_agg["Mois"].unique(), reverse=True)
    month_options_heatmap = [str(m) for m in available_months_heatmap]

    selected_months_heatmap = st.multiselect(
        "📅 Mois à inclure dans le calendrier",
        month_options_heatmap,
        default=[month_options_heatmap[-1]],  # Dernier mois par défaut
        help="Sélectionnez un ou plusieurs mois. Les quantités des mêmes jours seront additionnées.",
        key="months_heatmap"
    )

    if not selected_months_heatmap:
        st.warning("⚠️ Sélectionnez au moins un mois pour voir le heatmap")
        st.stop()

    # Convertir les mois sélectionnés
    selected_periods_heatmap = [pd.Period(m, freq="M") for m in selected_months_heatmap]

    # Filtrer les données pour les mois sélectionnés
    df_heatmap_multi = df_agg[df_agg["Mois"].isin(selected_periods_heatmap)].copy()
    df_heatmap_multi["Date"] = pd.to_datetime(df_heatmap_multi["Période"])
    df_heatmap_multi["Jour_Semaine"] = df_heatmap_multi["Date"].dt.dayofweek  # 0=Lundi, 6=Dimanche
    df_heatmap_multi["Jour"] = df_heatmap_multi["Date"].dt.day

    # Agréger par JOUR DU MOIS (pas par date absolue) pour additionner les mêmes jours
    # Jour 1 de tous les mois, Jour 2 de tous les mois, etc.
    heatmap_data = (
        df_heatmap_multi.groupby(["Jour", "Jour_Semaine"])["Quantité_totale"]
        .sum()  # Somme des quantités pour chaque jour du mois
        .reset_index()
    )

    # Filtrer pour enlever les dimanches (jour 6)
    heatmap_data = heatmap_data[heatmap_data["Jour_Semaine"] != 6].copy()

    # Créer un calendrier style matrice (semaines en lignes)
    # Utiliser le premier jour du premier mois sélectionné comme référence
    first_month_period = selected_periods_heatmap[0]
    first_month_date = first_month_period.to_timestamp()
    first_weekday = first_month_date.dayofweek  # 0=Lundi, 6=Dimanche

    # Créer une matrice pour le calendrier (max 6 semaines, 6 jours - sans dimanche)
    calendar_matrix = np.full((6, 6), np.nan)  # 6 semaines max, 6 jours (Lun-Sam)
    day_numbers = np.full((6, 6), "", dtype=object)  # Pour afficher le numéro du jour

    # Remplir la matrice par jour du mois
    for _, row in heatmap_data.iterrows():
        day = int(row["Jour"])
        weekday = int(row["Jour_Semaine"])
        quantity = row["Quantité_totale"]

        # Calculer la position dans le calendrier basé sur le jour du mois
        # On utilise le premier jour du mois de référence pour calculer la semaine
        day_offset = day - 1  # jour 1 = offset 0
        week_num = int((day_offset + first_weekday) // 7)

        # Ne pas traiter les dimanches (weekday 6)
        if week_num < 6 and weekday < 6:  # Sécurité et exclusion dimanche
            calendar_matrix[week_num, weekday] = quantity
            day_numbers[week_num, weekday] = str(day)

    # Supprimer les lignes complètement vides
    non_empty_rows = ~np.all(np.isnan(calendar_matrix), axis=1)
    calendar_matrix = calendar_matrix[non_empty_rows]
    day_numbers = day_numbers[non_empty_rows]

    # Labels des jours de la semaine (sans dimanche)
    jour_labels = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']

    # Labels des semaines
    week_labels = [f"Semaine {i+1}" for i in range(len(calendar_matrix))]

    # Créer les annotations avec le numéro du jour
    annotations = []
    for i in range(len(calendar_matrix)):
        for j in range(6):  # 6 jours (Lun-Sam), pas de dimanche
            if not np.isnan(calendar_matrix[i, j]):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"<b>{day_numbers[i, j]}</b><br>{int(calendar_matrix[i, j])}",
                        showarrow=False,
                        font=dict(size=11, color="white" if calendar_matrix[i, j] > np.nanpercentile(calendar_matrix, 70) else "black")
                    )
                )

    # Créer le heatmap style calendrier
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=calendar_matrix,
        x=jour_labels,
        y=week_labels,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Quantité"),
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>%{y}<br>Quantité: %{z:.0f}<extra></extra>',
        zmin=np.nanmin(calendar_matrix),
        zmax=np.nanmax(calendar_matrix)
    ))

    # Titre dynamique selon le nombre de mois
    if len(selected_months_heatmap) == 1:
        title_heatmap = f"Calendrier de la demande - {selected_months_heatmap[0]}"
    else:
        title_heatmap = f"Calendrier agrégé - {len(selected_months_heatmap)} mois"

    fig_heatmap.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Jour de la semaine",
        yaxis_title="",
        title=title_heatmap,
        annotations=annotations,
        xaxis=dict(side='top'),  # Jours en haut comme un calendrier
        yaxis=dict(autorange='reversed')  # Semaine 1 en haut
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Insights du heatmap
    col_insight1, col_insight2 = st.columns(2)

    # Mapper les jours pour les insights (sans dimanche)
    jour_labels_mapping = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi',
                           4: 'Vendredi', 5: 'Samedi'}

    with col_insight1:
        # Jour de la semaine avec le plus de demande (hors dimanche)
        df_heatmap_no_sunday = df_heatmap_multi[df_heatmap_multi["Jour_Semaine"] != 6].copy()
        demande_par_jour_semaine = df_heatmap_no_sunday.groupby("Jour_Semaine")["Quantité_totale"].sum()
        jour_max_num = demande_par_jour_semaine.idxmax()
        jour_max = jour_labels_mapping[jour_max_num]
        qte_max = demande_par_jour_semaine.max()
        st.metric(
            "📅 Jour de la semaine le plus actif",
            jour_max,
            f"{qte_max:.0f} unités totales (Lun-Sam)"
        )

    with col_insight2:
        # Jour du mois avec le plus de demande (agrégé)
        demande_par_jour_mois = df_heatmap_multi.groupby("Jour")["Quantité_totale"].sum()
        jour_mois_max = demande_par_jour_mois.idxmax()
        qte_jour_max = demande_par_jour_mois.max()
        st.metric(
            "📆 Jour du mois le plus actif (agrégé)",
            f"Jour {jour_mois_max}",
            f"{qte_jour_max:.0f} unités"
        )

    # Caption dynamique
    months_display_heatmap = ", ".join(selected_months_heatmap)
    if len(selected_months_heatmap) == 1:
        caption_text = f"💡 Calendrier pour {months_display_heatmap} (Lundi-Samedi, dimanches exclus)"
    else:
        caption_text = f"💡 Somme des quantités pour les mêmes jours sur {len(selected_months_heatmap)} mois: {months_display_heatmap}"

    st.caption(caption_text)

    # Export Excel
    st.markdown("---")
    st.subheader("📥 Téléchargement")

    monthly_buffer = io.BytesIO()
    with pd.ExcelWriter(monthly_buffer, engine='openpyxl') as writer:
        monthly_stats_df.to_excel(writer, sheet_name=f"Analyse_{selected_month_str}", index=False)

    monthly_buffer.seek(0)

    st.download_button(
        label=f"📥 Télécharger l'analyse mensuelle ({len(filtered_articles_month)} articles)",
        data=monthly_buffer,
        file_name=f"analyse_mensuelle_{selected_month_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_monthly",
        type="primary"
    )

    st.markdown("---")
    st.info("💡 Cette analyse vous aide à comprendre la volatilité de la demande et à optimiser vos stocks.")

    # ============================================================
    # MARKET VIEW
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Market View – Top 50 (Gainers / Losers / Volatility / New High-Low)")

    today_period = pd.Timestamp.today().to_period("M")
    default_end = str(today_period)
    default_start = str((today_period - 2))

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        lookback_start = st.text_input("Début (YYYY-MM)", value=default_start, key="mv_start")
    with c2:
        lookback_end = st.text_input("Fin (YYYY-MM)", value=default_end, key="mv_end")
    with c3:
        top_k = st.number_input("Top K (par bloc)", min_value=1, max_value=10, value=3, key="mv_topk")

    run_signals = st.button("✨ Calculer Market View")

    if run_signals:
        try:
            cfg_signals = ProductSignalsConfig(
                col_article="Description article",
                col_date="Période",
                col_qty="Quantité_totale",
                top_universe_n=50,
                min_points=5,
            )

            df_signals = compute_product_signals_calendar_months(
                df_daily=df_agg,
                lookback_start=lookback_start,
                lookback_end=lookback_end,
                cfg=cfg_signals,
                logger=logger,
            )

            dash_tbl = build_dashboard_table_solution_A(df_signals, top_k=int(top_k))

            view = dash_tbl.copy()

            icon_map = {
                "Top Gainers (slope)": "🟢⬆️",
                "Top Losers (slope)": "🔴⬇️",
                "Most Volatile": "🟠🌪️",
                "New High": "🟣🚀",
                "New Low": "🟣🧊",
            }
            view["icon"] = view["block"].map(icon_map).fillna("")

            def _main_value_row(r):
                b = r["block"]
                if "Gainers" in b or "Losers" in b:
                    v = r.get("gainer_score", np.nan)
                    return f"{v * 100:.2f}%/jour" if pd.notna(v) else ""
                if b == "Most Volatile":
                    v1 = r.get("volatility_level_std", np.nan)
                    v2 = r.get("volatility_return_std", np.nan)
                    vs = r.get("vol_score", np.nan)
                    s1 = f"{v1:.2f}" if pd.notna(v1) else "NA"
                    s2 = f"{v2:.2f}" if pd.notna(v2) else "NA"
                    ss = f"{vs:.2f}" if pd.notna(vs) else "NA"
                    return f"σlvl={s1} | σret={s2} | score={ss}"
                if b == "New High":
                    v = r.get("delta_high", np.nan)
                    return f"+{v:.0f}" if pd.notna(v) else ""
                if b == "New Low":
                    v = r.get("delta_low", np.nan)
                    return f"+{v:.0f}" if pd.notna(v) else ""
                return ""

            view["value_main"] = view.apply(_main_value_row, axis=1)

            show_cols = ["block", "icon", "article", "sum_qty_lookback", "value_main"]
            show_cols = [c for c in show_cols if c in view.columns]

            st.caption("Tableau unique – Top K par bloc")
            st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

            # ============================================================
            # ✅ RENDU SPARKLINES + 1 POINT / JOUR
            # ============================================================
            st.caption("Mini-graphes historiques (sparklines) – 1 point par jour sur la fenêtre lookback")

            lb_start = _parse_month_yyyy_mm(lookback_start)
            lb_end = _parse_month_yyyy_mm(lookback_end)

            if pd.isna(lb_start) or pd.isna(lb_end):
                st.warning("Lookback invalide. Utilise le format YYYY-MM (ex: 2025-10).")
            else:
                # fin de mois incluse
                lb_end = (lb_end + pd.offsets.MonthEnd(0)).normalize()

                all_days = pd.date_range(lb_start, lb_end, freq="D")

                df_lb = df_agg.copy()
                df_lb["Période"] = pd.to_datetime(df_lb["Période"], errors="coerce")
                df_lb = df_lb[(df_lb["Période"] >= lb_start) & (df_lb["Période"] <= lb_end)].copy()

                # Map article -> série daily complète (1 point par jour)
                series_map: dict[str, pd.Series] = {}
                for art, subdf in df_lb.groupby("Description article"):
                    s = (
                        subdf.sort_values("Période")
                        .set_index("Période")["Quantité_totale"]
                        .reindex(all_days, fill_value=0.0)
                    )
                    series_map[art] = s

                # Affichage par bloc, puis par ligne
                for blk in view["block"].unique():
                    st.markdown(f"**{blk}**")
                    sub = view[view["block"] == blk].copy().reset_index(drop=True)

                    for irow in range(len(sub)):
                        r = sub.loc[irow]
                        art = r["article"]
                        s_hist = series_map.get(art, pd.Series(index=all_days, data=np.zeros(len(all_days))))

                        spark = _sparkline_fig_sober(
                            s_hist,
                            height=72,
                            line_width=1.6,
                        )

                        cA, cB, cC = st.columns([7, 6, 7])
                        with cA:
                            st.write(f"{int(irow + 1)}. {r.get('icon','')} {art}")
                        with cB:
                            st.write(r.get("value_main", ""))
                        with cC:
                            st.plotly_chart(
                                spark,
                                use_container_width=True,
                                config={"displayModeBar": False, "staticPlot": True},
                            )

            # Exports
            st.download_button(
                "📥 Télécharger signaux complets (CSV)",
                data=df_signals.to_csv(index=False).encode("utf-8"),
                file_name=f"signals_{lookback_start}_to_{lookback_end}.csv",
                mime="text/csv",
                key="download_signals"
            )
            st.download_button(
                "📥 Télécharger tableau Market View (CSV)",
                data=dash_tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"market_view_{lookback_start}_to_{lookback_end}.csv",
                mime="text/csv",
                key="download_market_view"
            )

        except Exception as e:
            logger.exception("Erreur Market View: %s", str(e))
            st.error(f"⚠️ Erreur Market View: {str(e)}")

else:
    st.info("👆 Commencez par charger un fichier CSV ou Excel contenant les colonnes: Description article, Date de livraison, Quantite")
