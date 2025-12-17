import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from datetime import datetime

# Importer les fonctions Helper
from Helper.build_df import aggregate_quantities, keep_business_day

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

    # Convertir les dates et quantités
    df_raw[col_date] = pd.to_datetime(df_raw[col_date], dayfirst=True, errors="coerce")
    df_raw[col_qte] = (
        df_raw[col_qte]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .astype(float)
    )

    # Grouper par article et date
    df_daily = (
        df_raw.groupby([col_article, col_date])[col_qte]
        .sum()
        .reset_index()
        .rename(columns={col_qte: "Quantité_totale", col_date: "Période"})
    )

    # Liste des articles triés par quantité totale
    ranking = (
        df_daily.groupby("Description article")["Quantité_totale"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    articles_sorted = ranking["Description article"].tolist()

    # ÉTAPE 1: Sélection du MOIS d'abord
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

            # Créer l'histogramme
            fig_dist = go.Figure()

            fig_dist.add_trace(
                go.Histogram(
                    x=quantities_viz,
                    nbinsx=20,
                    name="Distribution",
                    marker=dict(color="rgba(52, 152, 219, 0.7)"),
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

else:
    st.info("👆 Commencez par charger un fichier CSV ou Excel contenant les colonnes: Description article, Date de livraison, Quantite")
