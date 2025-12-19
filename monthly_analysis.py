import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime

# =========================
# Fonctions utilitaires
# =========================

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

    # ========================================
    # SECTION 1 : TOP PRODUITS MENSUEL (TEMPOREL)
    # ========================================
    st.markdown("---")
    st.header("📊 Top Produits par Mois")
    st.markdown("Classement des meilleurs produits pour un ou plusieurs mois spécifiques")

    # Préparer les mois disponibles
    df_daily["Mois"] = pd.to_datetime(df_daily["Période"]).dt.to_period("M")
    available_months_ranking = sorted(df_daily["Mois"].unique(), reverse=True)  # Plus récent en premier
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
    df_selected_months = df_daily[df_daily["Mois"].isin(selected_periods)].copy()

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

    # Heatmap - Patterns de demande (calendrier)
    st.markdown("---")
    st.subheader("🔥 Heatmap - Calendrier du mois")

    # Préparer les données pour le heatmap (tous les articles du mois)
    df_heatmap = df_month_all.copy()
    df_heatmap["Date"] = pd.to_datetime(df_heatmap["Période"])
    df_heatmap["Jour_Semaine"] = df_heatmap["Date"].dt.dayofweek  # 0=Lundi, 6=Dimanche
    df_heatmap["Jour"] = df_heatmap["Date"].dt.day

    # Agréger par date complète
    heatmap_data = (
        df_heatmap.groupby("Date")["Quantité_totale"]
        .sum()
        .reset_index()
    )
    heatmap_data["Jour"] = heatmap_data["Date"].dt.day
    heatmap_data["Jour_Semaine"] = heatmap_data["Date"].dt.dayofweek

    # Filtrer pour enlever les dimanches (jour 6)
    heatmap_data = heatmap_data[heatmap_data["Jour_Semaine"] != 6].copy()

    # Créer un calendrier style matrice (semaines en lignes)
    # Obtenir le premier jour du mois pour savoir par quelle colonne commencer
    first_day = heatmap_data["Date"].min()
    first_weekday = first_day.dayofweek  # 0=Lundi, 6=Dimanche

    # Créer une matrice pour le calendrier (max 6 semaines, 6 jours - sans dimanche)
    calendar_matrix = np.full((6, 6), np.nan)  # 6 semaines max, 6 jours (Lun-Sam)
    day_numbers = np.full((6, 6), "", dtype=object)  # Pour afficher le numéro du jour

    # Remplir la matrice
    for _, row in heatmap_data.iterrows():
        day = row["Jour"]
        weekday = row["Jour_Semaine"]
        quantity = row["Quantité_totale"]

        # Calculer la semaine (ligne) et le jour de la semaine (colonne)
        days_from_first = (row["Date"] - first_day).days
        week_num = (days_from_first + first_weekday) // 7

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

    fig_heatmap.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Jour de la semaine",
        yaxis_title="",
        title=f"Calendrier de la demande - {selected_month_str}",
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
        df_heatmap_no_sunday = df_heatmap[df_heatmap["Jour_Semaine"] != 6].copy()
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
        # Jour du mois avec le plus de demande
        demande_par_jour_mois = df_heatmap.groupby("Jour")["Quantité_totale"].sum()
        jour_mois_max = demande_par_jour_mois.idxmax()
        qte_jour_max = demande_par_jour_mois.max()
        st.metric(
            "📆 Jour du mois le plus actif",
            f"Jour {jour_mois_max}",
            f"{qte_jour_max:.0f} unités"
        )

    st.caption("💡 Le heatmap aide à identifier les patterns temporels de la demande (Lundi-Samedi, dimanches exclus)")

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
