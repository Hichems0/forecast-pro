# 🎯 Enterprise Forecasting Platform

**Version 2.0.0** | Professional Edition

Une plateforme de prévision de demande alimentée par intelligence artificielle, conçue pour les entreprises nécessitant des prévisions précises et fiables.

---

## 🌟 Caractéristiques principales

### 🤖 Moteurs de prévision IA avancés
- **LSTM Bayésien**: Pour les séries temporelles régulières avec tendances complexes
- **Intermittent Forecaster**: Optimisé pour les demandes sporadiques
- **Sparse Spike Forecaster**: Spécialisé pour les pics périodiques

### 📊 Fonctionnalités professionnelles
- ✅ Prévisions individuelles avec analyse approfondie
- ✅ Traitement batch pour multiples articles
- ✅ Validation historique (backtesting)
- ✅ Support des jours ouvrés français
- ✅ Exclusion automatique des jours fériés
- ✅ Intervalles de confiance à 95%
- ✅ Visualisations interactives
- ✅ Export Excel professionnel avec totaux

### 🎨 Interface utilisateur professionnelle
- Design moderne et épuré
- Navigation intuitive
- Visualisations interactives Plotly
- Feedback utilisateur en temps réel
- Thème personnalisé pour l'entreprise

---

## 📋 Prérequis

- Python 3.8 ou supérieur
- 4 GB RAM minimum (8 GB recommandé pour batch processing)
- Connexion internet (pour l'API de prévision)

---

## 🚀 Installation

### 1. Cloner ou télécharger le projet

```bash
cd forecast-dataviz-pro
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration de l'API (optionnel)

Créez un fichier `.streamlit/secrets.toml`:

```toml
MODAL_API_URL = "https://your-api-endpoint.modal.run"
```

---

## 💻 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse: `http://localhost:8501`

### Workflow standard

1. **Import des données**
   - Formats supportés: CSV (séparateur `;`) ou Excel (.xlsx)
   - Colonnes requises: `Description article`, `Date de livraison`, `Quantite`

2. **Exploration des données**
   - Visualisation du classement des produits
   - Aperçu des données brutes
   - Statistiques descriptives

3. **Génération des prévisions**
   - **Article unique**: Analyse approfondie d'un produit
   - **Batch**: Traitement simultané de multiples produits
   - **Validation**: Évaluation de la précision historique

4. **Téléchargement des résultats**
   - Fichiers Excel avec totaux automatiques
   - Rapports consolidés pour batch
   - Métriques de performance pour validation

---

## 📊 Modules disponibles

### 📦 Prévision Article Unique

Analyse détaillée d'un article avec:
- Historique visuel complet
- Prévisions avec intervalles de confiance
- Trajectoires simulées (Monte Carlo)
- Diagnostics du modèle utilisé
- Export Excel individuel

### 🚀 Prévision Batch

Traitement en masse avec:
- Sélection multiple d'articles
- Progression en temps réel
- Gestion d'erreurs robuste
- Rapports consolidés
- Export groupé (un fichier Excel avec onglets)

### 📊 Validation Historique

Backtesting professionnel avec:
- Division train/test personnalisable
- Métriques de performance (MAE, RMSE, MAPE)
- Comparaison prévisions vs réalité
- Évaluation par article
- Export des résultats de validation

---

## 🛠️ Configuration avancée

### Paramètres configurables

Modifiez la classe `Config` dans `app.py`:

```python
class Config:
    DATA_MIN_POINTS = 50              # Minimum de points requis
    DEFAULT_TIMEOUT = 900             # Timeout API (15 min)
    BATCH_WARNING_THRESHOLD = 10      # Seuil d'alerte batch

    # Personnalisation des couleurs
    PRIMARY_COLOR = "#667eea"
    SECONDARY_COLOR = "#ff7f0e"
    # ...
```

### Thème personnalisé

Modifiez `.streamlit/config.toml` pour adapter l'apparence:

```toml
[theme]
primaryColor = "#667eea"      # Couleur principale
backgroundColor = "#f8f9fa"   # Fond de l'app
secondaryBackgroundColor = "#ffffff"  # Fond des cartes
textColor = "#2c3e50"         # Couleur du texte
```

---

## 📈 Caractéristiques techniques

### Jours ouvrés français
- Exclusion automatique des dimanches
- Prise en compte des jours fériés français
- Calcul précis des horizons de prévision

### Modèles IA
- **Routage intelligent**: Sélection automatique du meilleur modèle
- **Calibration avancée**: Shrinkage des probabilités et contrôle de masse
- **Incertitude quantifiée**: Intervalles de confiance fiables

### Performance
- Timeout adaptatif (jusqu'à 15 min par article)
- Traitement batch optimisé
- Gestion robuste des erreurs
- Sauvegarde automatique des résultats

---

## 🔧 Dépannage

### L'application ne démarre pas

```bash
# Vérifier l'installation de Streamlit
streamlit --version

# Réinstaller les dépendances
pip install -r requirements.txt --upgrade
```

### Erreurs de timeout API

- Augmentez `DEFAULT_TIMEOUT` dans Config
- Vérifiez votre connexion internet
- Réduisez le nombre d'articles en batch

### Données non chargées

- Vérifiez le format du fichier (CSV avec `;` ou Excel)
- Assurez-vous des noms de colonnes requis
- Vérifiez les dates (format DD/MM/YYYY ou DD-MM-YYYY)

---

## 📞 Support

Pour toute question ou problème:
- 📧 Email: support@luna-analytics.com
- 📚 Documentation: [lien vers documentation]
- 🐛 Rapports de bugs: [lien vers issue tracker]

---

## 📝 Changelog

### Version 2.0.0 (2025-12-09)
- ✨ Interface utilisateur professionnelle redesignée
- ✅ Support complet des jours ouvrés français
- ✅ Amélioration du traitement batch (timeout + gestion d'erreurs)
- ✅ Nouveaux diagnostics de modèles
- ✅ Export Excel amélioré avec totaux automatiques
- ✅ Validation historique avec métriques étendues

### Version 1.0.0
- Version initiale

---

## ⚖️ Licence

© 2025 Luna Analytics. Tous droits réservés.

Ce logiciel est propriétaire et confidentiel. Toute reproduction, distribution ou utilisation non autorisée est strictement interdite.

---

## 👥 Équipe

Développé par l'équipe Luna Analytics

**Technologies utilisées:**
- Streamlit (Interface)
- Plotly (Visualisations)
- Pandas & NumPy (Traitement de données)
- TensorFlow (Modèles LSTM)
- XGBoost (Modèles intermittents)
- Modal (Infrastructure cloud GPU)
