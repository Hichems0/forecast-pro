# 🚀 Guide de démarrage rapide

## Installation en 3 minutes

### Étape 1: Installation
```bash
cd forecast-dataviz-pro
pip install -r requirements.txt
```

### Étape 2: Lancement
```bash
streamlit run app.py
```

### Étape 3: Utilisation
1. Ouvrez votre navigateur à `http://localhost:8501`
2. Cliquez sur "Browse files" et sélectionnez votre fichier de données
3. Explorez les prévisions !

---

## Format des données

Votre fichier doit contenir ces colonnes:

| Description article | Date de livraison | Quantite |
|---------------------|-------------------|----------|
| VIVA LAIT 1L        | 01/01/2024        | 150      |
| LINDT CHOCOLAT      | 01/01/2024        | 75       |

**Formats acceptés**: CSV (séparateur `;`) ou Excel (.xlsx)

---

## Exemple d'utilisation

### Prévision simple (1 article)

1. Allez dans l'onglet "📦 Prévision Article Unique"
2. Recherchez votre produit dans la barre de recherche
3. Sélectionnez les dates de début et fin de forecast
4. Cliquez sur "🚀 Lancer la prévision IA"
5. Téléchargez les résultats en Excel

### Prévision batch (multiples articles)

1. Allez dans l'onglet "🚀 Prévision Batch"
2. Sélectionnez plusieurs articles
3. Configurez la période de forecast
4. Cliquez sur "🚀 Lancer le Batch Forecast"
5. Attendez la fin du traitement
6. Téléchargez le fichier Excel consolidé

---

## Astuces

💡 **Recherche rapide**: Tapez quelques lettres du nom de produit pour filtrer

💡 **Jours ouvrés**: L'horizon est automatiquement calculé en excluant dimanches et jours fériés

💡 **Batch large**: Pour >10 articles, prévoyez ~3 min/article

💡 **Validation**: Utilisez l'onglet "Validation" pour évaluer la précision des modèles

---

## Problèmes courants

### ❌ "Pas assez de données"
➡️ Solution: Votre article a moins de 50 points de données. Ajustez la fenêtre temporelle.

### ❌ "Timeout API"
➡️ Solution: L'API a pris trop de temps. Réessayez ou contactez le support.

### ❌ "Aucun jour ouvré"
➡️ Solution: Votre période contient uniquement des week-ends/jours fériés. Choisissez une période plus longue.

---

## Support

📧 support@luna-analytics.com

Bonne prévision ! 🎯
