# 🎓 Chatbot Intelligent ENSA Fès

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF9A00?style=for-the-badge&logo=huggingface&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-CPU-0064D9?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-6.12.0-FF7C00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Système de question-réponse intelligent basé sur une architecture RAG (Retrieval-Augmented Generation) pour l'assistance des étudiants de l'ENSA Fès.**

*Pipeline : Retriever BERT ▷ Reader RoBERTa ▷ Interface Gradio*

[Démo](#démo) · [Installation](#installation) · [Architecture](#architecture) · [Résultats](#résultats)

</div>

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Base de connaissances](#base-de-connaissances)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)
- [Auteurs](#auteurs)

---

## Aperçu

Ce projet implémente un chatbot universitaire capable de répondre aux questions des étudiants sur l'ENSA Fès — filières, admissions, clubs, stages, contacts et plus. Il repose sur une architecture **RAG** combinant :

- **Retrieval sémantique** via embeddings denses + index FAISS
- **Lecture extractive** via RoBERTa fine-tuné sur SQuAD 2.0
- **Score de confiance calibré** transparent pour l'utilisateur
- **Détection hors-domaine** à 100% de précision

> Zéro hallucination par design — toutes les réponses sont extraites textuellement de la base de connaissances.

---

## Architecture

```
Question Utilisateur
        │
        ▼
┌───────────────────┐
│  Social Detector  │ ──► Réponse sociale (confiance 99%)
└───────────────────┘
        │ (non social)
        ▼
┌───────────────────┐
│  Embedding BERT   │  multi-qa-MiniLM-L6-cos-v1 (384d)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   FAISS FlatIP    │  Top-3 passages (similarité cosinus)
└───────────────────┘
        │
   cos < 0.40 ──► Fallback hors-KB (confiance 0%)
        │
        ▼
┌───────────────────┐
│  Reader RoBERTa   │  deepset/roberta-base-squad2
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Réponse + Score + Badge + Source     │
│  confiance = 0.6×cos + 0.4×QA_score  │
└───────────────────────────────────────┘
```

### Modèles utilisés

| Composante | Modèle | Détails |
|---|---|---|
| Embedding | `multi-qa-MiniLM-L6-cos-v1` | 384 dim, entraîné QA multi-domaines |
| Reader | `deepset/roberta-base-squad2` | Fine-tuné SQuAD 2.0, extractif |
| Index | `FAISS IndexFlatIP` | Similarité cosinus après normalisation L2 |

---

## Installation

### Prérequis

- Python 3.10+
- Anaconda (recommandé)
- ~2 Go RAM minimum
- **Pas de GPU requis** — fonctionne entièrement sur CPU

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/chatbot-ensa-fes.git
cd chatbot-ensa-fes
```

### 2. Créer l'environnement conda

```bash
conda create -n chatbot-ensa python=3.10
conda activate chatbot-ensa
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Contenu de requirements.txt</summary>

```
transformers==4.40.0
sentence-transformers
faiss-cpu==1.7.4
gradio==6.12.0
torch
pandas
numpy
```

</details>

### 4. Construire l'index FAISS

```bash
jupyter notebook 03_retriever.ipynb
# Exécuter toutes les cellules — génère ensa_faiss_v3.index
```

### 5. Lancer l'interface

```bash
jupyter notebook 08_interface_gradio.ipynb
# Ou directement :
python app.py
```

L'interface est accessible sur `http://localhost:7860`

---

## Utilisation

### Interface Gradio

```python
# Lancement local
demo.launch(share=False, server_port=7860)

# Lancement avec lien public (valable 72h)
demo.launch(share=True, server_port=7860)
```

### Utilisation programmatique

```python
from pipeline import chatbot_ensa

reponse, confiance, categorie, source, passages = chatbot_ensa(
    "What programs does ENSA Fes offer?"
)

print(f"Réponse   : {reponse}")
print(f"Confiance : {confiance}%")
print(f"Catégorie : {categorie}")
print(f"Source    : {source}")
```

### Système de badges de confiance

| Badge | Plage | Signification |
|---|---|---|
| 🟢 VERT | ≥ 70% | Réponse fiable |
| 🟠 ORANGE | 40–69% | Vérifier si critique |
| 🔴 ROUGE | < 40% | Contacter l'administration |
| ⚫ N/A | 0% | Question hors base |

---

## Base de connaissances

320 paires QA réparties sur 9 catégories, au format JSON inspiré de SQuAD :

```json
{
  "id": "f001",
  "category": "filieres",
  "question": "What engineering programs does ENSA Fes offer?",
  "answer": "ENSA Fes offers 11 initial training engineering specializations...",
  "context": "ENSA Fes is a public engineering school offering multiple 5-year programs."
}
```

| Catégorie | Entrées | % |
|---|---|---|
| filières | 50 | 15.6% |
| départements | 50 | 15.6% |
| admissions | 40 | 12.5% |
| scolarité | 30 | 9.4% |
| stages | 30 | 9.4% |
| clubs | 30 | 9.4% |
| contacts | 30 | 9.4% |
| événements | 30 | 9.4% |
| services | 30 | 9.4% |
| **Total** | **320** | **100%** |

---

## Résultats

### Benchmark — 15 questions de test

| Métrique | Résultat |
|---|---|
| Score de confiance moyen | **71.5%** |
| Couverture KB (0 ROUGE) | **100%** |
| Détection hors-domaine | **100% (6/6)** |
| Questions VERT (≥70%) | **8/15 (53%)** |
| Temps de réponse moyen (CPU) | **< 3 secondes** |

### Scores par catégorie

```
Clubs        ████████████████████  84%
Contacts     ███████████████████   79%
Services     ███████████████████   79%
Scolarité    █████████████████     70%
Admissions   █████████████████     68%
Filières     ████████████████      67%
Stages       ████████████████      65%
Bourses      ███████████████       63%
Départements ██████████████        57%
```

---

## Structure du projet

```
chatbot-ensa-fes/
├── 01_squad_exploration.ipynb   # Exploration du dataset SQuAD
├── 02_ensa_kb.ipynb             # Construction de la base de connaissances
├── 03_retriever.ipynb           # Module FAISS — construction de l'index
├── 04_generator.ipynb           # Module RoBERTa — reader extractif
├── 05_pipeline_rag_FINAL.ipynb  # Pipeline RAG complet
├── 06_tests_validation.ipynb    # Tests & comparaisons de modèles
├── 07_benchmark_final.ipynb     # Benchmark & métriques
├── 08_interface_gradio.ipynb    # Interface utilisateur Gradio
├── 09_ameliorations.ipynb       # Social detection + calibration confiance
│
├── ensa_kb.json                 # Base de connaissances (320 paires QA)
├── ensa_faiss_v3.index          # Index FAISS normalisé (généré)
├── squad_train_clean.json       # SQuAD v1.1 train (référence)
├── squad_val_clean.json         # SQuAD v1.1 validation (référence)
│
├── requirements.txt
└── README.md
```

---

## Auteurs

| Nom | Numéro |
|---|---|
| EL HOUTI TLÉMÇANI Yahya | No 12 |
| ALALOUCHE Walid | No 1 |

Encadré par **Mr. Gannour** — ENSA Fès, Université Sidi Mohamed Ben Abdellah

Année scolaire **2025/2026**

---

## Références

- Devlin et al. (2018) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Liu et al. (2019) — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Rajpurkar et al. (2016) — *SQuAD: 100,000+ Questions for Machine Comprehension*
- Johnson et al. (2019) — *Billion-scale similarity search with GPUs* (FAISS)

---

<div align="center">
<sub>ENSA Fès · Deep Learning / NLP · Architecture RAG · Déploiement CPU</sub>
</div>
