"""
RAG Pipeline ENSA Fès — VERSION CORRIGÉE
Corrections :
  1. Alignement FAISS <-> données (passages.json utilisé directement)
  2. Générateur remplacé par roberta-base-squad2 (extractif, précis)
  3. Score de confiance recalibré (seuil 1.2, formule améliorée)
"""

from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
import faiss
import numpy as np
import json

print("Chargement des modèles...")

# ── Retriever ──────────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── CORRECTION 1 : charger ensa_kb.json ET reconstruire les passages alignés ──
# Le FAISS index a été construit sur passages.json (question + réponse concaténés)
# On recharge ensa_kb.json et on recrée les passages dans le même ordre
with open("ensa_kb.json", "r", encoding="utf-8") as f:
    ensa_data = json.load(f)

# passages alignés = ce sur quoi l'index FAISS a été construit
passages_alignes = [
    item["question"] + " " + item["answer"]
    for item in ensa_data
]

# Charger l'index FAISS
index = faiss.read_index("ensa_faiss.index")
print(f"  Index FAISS chargé : {index.ntotal} vecteurs")
print(f"  KB chargée : {len(ensa_data)} entrées")

# ── CORRECTION 2 : Générateur extractif (roberta-base-squad2) ─────────────────
# Remplace gpt-fr-cased-small qui générait du texte incohérent
# deepset/roberta-base-squad2 : fine-tuné sur SQuAD, répond avec un extrait précis
qa_pipeline = hf_pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)
print("  Générateur QA chargé (roberta-base-squad2)")
print("Tout est chargé !\n")


# ── Retriever ──────────────────────────────────────────────────────────────────
def retrieve(query, top_k=3):
    """Retourne les top_k passages les plus proches dans FAISS."""
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(ensa_data):
            results.append({
                "rank": i + 1,
                "score": float(distances[0][i]),   # distance L2 (plus petit = meilleur)
                "answer": ensa_data[idx]["answer"],
                "question": ensa_data[idx]["question"],
                "category": ensa_data[idx]["category"],
                "passage": passages_alignes[idx]
            })
    return results


# ── Générateur ─────────────────────────────────────────────────────────────────
def generate_response(query, context):
    """
    Utilise roberta-base-squad2 (extractif) :
    extrait la span de réponse directement depuis le contexte.
    """
    try:
        result = qa_pipeline(question=query, context=context)
        return result["answer"], result["score"]  # score entre 0 et 1
    except Exception as e:
        return f"Erreur de génération : {e}", 0.0


# ── CORRECTION 3 : Score de confiance recalibré ────────────────────────────────
# Ancienne formule : confiance = (1 - score_faiss / 1.5) * 100  → trop bas
# Nouvelle formule : on combine score FAISS + score QA du générateur
SEUIL_FAISS = 1.2   # distance L2 max acceptable (all-MiniLM : bon match < 0.8)

def calculer_confiance(score_faiss, score_qa):
    """
    score_faiss : distance L2, plus petit = meilleur (typiquement 0.3–1.2)
    score_qa    : score extractif roberta, entre 0 et 1
    Retourne un pourcentage 0–100
    """
    # Normaliser score_faiss : 0 (parfait) → 100%, 1.2 (seuil) → 0%
    conf_faiss = max(0.0, 1.0 - score_faiss / SEUIL_FAISS)
    # Combiner : 60% poids retriever + 40% poids générateur
    confiance = (0.6 * conf_faiss + 0.4 * score_qa) * 100
    return round(confiance)


# ── Pipeline principal ─────────────────────────────────────────────────────────
def chatbot_ensa(query, top_k=3, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question : {query}")
        print(f"{'='*60}")

    # ÉTAPE 1 : Retrieval
    passages = retrieve(query, top_k=top_k)
    if not passages:
        return {
            "reponse": "Aucun passage trouvé.",
            "confiance": 0,
            "source": None
        }

    meilleur = passages[0]

    if verbose:
        print(f"\n[RETRIEVER] Meilleur passage :")
        print(f"  Catégorie : {meilleur['category']}")
        print(f"  Score FAISS (L2) : {meilleur['score']:.3f}  (< {SEUIL_FAISS} = acceptable)")
        print(f"  Source : {meilleur['question']}")

    # ÉTAPE 2 : Vérification du seuil FAISS
    if meilleur["score"] > SEUIL_FAISS:
        if verbose:
            print(f"  ⚠️  Score trop élevé ({meilleur['score']:.3f} > {SEUIL_FAISS}) → réponse générique")
        return {
            "reponse": "Je n'ai pas trouvé d'information précise sur ce sujet dans ma base ENSA Fès. "
                       "Veuillez contacter l'administration à contact@ensa-fes.ma",
            "confiance": 0,
            "source": None,
            "passage": None
        }

    # ÉTAPE 3 : Construction du contexte combiné (top-3)
    contexte_combine = " ".join([p["answer"] for p in passages])

    # ÉTAPE 4 : Génération extractive
    if verbose:
        print(f"\n[GENERATOR] Extraction de la réponse depuis le contexte...")
    reponse, score_qa = generate_response(query, contexte_combine)

    # ÉTAPE 5 : Score de confiance combiné
    confiance = calculer_confiance(meilleur["score"], score_qa)

    if verbose:
        print(f"\n[RÉPONSE FINALE]")
        print(f"  Score QA    : {score_qa:.3f}")
        print(f"  Score FAISS : {meilleur['score']:.3f}")
        print(f"  Confiance   : {confiance}%")
        print(f"  Réponse     : {reponse}")

    return {
        "reponse": reponse,
        "source": meilleur["question"],
        "confiance": confiance,
        "passage": meilleur["answer"],
        "category": meilleur["category"],
        "score_faiss": round(meilleur["score"], 3),
        "score_qa": round(score_qa, 3)
    }


# ── Test rapide ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions_test = [
        "Comment intégrer l'ENSA Fès ?",
        "Quels clubs y a-t-il à l'école ?",
        "C'est quoi les débouchés après l'ENSA ?",
        "Y a-t-il des stages obligatoires ?",
        "Où se trouve l'ENSA Fès ?",
        "Quelle est la durée de la formation ?",
    ]

    resultats = []
    for q in questions_test:
        res = chatbot_ensa(q)
        resultats.append({
            "question": q,
            "reponse": res["reponse"],
            "confiance": f"{res['confiance']}%",
            "source": res.get("source", "—")
        })
        print()

    print("\n" + "="*60)
    print("TABLEAU DES RÉSULTATS")
    print("="*60)
    for r in resultats:
        print(f"\nQ : {r['question']}")
        print(f"R : {r['reponse']}")
        print(f"Confiance : {r['confiance']}  |  Source : {r['source']}")