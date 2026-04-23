
from transformers import pipeline

print("Chargement du generator...")
generator = pipeline(
    "text-generation",
    model="asi/gpt-fr-cased-small",
    tokenizer="asi/gpt-fr-cased-small"
)

def generate_response(query, context, max_length=200):
    prompt = f"""Tu es l assistant intelligent de l ENSA Fes. Reponds a la question de l etudiant en te basant sur le contexte fourni.

Contexte : {context}

Question : {query}

Reponse :"""

    output = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )
    texte_complet = output[0]["generated_text"]
    reponse = texte_complet.split("Reponse :")[-1].strip()
    return reponse
