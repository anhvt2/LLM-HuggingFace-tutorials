"""
* Retrieval step: query → embedding → FAISS nearest neighbor.
* Augmentation step: retrieved docs + query form the augmented prompt.
* Generation step: Hugging Face LLM produces grounded response.

Improvement options: swap FAISS for Pinecone/Weaviate, use transformers.RagTokenizer and RagTokenForGeneration for full integration, or fine-tune retriever.

User Question → Embed Query → Vector DB Search → Retrieve Top-k Docs → Context = Combine Docs + Query → Prompt LLM → Generate Grounded Answer
"""


# -------- Step-1: Install dependencies
# !pip install transformers sentence-transformers faiss-cpu

# -------- Step-2: Load an embedding model

from sentence_transformers import SentenceTransformer

# Load embedding model (small, fast one for demo)
embedder = SentenceTransformer("all-mpnet-base-v2") # "sentence-transformers/all-MiniLM-L6-v2"

# -------- Step-3: Build knowledge base + index in FAISS
import faiss
import numpy as np

# Example documents (in reality: company FAQs, policies, reports)
# Demo corpus (fictionalized) — 15 entries, 1–3 sentences each
documents = [
    "Manulife Vitality overview: The program encourages healthy living by rewarding members for activities like workouts, wellness checkups, and sleep tracking. Points convert to monthly status tiers that unlock additional rewards.",
    "Gym & fitness benefits: Eligible members may receive discounts on participating gym memberships and virtual workout subscriptions. Proof of activity can be synced via wearables or submitted manually.",
    "Healthy food incentives: Members can earn weekly points or cash-back offers on select nutritious groceries at participating retailers. Eligible items and partners vary by region and are listed in the member portal.",
    "Wearable device integration: Vitality integrates with Apple Health, Google Fit, and select fitness trackers. Data syncing typically occurs daily; members should verify device permissions and account linkage.",
    "Health assessment & screenings: Completing an annual online health review and preventive screenings (e.g., blood pressure, cholesterol) earns points. Documentation may be required for verification.",
    "Status tiers & rewards: Bronze, Silver, Gold, and Platinum tiers reflect cumulative points in a policy year. Higher tiers unlock larger discounts, partner coupons, and occasional premium savings where applicable.",
    "Points earning rules: Activities have caps per day/week to ensure fairness (e.g., max steps credited per day). Points post within 24–72 hours depending on the activity and data source.",
    "Premium impacts (illustrative): In some products, maintaining higher Vitality tiers can qualify members for premium adjustments at renewal. Actual premium effects depend on product, jurisdiction, and underwriting rules.",
    "Claims & eligibility: Vitality rewards are separate from insurance claims. Participation does not guarantee claim approval; all claims follow standard policy provisions, deductibles, and exclusions.",
    "Privacy & data use: Health and activity data are used to administer program rewards, combat fraud, and improve services. Members can revoke connections at any time via their device settings and the Vitality dashboard.",
    "Policy coverage basics: Typical health or life insurance policies outline coverage, exclusions, waiting periods, and beneficiary rules. Always refer to the official policy contract for binding terms.",
    "Common exclusions example: Self-inflicted injuries, non-prescribed drug use, or experimental treatments may be excluded. Exclusions vary by product and region; consult your policy schedule.",
    "Customer support: Members can contact support via chat, phone, or email for issues like device sync failures, missing points, or account access. Typical response times are one to two business days.",
    "Appeals & corrections: If points are missing, members can submit supporting evidence (e.g., lab results, gym visit logs). Reviews usually complete within 5–10 business days depending on volume.",
    "Program changes notice: Partners, rewards, and rules may change during the year with prior notice on the member site. Archived terms are kept for reference, and key changes are highlighted in update emails."
]

# Encode documents into embeddings
doc_embeddings = embedder.encode(documents)

# Create FAISS index
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

# -------- Step-4: Retrieval function
def retrieve(query, top_k=2):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    return [documents[i] for i in indices[0]]

# -------- Step-5: Use Huggingface LLM for generation
from transformers import pipeline

# Use a small LLM for demo
# generator = pipeline("text2text-generation", model="openai-community/gpt2") # "text-generation", "facebook/bart-large-cnn"
generator = pipeline("text2text-generation", model="google/flan-t5-base")


def rag_pipeline(query):
    # Step 1: Retrieve docs
    retrieved_docs = retrieve(query)
    
    # Step 2: Construct prompt
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Step 3: Generate answer
    result = generator(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
    return result

# -------- Step-6: Test
query = "What benefits are included in Manulife Vitality?"
print(rag_pipeline(query))


