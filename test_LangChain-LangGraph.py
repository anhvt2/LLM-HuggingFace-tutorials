# pip install langchain langchain-community langchain-text-splitters faiss-cpu transformers sentence-transformers

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import numpy as np

# ---- (A) Build LC embeddings + vector store (reusing your model & FAISS) ----
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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

docs = [Document(page_content=t, metadata={"source": f"doc-{i}"}) for i, t in enumerate(documents)]

# Create FAISS from docs (LangChain wraps FAISS for you)
vectordb = FAISS.from_documents(docs, embedder,  # will compute embeddings
                                docstore=InMemoryDocstore({}),
                                index=None)       # LC will make a fresh IndexFlatL2

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.2})

# ---- (B) Wrap your FLAN-T5 generator as an LC LLM ----
gen_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=gen_pipe)

# ---- (C) Simple RAG function using LC retriever ----
SYSTEM = (
    "You are a grounded assistant. Use ONLY the provided context. "
    "If the answer is not in the context, say you don't know."
)

def rag_answer(question: str) -> str:
    # 1) retrieve
    hits = retriever.invoke(question)
    context = "\n\n".join(f"[{i+1}] {d.metadata['source']}: {d.page_content}" for i, d in enumerate(hits))

    # 2) prompt
    prompt = (
        f"{SYSTEM}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer with citations like [1], [2]:"
    )

    # 3) generate
    return llm.invoke(prompt)

print(rag_answer("What benefits are included in Manulife Vitality?"))
