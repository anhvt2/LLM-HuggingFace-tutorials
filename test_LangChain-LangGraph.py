# pip install langchain langchain-community langchain-text-splitters faiss-cpu transformers sentence-transformers

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
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
# vectordb = FAISS.from_documents(docs, embedder,  # will compute embeddings
#                                 docstore=InMemoryDocstore({}),
#                                 index=None)       # LC will make a fresh IndexFlatL2
vectordb = FAISS.from_documents(docs, embedder)

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

# --------------------------------------------------------------------------------
# pip install langgraph

from typing import List, TypedDict, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Reuse `retriever` and `llm` from above.

class RAGState(TypedDict):
    question: str
    query: str                # current working query (may be rewritten)
    docs: List[Document]
    answer: Optional[str]
    needs_more: bool
    iter: int

def rewrite_query(state: RAGState) -> RAGState:
    # Lightweight rewrite to improve recall
    prompt = (
        "Rewrite the user question to improve document recall, preserving intent. "
        "Return only the rewritten query.\n\n"
        f"User question: {state['question']}"
    )
    state["query"] = llm.invoke(prompt).strip() or state["question"]
    return state

def retrieve(state: RAGState) -> RAGState:
    state["docs"] = retriever.invoke(state["query"])
    return state

def grade(state: RAGState) -> RAGState:
    # LLM-as-judge: do these snippets look sufficient?
    snippets = "\n\n".join(d.page_content[:500] for d in state["docs"])
    judgement = llm.invoke(
        "Given the question and snippets, is there enough information to answer?\n"
        "Reply ONLY YES or NO.\n\n"
        f"Question: {state['question']}\n\nSnippets:\n{snippets}"
    ).strip().upper()
    state["needs_more"] = (judgement != "YES")
    return state

def generate(state: RAGState) -> RAGState:
    context = "\n\n".join(f"[{i+1}] {d.metadata.get('source','?')}: {d.page_content}"
                          for i, d in enumerate(state["docs"]))
    prompt = (
        "Use ONLY the context to answer. If unknown, say you don't know. "
        "Cite sources with bracketed numbers like [1], [2].\n\n"
        f"Question: {state['question']}\n\nContext:\n{context}\n\nAnswer:"
    )
    state["answer"] = llm.invoke(prompt)
    return state

MAX_ITERS = 2

def expand(state: RAGState) -> RAGState:
    state["iter"] += 1
    if state["iter"] >= MAX_ITERS:   # one retry max
        state["needs_more"] = False  # break the loop and answer with best effort
        return state
    # Expand query with related keywords from current docs
    kws = ", ".join({d.metadata.get("source", "") for d in state["docs"]})
    prompt = (
        "Expand the following query by adding relevant keywords/entities, "
        "comma-separated, then return the final expanded query only.\n\n"
        f"Original: {state['question']}\nSeen: {kws}"
    )
    state["query"] = llm.invoke(prompt).strip()
    return state

graph = StateGraph(RAGState)
graph.add_node("rewrite", rewrite_query)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("expand", expand)
graph.add_node("generate", generate)

graph.set_entry_point("rewrite")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "grade")

def route(state: RAGState):
    return "expand" if state["needs_more"] else "generate"

# Instead of an unconditional edge, make expand conditional:
def after_expand(state: RAGState):
    return "retrieve" if state["iter"] < MAX_ITERS and state["needs_more"] else "generate"

def route_after_grade(state: RAGState):
    return "expand" if state.get("needs_more") else "generate"

graph.add_conditional_edges("grade", route_after_grade, {
    "expand": "expand", 
    "generate": "generate"})
graph.add_edge("expand", "retrieve")
graph.add_edge("generate", END)

app = graph.compile()

initial = {"question": "What benefits are included in Manulife Vitality?",
           "query": "", "docs": [], "answer": None, "needs_more": False, "iter": 0}

# result = app.invoke(initial)
result = app.invoke(initial, config={"recursion_limit": 60})
print(result["answer"])
