from agents.state import ChatbotState, HyDEDocument, DocumentGrader, QueryRewriter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from ml.llm import get_groq_model
from core.config import settings
import json

def HyDE(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: HyDE")
    query  = state.get("translated_query", state["user_input"])
    parser = PydanticOutputParser(pydantic_object=HyDEDocument)
    sys_prompt = """You are an expert mental health counselor.
Generate TWO distinct theoretical ideal clinical responses to the patient query.
{format_instructions}
Output strictly raw JSON starting with {{ and ending with }}."""
    chain = ChatPromptTemplate.from_messages([
        ("system", sys_prompt), ("human", "{question}")
    ]) | get_groq_model()
    raw = chain.invoke({"question": query, "format_instructions": parser.get_format_instructions()})
    if hasattr(raw, "response_metadata"):
        if raw.response_metadata.get("finish_reason") in ("content_filter","safety","recitation"):
            return {"hypothetical_answers": [query,query,query], "status_update": ["HyDE blocked by safety filter."]}
    text = (raw if isinstance(raw, str) else raw.content).replace("```json","").replace("```","").strip()
    try:
        parsed = parser.parse(text); r1, r2 = parsed.response_1, parsed.response_2
    except Exception:
        try:
            d = json.loads(text); d = d.get("properties", d)
            r1, r2 = d.get("response_1", query), d.get("response_2", query)
        except Exception: r1 = r2 = query
    return {"hypothetical_answers": [query, r1, r2], "status_update": ["Generated HyDE search vectors."]}

def RAG(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: RAG")
    from agents.graph import hybrid_retriever
    queries = ([state["search_query"]] if state.get("search_query")
               else state.get("hypothetical_answers")
               or [state.get("translated_query", state["user_input"])])
    seen, unique = set(), []
    for q in queries:
        sparse_vec   = hybrid_retriever.sparse_encoder.encode_queries(q)
        valid_sparse = bool(sparse_vec and len(sparse_vec.get("indices", [])) > 0)
        if not valid_sparse:
            print(f"   [RAG] OOV query, dense-only fallback: '{q}'")
            dense_vec = hybrid_retriever.embeddings.embed_query(q)
            response  = hybrid_retriever.index.query(vector=dense_vec,
                            top_k=hybrid_retriever.top_k, include_metadata=True)
            raw_docs  = [Document(
                page_content=m["metadata"].get("text", m["metadata"].get("Context","")),
                metadata=m["metadata"]) for m in response.get("matches",[])]
        else:
            raw_docs = hybrid_retriever.invoke(q)
        for doc in raw_docs:
            t = doc.metadata.get("Response","")
            if t and t not in seen: seen.add(t); unique.append(t)
    top = unique[:10]
    return {"retrieved_docs": top, "status_update": [f"Retrieved {len(top)} docs from RAG."]}

def ReRanker(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: ReRanker")
    from rag.reranker import rerank
    docs = list(set(d.strip() for d in state.get("retrieved_docs", [])))
    if not docs:
        return {"reranked_docs": [], "status_update": ["No documents to rerank."]}
    query    = state.get("search_query") or state.get("translated_query") or state["user_input"]
    reranked = rerank(docs, query, top_k=settings.RERANKER_TOP_K)
    return {"reranked_docs": reranked, "status_update": [f"Reranked, selected top {len(reranked)} docs."]}

def grade_document(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: grade_document")
    question = state.get("search_query") or state.get("translated_query") or state["user_input"]
    docs     = state.get("reranked_docs", [])
    attempts = state.get("checking_attempts", 0)
    if not docs:
        return {"relevance_score": 0.0, "missing_info_feedback": "No docs found.",
                "checking_attempts": attempts+1, "status_update": ["Grading failed: no docs."]}
    parser = PydanticOutputParser(pydantic_object=DocumentGrader)
    sys_prompt = """Grade if retrieved docs answer the question safely. Score>=0.7 = sufficient.
{format_instructions}
Output strictly raw JSON: {{"score": 0.8, "missing_info_feedback": "..."}}"""
    chain = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Question: {question}\n\nDocuments:\n{context}\n\n{format_instructions}"),
    ]) | get_groq_model()
    raw  = chain.invoke({"question": question, "context": "\n\n".join(docs),
                          "format_instructions": parser.get_format_instructions()})
    text = (raw if isinstance(raw, str) else raw.content).replace("```json","").replace("```","").strip()
    try:
        parsed = parser.parse(text)
    except Exception:
        try:
            d = json.loads(text); d = d.get("properties", d)
            parsed = DocumentGrader(score=float(d.get("score",0.0)),
                                    missing_info_feedback=d.get("missing_info_feedback",""))
        except Exception:
            parsed = DocumentGrader(score=0.0, missing_info_feedback="Parse failure.")
    return {
        "relevance_score":       parsed.score,
        "missing_info_feedback": parsed.missing_info_feedback if parsed.score < settings.RELEVANCE_THRESHOLD else "Relevant.",
        "checking_attempts":     attempts + 1,
        "status_update":         [f"Relevance score: {parsed.score:.2f}"],
    }

def check_relevance(state: ChatbotState) -> str:
    print("--> [EDGE] Executing: check_relevance")
    if state.get("relevance_score", 0.0) >= settings.RELEVANCE_THRESHOLD:
        return "generate"
    return "rewrite" if state.get("checking_attempts", 0) < settings.MAX_REWRITE_ATTEMPTS else "generate"

def query_rewrite(state: ChatbotState) -> dict:
    print("--> [NODE] Executing: query_rewrite")
    original     = state.get("translated_query", state["user_input"])
    missing_info = state.get("missing_info_feedback", "")
    if not missing_info:
        return {"search_query": original, "status_update": ["No rewrite needed."]}
    parser = PydanticOutputParser(pydantic_object=QueryRewriter)
    sys_prompt = """Rewrite into a single effective clinical search query combining Original + Missing info.
{format_instructions}
Output strictly raw JSON: {{"search_query": "..."}}"""
    chain = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Original: {question}\nMissing: {missing}\n\n{format_instructions}"),
    ]) | get_groq_model()
    raw  = chain.invoke({"question": original, "missing": missing_info,
                          "format_instructions": parser.get_format_instructions()})
    text = (raw if isinstance(raw, str) else raw.content)
    try:
        parsed = parser.parse(text)
    except Exception:
        try:    parsed = QueryRewriter(**(json.loads(text).get("properties", json.loads(text))))
        except: parsed = QueryRewriter(search_query=f"{original} {missing_info}")
    return {"search_query": parsed.search_query, "status_update": [f"Rewritten: {parsed.search_query}"]}
