from agents.state import RagState
from store import LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


class GenerationNodes:
    def __init__(self):
        self.llm = LLM().get_model()

    def generate_rag_answer(self, state: RagState):
        current_question = (
            state.get("translated_question")
            if state.get("translated_question")
            else state["question"]
        )
        retrieved_docs = state["retrieved_docs"]
        language = state["language"]
        emotion = state["emotion"]

        # بناخد الـ history للقراءة فقط وتمريره للموديل
        history = state.get("history", [])

        context = "\n\n".join(
            [
                f"Questions {i+1}: {doc.page_content}\nPossible Answers: {', '.join(doc.metadata['answers'])}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

        system_prompt = """[ROLE]
        You are an empathetic, evidence-based wellness and support assistant. Warm, validating, professional.
        [TASK] Answer the user's question directly based on the context. Do NOT repeat or include the user's question/input in your response.
        [CONSTRAINTS]
        1. Ground response in retrieved context. If empty, rely on empathy and active listening.
        2. Keep paragraphs concise.
        3. Never make a definitive clinical diagnosis. Never prescribe medications or dosages.
        4. PROFESSIONAL REFERRAL RULE: If question involves symptoms/patterns/clinical concerns,
           end with one sentence recommending a mental health professional.
        5. Plain conversational prose only. No bullet lists unless asked."""

        messages = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    f"[CONTEXT]\n{context}\n\n"
                    f"[USER EMOTION] {emotion}\n\n"
                    f"[USER QUESTION]\n{current_question}\n\n"
                    f"---"
                    f"\n[INSTRUCTION] Respond only in {language}. Write your direct response now (Do NOT repeat or quote the user question):",
                ),
            ]
        )

        final_rag_answer = (
            (messages | self.llm).invoke({"history": history}).content.strip()
        )

        # 🔥 التعديل السحري هنا 🔥
        # بنرجع فقط التحديثات الصافية، والـ Graph هيدمج الرسايل دي صح بدون تكرار
        return {
            "answer": final_rag_answer,
            "history": [
                HumanMessage(content=state["question"]),
                AIMessage(content=final_rag_answer),
            ],
        }

    def generate_normal_answer(self, state: RagState):
        current_question = (
            state.get("translated_question")
            if state.get("translated_question") is not None
            else state["question"]
        )
        language = state["language"]
        emotion = state["emotion"]
        history = state.get("history", [])

        system_prompt = f"""
            You are a helpful and empathetic mental health assistant. Answer the user's question in a supportive and understanding manner.

            ### Instructions:
            - Provide a helpful and informative response directly.
            - Do NOT append, repeat, or copy the user's query/greeting into your final answer.
            - Always maintain a compassionate and non-judgmental tone.
            - Answer in the requested language: {language}.
            - If the user is expressing a strong emotion, acknowledge it. Detected emotion: {emotion}.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    f"[USER QUESTION]\n{current_question}\n\n"
                    f"---"
                    f"\n[INSTRUCTION] Respond only in {language}. Write your direct response now (Do NOT repeat or quote the user question):",
                ),
            ]
        )

        normal_response = (
            (prompt | self.llm).invoke({"history": history}).content.strip()
        )

        # 🔥 التعديل السحري هنا أيضاً 🔥
        return {
            "answer": normal_response,
            "history": [
                HumanMessage(content=state["question"]),
                AIMessage(content=normal_response),
            ],
        }
