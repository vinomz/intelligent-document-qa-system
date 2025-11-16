from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from .prompts import SYSTEM_PROMPT, fallback_answer
from utils.performance_calc import Metrics
import time

metrics = Metrics()

class TimedLLM:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input):
        t0 = time.time()
        out = self.llm.invoke(input)
        ms = (time.time() - t0) * 1000
        metrics.llm.record(ms)
        return out

class RAGChainBuilder:
    def __init__(self, retriever):
        self.retriever = retriever

    def timed_retriever(self, input):
        t0 = time.time()
        docs = self.retriever.invoke(input)

        total_ms = (time.time() - t0) * 1000   # embed + search

        # Subtract Gemini embed latency for this query
        if metrics.embedding.values:  
            last_embed_ms = metrics.embedding.values[-1]
            chroma_ms = total_ms - last_embed_ms
        else:
            chroma_ms = total_ms

        metrics.retrieval.record(chroma_ms)
        return docs

    def build(self):
        raw_llm  = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        timed_llm = TimedLLM(raw_llm)
        llm_runnable = RunnableLambda(lambda x: timed_llm.invoke(x))

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT),
             ("human", "{input}")]
        )

        timed_retriever_runnable = RunnableLambda(lambda x: self.timed_retriever(x))

        retrieve = RunnableParallel(
            retrieved_docs=timed_retriever_runnable,
            input=RunnablePassthrough()
        )

        def docs_to_context(data):
            docs = data["retrieved_docs"]
            context_text = "\n\n".join(doc.page_content for doc in docs)

            return {
                "context": context_text,
                "input": data["input"],
                "retrieved_docs": docs
            }

        rag_chain = (
            retrieve
            | docs_to_context
            | {
                "response": prompt | llm_runnable,
                "retrieved_docs": lambda x: x["retrieved_docs"],
            }
        )

        return rag_chain
