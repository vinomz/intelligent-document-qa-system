from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from .prompts import SYSTEM_PROMPT, fallback_answer

class RAGChainBuilder:
    def __init__(self, retriever):
        self.retriever = retriever

    def build(self):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT),
             ("human", "{input}")]
        )

        retrieve = RunnableParallel(
            retrieved_docs=self.retriever,
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
                "response": prompt | llm,
                "retrieved_docs": lambda x: x["retrieved_docs"],
            }
        )
        return rag_chain
