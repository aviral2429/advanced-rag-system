from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided PDF document context.

If you cannot find the answer in the context, say "I don't have enough information in the document to answer this question."
Do NOT make up answers or use knowledge outside the provided context.

Context from the document:
{context}
"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever, groq_api_key, model_name="llama3-8b-8192"):
    """Build a RAG chain using LCEL (LangChain Expression Language)."""
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Store retriever for source doc retrieval
    chain = {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    return {"chain": chain, "retriever": retriever}


def ask_question(qa_bundle, question):
    """Ask a question using the QA chain and return the answer + sources."""
    chain    = qa_bundle["chain"]
    retriever = qa_bundle["retriever"]

    answer       = chain.invoke(question)
    source_docs  = retriever.invoke(question)
    return answer, source_docs
