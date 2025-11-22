from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from ai_eval.utils.utils import validate_documents


class RAG(ABC):
    """Abstract base class for Retrieval-Augmented Generation (RAG) pipelines.

    Attributes:
        llm: Language model instance.
        documents: List of Document objects.
        k: Number of documents to retrieve.
        vectorstore: Optional vectorstore for retrieval (FAISS, etc.).
    """

    def __init__(
        self,
        llm: LLM,
        documents: List[Document] | None = None,
        k: int = 3,
        vectorstore: VectorStore | None = None,
    ) -> None:
        """Initialize RAG with a language model, documents, k value, and optional vectorstore.

        Args:
            llm: Language model instance.
            documents: List of Document objects (optional).
            k: Number of documents to retrieve (default: 3).
            vectorstore: Optional vectorstore for retrieval.
        """
        self.llm = llm
        self.documents = documents if documents is not None else []
        self.k = k
        self.vectorstore = vectorstore

    def _retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        """Internal method to retrieve relevant documents for the given question.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects relevant to the question.
        """
        docs = self.retrieve(question, *args, **kwargs)
        assert isinstance(docs, list), f"relevant_docs must be list, got {type(docs)}"
        validate_documents(docs)
        return docs

    @abstractmethod
    def retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        """Implementation for retrieving relevant documents.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects.
        """
        pass

    def _generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Internal method to generate an answer using the question and context.

        Args:
            question: Input question string.
            context: Context string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated answer string.
        """
        assert isinstance(context, str), f"context must be str, got {type(context)}"
        response = self.generate(question, context, *args, **kwargs)
        assert isinstance(response, str), f"response must be str, got {type(response)}"
        return response

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Implementation for generating an answer from question and context.

        Args:
            question: Input question string.
            context: Context string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated answer string.
        """
        pass

    def answer(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, List[Document]]:
        """Answer a question by retrieving documents, building context, and generating a response.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (answer string, list of relevant Document objects).
        """
        assert isinstance(question, str), f"question must be str, got {type(question)}"
        relevant_docs = self._retrieve(question, *args, **kwargs)
        context = self.build_context(relevant_docs)
        response = self._generate(question, context, *args, **kwargs)
        return response, relevant_docs

    def build_context(self, relevant_docs: List[Document]) -> str:
        """Concatenate page content from relevant documents into a single context string.

        Args:
            relevant_docs: List of Document objects.

        Returns:
            Concatenated context string. Returns an empty string if no documents are provided.
        """
        if not relevant_docs:
            return ""
        return " ".join(
            [getattr(doc, "page_content", str(doc)) for doc in relevant_docs]
        )


# Some example implementations of RAG using TFIDF and FAISS:
# ----------------------------------------------------------
class TFIDFRAG(RAG):
    def __init__(self, llm: LLM, documents: List[Document], k: int = 3):
        super().__init__(llm, documents, k)
        # Extract texts from Document objects
        self.texts = [doc.page_content for doc in documents]
        self.retriever = TFIDFRetriever.from_texts(self.texts)

    def retrieve(self, question: str, *args, **kwargs) -> List[Document]:
        """Retrieve relevant documents for the given question."""
        relevant_docs = self.retriever.invoke(question, k=self.k)
        k = min(self.k, len(relevant_docs))
        return relevant_docs[:k]

    def generate(self, question: str, context: str, *args, **kwargs) -> str:
        """Generate an answer using the question and context."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer with facts from the context only."),
                ("human", "{input}\nContext: {context}"),
            ]
        )
        str_parser = StrOutputParser()
        answer = (prompt | self.llm | str_parser).invoke(
            {"input": question, "context": context}
        )
        return answer


class FAISSRAG(RAG):
    """RAG implementation using FAISS vectorstore for retrieval."""

    def __init__(
        self,
        llm: LLM,
        documents: List[Document],
        k: int = 3,
        vectorstore: VectorStore = None,
    ) -> None:
        super().__init__(llm, documents, k, vectorstore)

    def retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        relevant_docs = retriever.invoke(question)
        return relevant_docs

    def generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        rag_generator_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Using the following context, answer the question: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        final_prompt = rag_generator_prompt.format(context=context, question=question)
        answer = self.llm.invoke(final_prompt)
        # Ensure the output is a string
        if hasattr(answer, "content"):
            return answer.content
        return str(answer)


class RAGAnythingRAG(RAG):
    """Optimized RAG implementation using RAG-Anything for multimodal document processing.

    This wrapper integrates RAG-Anything's powerful document parsing and retrieval
    capabilities with the existing evaluation framework. It supports:
    - Multimodal content (text, images, tables, formulas)
    - Advanced document parsing with MinerU/Docling
    - Optimized retrieval with better chunking
    """

    def __init__(
        self,
        llm: LLM,
        documents: List[Document] | None = None,
        k: int = 3,
        vectorstore: VectorStore | None = None,
        rag_anything_instance: Any = None,
        processed_docs_cache: List[Document] | None = None,
    ) -> None:
        """Initialize RAG-Anything RAG wrapper.

        Args:
            llm: Language model instance for generation.
            documents: List of Document objects (optional, for compatibility).
            k: Number of documents to retrieve (default: 3).
            vectorstore: Optional vectorstore (not used, kept for compatibility).
            rag_anything_instance: Initialized RAG-Anything instance.
            processed_docs_cache: Pre-processed documents from RAG-Anything.
        """
        super().__init__(llm, documents, k, vectorstore)
        self.rag_anything = rag_anything_instance
        self.processed_docs_cache = processed_docs_cache or []

        if self.rag_anything is None:
            raise ValueError(
                "rag_anything_instance must be provided. "
                "Initialize RAG-Anything first and pass it here."
            )

    def retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve relevant documents using RAG-Anything's optimized retrieval.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments (supports top_k override).

        Returns:
            List of Document objects relevant to the question.
        """
        import asyncio

        # Allow overriding k via kwargs
        top_k = kwargs.get("top_k", self.k)

        # Use RAG-Anything's aquery method (async query)
        # Note: RAG-Anything's aquery returns a string answer, not documents
        # We need to use context extraction or query with multimodal to get documents
        try:
            # Try to use nest_asyncio for Jupyter compatibility
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                pass  # nest_asyncio not available, will try other methods

            # Check if we're in a running event loop
            try:
                asyncio.get_running_loop()
                # We're in a running loop, use nest_asyncio
                # Try with nest_asyncio (should work if applied)
                # Use aquery_with_multimodal to get context documents
                if hasattr(self.rag_anything, "aquery_with_multimodal"):
                    result = asyncio.run(
                        self.rag_anything.aquery_with_multimodal(question, mode="mix")
                    )
                else:
                    # Fallback to regular aquery
                    answer_text = asyncio.run(
                        self.rag_anything.aquery(question, mode="mix")
                    )
                    result = {"answer": answer_text, "contexts": []}
            except RuntimeError:
                # No running loop, create new one
                if hasattr(self.rag_anything, "aquery_with_multimodal"):
                    result = asyncio.run(
                        self.rag_anything.aquery_with_multimodal(question, mode="mix")
                    )
                else:
                    answer_text = asyncio.run(
                        self.rag_anything.aquery(question, mode="mix")
                    )
                    result = {"answer": answer_text, "contexts": []}
        except Exception:
            # Fallback: use cached processed docs with simple search
            results = self._fallback_retrieve(question, top_k)
            return results

        # Extract contexts from result
        # RAG-Anything's aquery returns a string answer, not documents
        # We need to get contexts from the underlying LightRAG or use the answer as context
        contexts = []

        # Try to get contexts from underlying LightRAG if available
        if (
            hasattr(self.rag_anything, "lightrag")
            and self.rag_anything.lightrag is not None
        ):
            try:
                # Try to query LightRAG directly for contexts
                lightrag_result = asyncio.run(
                    self.rag_anything.lightrag.aquery(question, mode="mix")
                )
                # LightRAG might return contexts in the result
                if isinstance(lightrag_result, dict) and "contexts" in lightrag_result:
                    contexts = lightrag_result["contexts"]
                elif isinstance(lightrag_result, str):
                    # If we only get answer, use it as context
                    contexts = [lightrag_result]
            except Exception:
                pass

        # If we still don't have contexts, use the answer as context
        if not contexts:
            if isinstance(result, dict):
                # Try to get contexts from result
                contexts = result.get("contexts", result.get("documents", []))
                if not contexts and "answer" in result:
                    # If we only have answer, create a document from it
                    contexts = [result["answer"]]
            elif isinstance(result, str):
                # Just an answer string, create document from it
                contexts = [result]
            else:
                # Unknown format, try to extract
                contexts = [str(result)]

        # Convert to LangChain Documents
        langchain_docs = []
        for i, context in enumerate(contexts[:top_k]):
            if isinstance(context, str):
                content = context
                metadata = {}
            elif hasattr(context, "content"):
                content = context.content
                metadata = getattr(context, "metadata", {}) or {}
            elif isinstance(context, dict):
                content = context.get("content", context.get("text", str(context)))
                metadata = context.get("metadata", {})
            else:
                content = str(context)
                metadata = {}

            # Create LangChain Document
            doc = Document(
                page_content=content,
                metadata={
                    **metadata,
                    "source": metadata.get("source", "raganything"),
                    "type": metadata.get("type", "text"),
                    "retrieval_index": i,
                },
            )
            langchain_docs.append(doc)

        return langchain_docs

    def _fallback_retrieve(self, question: str, top_k: int) -> List[Any]:
        """Fallback retrieval method if RAG-Anything query fails.

        This is a simple text-based search on cached documents.
        """
        # Simple keyword-based fallback
        question_lower = question.lower()
        scored_docs = []

        for doc in self.processed_docs_cache:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            # Simple keyword matching score
            score = sum(1 for word in question_lower.split() if word in content.lower())
            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Generate an answer using the question and context.

        Uses an optimized prompt template that works well with multimodal content.

        Args:
            question: Input question string.
            context: Context string (may contain text, tables, formulas).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated answer string.
        """
        # Optimized prompt for multimodal RAG
        rag_generator_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant answering questions based on the provided context. "
                "The context may include text, tables, formulas, and descriptions of images.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Provide a clear, accurate answer based solely on the context provided. "
                "If the context contains tables or formulas, reference them appropriately. "
                "If you cannot find the answer in the context, say so.\n\n"
                "Answer:"
            ),
        )
        final_prompt = rag_generator_prompt.format(context=context, question=question)
        answer = self.llm.invoke(final_prompt)
        # Ensure the output is a string
        if hasattr(answer, "content"):
            return answer.content
        return str(answer)
