from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import get_settings

settings = get_settings()


class QueryRewriter:
    """
    Conservative query rewriter for production RAG.

    Goal:
    - Preserve the user's original intent.
    - Preserve important entities and keywords.
    - Expand ambiguous terms when useful.
    - Do NOT invent technologies, products, people, or facts.
    - Keep the rewritten query focused enough for retrieval.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a production RAG query rewriter.

Your job is to rewrite the user's question into a focused
search query for document retrieval.

STRICT RULES:

1. Preserve the original intent exactly.
2. Preserve all names, technologies, roles, products,
   organizations, and other specific entities from the question.
3. You may add generic synonyms when they improve retrieval.
4. You may expand abbreviations when appropriate.
5. Do NOT invent specific technologies, frameworks,
   programming languages, products, companies, or facts.
6. Do NOT add examples that are not present in the user question.
7. Do NOT answer the question.
8. Do NOT remove important keywords.
9. Keep the rewritten query concise and retrieval-focused.
10. Return ONLY the rewritten query on one line.

IMPORTANT:

If the user asks:

"What technologies did Shayan use for frontend development?"

A good rewrite is:

"Shayan frontend development technologies frontend skills"

It is NOT acceptable to add:

"Angular, Vue, Svelte"

because those technologies were not mentioned by the user.

If the user asks:

"Does Shayan have experience with Java?"

A good rewrite is:

"Shayan Java experience Java programming language skills"

Do not add Python, C++, JavaScript, or other languages.

If the user asks:

"What is Shayan's primary professional role?"

A good rewrite is:

"Shayan primary professional role job title occupation"

Do not invent a specific job title.

OUTPUT:
Return ONLY the rewritten query.
No explanation.
No markdown.
No quotes.
""",
                ),
                (
                    "human",
                    """
Conversation history:
{chat_history}

Original user question:
{question}

Rewritten search query:
""",
                ),
            ]
        )

    def rewrite(
        self,
        question: str,
        chat_history: str = "",
    ) -> str:
        """
        Rewrite the user question into a focused retrieval query.
        """

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            }
        )

        rewritten = response.content

        if not isinstance(rewritten, str):
            rewritten = str(rewritten)

        rewritten = rewritten.strip()

        if not rewritten:
            return question

        return rewritten

    def rewrite_with_fallback(
        self,
        question: str,
        chat_history: str = "",
    ) -> str:
        """
        Production-safe rewrite with fallback to
        the original question if rewriting fails.
        """

        try:
            return self.rewrite(
                question=question,
                chat_history=chat_history,
            )

        except Exception as exc:
            print(
                f"Warning: Query rewrite failed ({exc}). "
                f"Using original query."
            )

            return question