
from __future__ import annotations

from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import get_settings

settings = get_settings()


class MemoryCandidate(BaseModel):
    """
    A single durable fact explicitly provided by the user.
    """

    key: str = Field(
        description=(
            "Stable memory key such as name, university, degree, "
            "job_title, skill, or preference."
        )
    )

    value: str = Field(
        description="The exact durable fact provided by the user."
    )

    memory_type: Literal[
        "profile",
        "education",
        "career",
        "skill",
        "preference",
        "other",
    ] = Field(
        description="Category of the durable user fact."
    )


class MemoryExtractionResult(BaseModel):
    """
    Structured result returned by the memory extractor.
    """

    memories: list[MemoryCandidate] = Field(
        default_factory=list,
        description=(
            "Durable facts explicitly provided by the user. "
            "Return an empty list when there are no new durable facts."
        ),
    )


class MemoryExtractor:
    """
    Extracts explicit, durable user facts from a user message.

    Important:
    - Extract only information explicitly stated by the user.
    - Never infer facts.
    - Never extract facts from the assistant's answer.
    - Questions normally produce no memories.
    """

    def __init__(self) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_output_tokens=1024,
        )

        self.structured_model = self.model.with_structured_output(
            MemoryExtractionResult
        )

    async def extract(
        self,
        user_message: str,
    ) -> MemoryExtractionResult:
        """
        Extract durable memories from one user message.
        """

        if not user_message.strip():
            return MemoryExtractionResult(memories=[])

        prompt = """
You are a production-grade long-term memory extractor for a RAG application.

Your task is to identify ONLY durable facts that the USER explicitly
provided about themselves.

IMPORTANT RULES:

1. Extract only facts explicitly stated by the user.
2. Never infer, guess, or assume information.
3. Do not extract information from the assistant's response.
4. Do not treat a question as a new user fact.
5. Do not save temporary conversation details.
6. Do not save greetings or casual statements.
7. Do not save secrets, passwords, API keys, tokens, or authentication data.
8. Prefer stable facts that may be useful in future conversations.
9. Preserve the user's factual wording when possible.
10. If there are no durable facts, return an empty memories list.

Examples:

User:
"My name is Shayan Javed."

Extract:
name = Shayan Javed
type = profile

User:
"I study BS Computer Science at Sindh Madressatul Islam University."

Extract:
degree = BS Computer Science
type = education

Extract:
university = Sindh Madressatul Islam University
type = education

User:
"What is my name?"

Extract:
No memories.

User:
"Today I am working on a RAG project."

Extract:
No memories unless the statement clearly establishes a durable preference,
role, project ownership, or other long-term fact.

User:
"I am an AI/ML engineer and I mainly work with Python."

Extract:
job_title = AI/ML engineer
type = career

skill = Python
type = skill

User:
"My password is Test123!"

Extract:
No memories.

Return only structured memory candidates.

USER MESSAGE:
{user_message}
"""

        return await self.structured_model.ainvoke(
            prompt.format(user_message=user_message.strip())
        )
