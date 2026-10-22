from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

from app.core.config import get_settings

settings = get_settings()


class GroqJudge(DeepEvalBaseLLM):
    """Adapter that allows DeepEval to use Groq as the judge LLM."""

    def __init__(self):
        self.model_name = settings.groq_model

        self.llm = ChatGroq(
            model=self.model_name,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=2048,
        )

    def load_model(self):
        return self.llm

    def generate(
        self,
        prompt: str,
        schema: Any = None,
    ) -> str:

        response = self.llm.invoke(prompt)

        content = response.content

        if not isinstance(content, str):
            content = str(content)

        return content.strip()

    async def a_generate(
        self,
        prompt: str,
        schema: Any = None,
    ) -> str:

        response = await self.llm.ainvoke(prompt)

        content = response.content

        if not isinstance(content, str):
            content = str(content)

        return content.strip()

    def get_model_name(self) -> str:
        return self.model_name