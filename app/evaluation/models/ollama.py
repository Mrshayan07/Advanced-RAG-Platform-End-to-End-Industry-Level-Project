
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama


class OllamaJudge(DeepEvalBaseLLM):
    """
    DeepEval judge adapter for local Qwen3:4B through Ollama.

    This model is used only for evaluation/judging.
    The main RAG generation model remains unchanged.

    Architecture:

        RAG Application
              |
              v
        Gemini Generation
              |
              v
        DeepEval Metrics
              |
              v
        Qwen3:4B via Ollama
    """

    def __init__(self):
        self.model_name = "qwen3:4b-instruct"

        self.llm = ChatOllama(
            model=self.model_name,
            base_url="http://localhost:11434",
            temperature=0,
            num_ctx=4096,
            num_predict=1024,
            format="json",
        )

    def load_model(self):
        """
        Return the underlying Ollama LangChain model.
        """
        return self.llm

    def generate(
        self,
        prompt: str,
        schema: Any = None,
    ) -> str:
        """
        Synchronous generation for DeepEval.
        """

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
        """
        Asynchronous generation for DeepEval.

        DeepEval may execute metric evaluation asynchronously,
        so use ChatOllama.ainvoke() directly.
        """

        response = await self.llm.ainvoke(prompt)

        content = response.content

        if not isinstance(content, str):
            content = str(content)

        return content.strip()

    def get_model_name(self) -> str:
        """
        Return the model identifier used by the evaluator.
        """
        return self.model_name
