from typing import Any

from deepeval.models import DeepEvalBaseLLM
from google import genai
from google.genai import types

from app.core.config import get_settings

settings = get_settings()


class GeminiJudge(DeepEvalBaseLLM):
    """Gemini judge model for DeepEval."""

    def __init__(self):
        self.model_name = settings.gemini_model
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def load_model(self):
        return self.client

    def generate(
        self,
        prompt: str,
        schema: Any = None,
    ) -> str:

        config = None

        if schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0,
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        return response.text

    async def a_generate(
        self,
        prompt: str,
        schema: Any = None,
    ) -> str:

        config = None

        if schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0,
            )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        return response.text

    def get_model_name(self) -> str:
        return self.model_name