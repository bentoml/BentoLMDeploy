from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import bentoml
import pydantic
from annotated_types import Ge, Le
from typing_extensions import Annotated

from openai_endpoints import openai_api_app


SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

class BentoArgs(pydantic.BaseModel):
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_tokens: int = pydantic.Field(default=1024)
    tp: int = 2


bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentolmdeploy-llama-3.3-instruct-70b-service",
    traffic={ "timeout": 300, },
    resources={
        "gpu": bento_args.tp,
        "gpu_type": "nvidia-h100-80gb",
    },
)
class LMDeploy:
    model_ref = bentoml.models.HuggingFaceModel(
        bento_args.model_id, exclude=["original", "consolidated*", "*.pth", "*.pt", "original/**/*"]
    )

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from lmdeploy.serve.async_engine import AsyncEngine
        from lmdeploy.messages import TurbomindEngineConfig

        import lmdeploy.serve.openai.api_server as lmdeploy_api_sever

        engine_config = TurbomindEngineConfig(
            model_name=bento_args.model_id,
            model_format="hf",
            cache_max_entry_count=0.85,
            enable_prefix_caching=True,
            tp=bento_args.tp,
        )
        self.engine = AsyncEngine(self.model_ref, backend_config=engine_config)

        lmdeploy_api_sever.VariableInterface.async_engine = self.engine

        tokenizer = AutoTokenizer.from_pretrained(self.model_ref)
        self.stop_tokens = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.eos_token_id,
            ),
            "<|eot_id|>",
        ]

    @bentoml.api
    async def generate(
        self,
        ctx: bentoml.Context,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: str | None = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(bento_args.max_tokens)] = bento_args.max_tokens,
    ) -> AsyncGenerator[str, None]:
        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            stop_words=self.stop_tokens,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)

        session_id = abs(uuid.uuid4().int >> 96)
        stream = self.engine.generate(prompt, session_id=session_id, gen_config=gen_config)

        async for request_output in stream:
            if await ctx.request.is_disconnected():
                await self.engine.stop_session(session_id)
                await self.engine.end_session(session_id)
                return
            yield request_output.response
