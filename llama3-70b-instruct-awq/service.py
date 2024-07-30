import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from openai_endpoints import openai_api_app


MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "casperhansen/llama-3-70b-instruct-awq"
BENTO_MODEL_TAG = MODEL_ID.lower().replace("/", "--")


@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentolmdeploy-llama3-70b-instruct-awq-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class LMDeploy:

    bento_model_ref = bentoml.models.get(BENTO_MODEL_TAG)

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from lmdeploy.serve.async_engine import AsyncEngine
        from lmdeploy.messages import TurbomindEngineConfig

        engine_config = TurbomindEngineConfig(
            model_name=MODEL_ID,
            model_format="awq",
            cache_max_entry_count=0.9,
            enable_prefix_caching=True,
        )
        self.engine = AsyncEngine(
            self.bento_model_ref.path, backend_config=engine_config
        )

        import lmdeploy.serve.openai.api_server as lmdeploy_api_sever
        lmdeploy_api_sever.VariableInterface.async_engine = self.engine

        tokenizer = AutoTokenizer.from_pretrained(self.bento_model_ref.path)
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
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:

        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens, stop_words=self.stop_tokens,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)

        session_id = abs(uuid.uuid4().int >> 96)
        stream = self.engine.generate(
            prompt, session_id=session_id, gen_config=gen_config
        )

        async for request_output in stream:
            if await ctx.request.is_disconnected():
                await self.engine.stop_session(session_id)
                await self.engine.end_session(session_id)
                return
            yield request_output.response
