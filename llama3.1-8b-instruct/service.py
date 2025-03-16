import uuid
from typing import AsyncGenerator, Optional

import bentoml
import fastapi
from annotated_types import Ge, Le
from typing_extensions import Annotated

openai_api_app = fastapi.FastAPI()


MAX_SESSION_LEN = 2048
MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@bentoml.asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentolmdeploy-llama3.1-8b-insruct-service",
    image=bentoml.images.PythonImage(python_version="3.11").requirements_file("requirements.txt"),
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class LMDeploy:
    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from lmdeploy import ChatTemplateConfig
        from lmdeploy.serve.async_engine import AsyncEngine
        from lmdeploy.messages import TurbomindEngineConfig

        engine_config = TurbomindEngineConfig(
            model_name=MODEL_ID,
            model_format="hf",
            cache_max_entry_count=0.9,
            enable_prefix_caching=True,
            session_len=MAX_SESSION_LEN,
        )
        self.engine = AsyncEngine(
            self.hf_model,
            backend_config=engine_config,
            model_name=MODEL_ID,
            chat_template_config=ChatTemplateConfig("llama3_1"),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.stop_tokens = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.eos_token_id,
            ),
            "<|eot_id|>",
        ]

        import lmdeploy.serve.openai.api_server as lmdeploy_api_server
        lmdeploy_api_server.VariableInterface.async_engine = self.engine

        OPENAI_ENDPOINTS = [
            ["/chat/completions", lmdeploy_api_server.chat_completions_v1, ["POST"]],
            ["/completions", lmdeploy_api_server.completions_v1, ["POST"]],
            ["/models", lmdeploy_api_server.available_models, ["GET"]],
        ]

        for route, endpoint, methods in OPENAI_ENDPOINTS:
            openai_api_app.add_api_route(
                path=route,
                endpoint=endpoint,
                methods=methods,
                include_in_schema=True,
            )


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

        messages = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=prompt),
        ]

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )

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
