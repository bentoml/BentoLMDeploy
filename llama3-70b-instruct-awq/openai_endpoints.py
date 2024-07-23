import fastapi
import lmdeploy.serve.openai.api_server as lmdeploy_api_server

openai_api_app = fastapi.FastAPI()

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
