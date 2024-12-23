<div align="center">
    <h1 align="center">Self-host LLMs with LMDeploy and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy open-source Large Language Models (LLMs) using [LMDeploy](https://github.com/InternLM/lmdeploy), a toolkit for compressing, deploying, and serving LLMs.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or LMDeploy options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- If you want to test the Service locally, you need a Nvidia GPU with at least 24G VRAM.
- This example uses Mistral 7B Instruct. Make sure you have [gained access to the model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoLMDeploy.git
cd BentoLMDeploy/mistral-7b-instruct
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-07-05T23:57:36+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:LMDeploy" listening on http://localhost:3000 (Press CTRL+C to quit)                                     
2024-07-05 23:57:38,582 - lmdeploy - INFO - input backend=turbomind, backend_config=TurbomindEngineConfig(model_name='mistralai/Mistral-7B-Instruct-v0.2', model_format='hf', tp=1, session_len=N
one, max_batch_size=128, cache_max_entry_count=0.95, cache_block_seq_len=64, enable_prefix_caching=False, quant_policy=0, rope_scaling_factor=0.0, use_logn_attn=False, download_dir=None, revisi
on=None, max_prefill_token_num=8192, num_tokens_per_iter=0, max_prefill_iters=1)                                                                                                                 
2024-07-05 23:57:38,582 - lmdeploy - INFO - input chat_template_config=None                                                                                                                      
2024-07-05 23:57:38,616 - lmdeploy - INFO - updated chat_template_onfig=ChatTemplateConfig(model_name='mistral', system=None, meta_instruction=None, eosys=None, user=None, eoh=None, assistant=N
one, eoa=None, separator=None, capability=None, stop_words=None)                                                                                                                                 
2024-07-05 23:57:38,652 - lmdeploy - INFO - model_source: hf_model

...
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors like I'\''m five years old",
  "max_tokens": 1024
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        max_tokens=1024
    )
    for response in response_generator:
        print(response, end='')
```

</details>

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
