 just call `qa(...)` in a loop with the same input, it will always
return the same value! That\'s by design.

To loop and generate one output at a time with the same input, bypass
the cache by making sure each request is (slightly) unique, as below.

```python
for idx in range(5):
    response = qa(question="How many floors are in the castle David Gregory inherited?", config=dict(temperature=0.7+0.0001*idx))
    print(f'{idx+1}.', response.answer)
```
**Output:**
```text
1. The specific number of floors in David Gregory's inherited castle is not provided here, so further research would be needed to determine the answer.
2. It is not possible to determine the exact number of floors in the castle David Gregory inherited without specific information about the castle's layout and history.
3. The castle David Gregory inherited has 5 floors.
4. We need more information to determine the number of floors in the castle David Gregory inherited.
5. The castle David Gregory inherited has a total of 6 floors.
```

## Remote LMs.

These models are managed services. You just need to sign up and obtain an API key. Calling any of the remote LMs below assumes authentication and mirrors the following format for setting up the LM:

```python
lm = dspy.{provider_listed_below}(model="your model", model_request_kwargs="...")
```

1.  `dspy.OpenAI` for GPT-3.5 and GPT-4.

2.  `dspy.Cohere`

3.  `dspy.Anyscale` for hosted Llama2 models.

4.  `dspy.Together` for hosted various open source models.

5.  `dspy.PremAI` for hosted best open source and closed source models.

### Local LMs.

You need to host these models on your own GPU(s). Below, we include pointers for how to do that.

1.  `dspy.HFClientTGI`: for HuggingFace models through the Text Generation Inference (TGI) system. [Tutorial: How do I install and launch the TGI server?](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/local_models/HFClientTGI)

```python
tgi_mistral = dspy.HFClientTGI(model="mistralai/Mistral-7B-Instruct-v0.2", port=8080, url="http://localhost")
```

2.  `dspy.HFClientVLLM`: for HuggingFace models through vLLM. [Tutorial: How do I install and launch the vLLM server?](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/local_models/HFClientVLLM)

```python
vllm_mistral = dspy.HFClientVLLM(model="mistralai/Mistral-7B-Instruct-v0.2", port=8080, url="http://localhost")
```

3.  `dspy.HFModel` (experimental) [Tutorial: How do I initialize models using HFModel](https://dspy-docs.vercel.app/api/local_language_model_clients/HFModel)

```python
mistral = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2')
```

4.  `dspy.Ollama` (experimental) for open source models through [Ollama](https://ollama.com). [Tutorial: How do I install and use Ollama on a local computer?](https://dspy-docs.vercel.app/api/local_language_model_clients/Ollama)\n",

```python
ollama_mistral = dspy.OllamaLocal(model='mistral')
```

5.  `dspy.ChatModuleClient` (experimental): [How do I install and use MLC?](https://dspy-docs.vercel.app/api/local_language_model_clients/MLC)

```python
model = 'dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1'
model_path = 'dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so'

llama = dspy.ChatModuleClient(model=model, model_path=model_path)
```