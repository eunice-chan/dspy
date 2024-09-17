---
sidebar_position: 1
---

# [01] RAG: Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) is an approach that allows LLMs to tap into a large corpus of knowledge from sources and query its knowledge store to find relevant passages/content and produce a well-refined response.

RAG ensures LLMs can dynamically utilize real-time knowledge even if not originally trained on the subject and give thoughtful answers. However, with this nuance comes greater complexities in setting up refined RAG pipelines. To reduce these intricacies, we turn to **DSPy**, which offers a seamless approach to setting up prompting pipelines!

## Configuring LM and RM

We'll start by setting up the language model (LM) and retrieval model (RM), which **DSPy** supports through multiple [LM](https://dspy-docs.vercel.app/docs/category/language-model-clients) and [RM](https://dspy-docs.vercel.app/docs/category/retrieval-model-clients) APIs and [local models hosting](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/local_models/HFClientTGI).

In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the `ColBERTv2` retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)). We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval. 

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
```


## Loading the Dataset

For this tutorial, we make use of the `HotPotQA` dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion. We can load this dataset provided by DSPy through the `HotPotQA` class:

```python
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)
```
**Output:**
```text
(20, 50)
```

## Building Signatures

Now that we have the data loaded, let's start defining the [signatures](https://dspy-docs.vercel.app/docs/building-blocks/signatures) for the sub-tasks of our pipeline.

We can identify our simple input `question` and output `answer`, but since we are building out a RAG pipeline, we wish to utilize some contextual information from our ColBERT corpus. So let's define our signature: `context, question --> answer`.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

We include small descriptions for the `context` and `answer` fields to define more robust guidelines on what the model will receive and should generate. 

## Building the Pipeline

We will build our RAG pipeline as a [DSPy module](https://dspy-docs.vercel.app/docs/building-blocks/modules) which will require two methods:

* The `__init__` method will simply declare the sub-modules it needs: `dspy.Retrieve` and `dspy.ChainOfThought`. The latter is defined to implement our `GenerateAnswer` signature.
* The `forward` method will describe the control flow of answering the question using the modules we have: Given a question, we'll search for the top-3 relevant passages and then feed them as context for answer generation.


```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        