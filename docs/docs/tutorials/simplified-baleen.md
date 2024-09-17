_position: 2
---

# [02] Multi-Hop Question Answering

A single search query is often not enough for complex QA tasks. For instance, an example within `HotPotQA` includes a question about the birth city of the writer of "Right Back At It Again". A search query often identifies the author correctly as "Jeremy McKinnon", but lacks the capability to compose the intended answer in determining when he was born.

The standard approach for this challenge in retrieval-augmented NLP literature is to build multi-hop search systems, like GoldEn (Qi et al., 2019) and Baleen (Khattab et al., 2021). These systems read the retrieved results and then generate additional queries to gather additional information when necessary before arriving to a final answer. Using DSPy, we can easily simulate such systems in a few lines of code.

## Configuring LM and RM

We'll start by setting up the language model (LM) and retrieval model (RM), which **DSPy** supports through multiple [LM](https://dspy-docs.vercel.app/docs/category/language-model-clients) and [RM](https://dspy-docs.vercel.app/docs/category/retrieval-model-clients) APIs and [local models hosting](https://dspy-docs.vercel.app/docs/category/local-language-model-clients).

In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the `ColBERTv2` retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)). We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval. 

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
```

## Loading the Dataset

For this tutorial, we make use of the mentioned `HotPotQA` dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion. We can load this dataset provided by DSPy through the `HotPotQA` class:

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

## Building Signature

Now that we have the data loaded let's start defining the signatures for sub-tasks of out Baleen pipeline.

We'll start by creating the `GenerateAnswer` signature that'll take `context` and `question` as input and give `answer` as output.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

Unlike usual QA pipelines, we have an intermediate question-generation step in Baleen for which we'll need to define a new Signature for the "hop" behavior: inputting some context and a question to generate a search query to find missing information. 

```python
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
```

:::info
We could have written `context = GenerateAnswer.signature.context` to avoid duplicating the description of the context field.
:::

Now that we have the necessary signatures in place, we can start building the Baleen pipeline!

## Building the Pipeline

So, let's define the program itself `SimplifiedBaleen`. There are many possible ways to implement this, but we'll keep this version down to the key elements.

```python
from dsp.utils import deduplicate

clas