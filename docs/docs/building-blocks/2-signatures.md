 League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```


Many DSPy modules (except `dspy.Predict`) return auxiliary information by expanding your signature under the hood.

For example, `dspy.ChainOfThought` also adds a `rationale` field that includes the LM's reasoning before it generates the output `summary`.

```python
print("Rationale:", response.rationale)
```
**Output:**
```text
Rationale: produce the summary. We need to highlight the key points about Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```

## **Class-based** DSPy Signatures

For some advanced tasks, you need more verbose signatures. This is typically to:

1. Clarify something about the nature of the task (expressed below as a `docstring`).

2. Supply hints on the nature of an input field, expressed as a `desc` keyword argument for `dspy.InputField`.

3. Supply constraints on an output field, expressed as a `desc` keyword argument for `dspy.OutputField`.


### Example C: Classification

Notice how the docstring contains (minimal) instructions, which in this case are necessary to have a fully-defined task.

Some optimizers in DSPy, like `COPRO`, can take this simple docstring and then generate more effective variants if needed.

```python
class Emotion(dspy.Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""
    
    sentence = dspy.InputField()
    sentiment = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)
```
**Output:**
```text
Prediction(
    sentiment='Fear'
)
```

**Tip:** There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of the your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).


### Example D: A metric that evaluates faithfulness to citations

```python
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context = dspy.InputField(desc="facts here are assumed to be true")
    text = dspy.InputField()
    faithfulness = dspy.OutputField(desc="True/False indicating if text is faithful to context")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**Output:**
```text
Prediction(
    rationale="produce the faithfulness. We know that Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. However, there is no mention of him scoring three goals for Colchester United.",
    faithfulness='False'
)
```


## Using signatures to build modules & compiling them

While signatures are convenient for prototyping with structured inputs/outputs, that's not the main reason to use them!

You should compose multiple signatures into bigger [DSPy modules](https://dspy-docs.vercel.app/docs/building-blocks/modules) and [compile these modules into optimized prompts](https://dspy-docs.vercel.app/docs/building-blocks/optimizers#what-does-a-dspy-optimizer-tune-how-does-it-tune-them) and finetunes.
