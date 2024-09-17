 over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = cot_predictor(input=doc_query_pair)
```

## Typed Predictors as Decorators

While the `dspy.TypedPredictor` and `dspy.TypedChainOfThought` provide a convenient way to use typed predictors, you can also use them as decorators to enforce type constraints on the inputs and outputs of the function. This relies on the internal definitions of the Signature class and its function arguments, outputs, and docstrings.

```python
@dspy.predictor
def answer(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you are about the answer."""
    pass

@dspy.cot
def answer(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you are about the answer."""
    pass

prediction = answer(doc_query_pair=doc_query_pair)
```

## Composing Functional Typed Predictors in `dspy.Module`

If you're creating DSPy pipelines via `dspy.Module`, then you can simply use Functional Typed Predictors by creating these class methods and using them as decorators. Here is an example of using functional typed predictors to create a `SimplifiedBaleen` pipeline:

```python
class SimplifiedBaleen(FunctionalModule):
    def __init__(self, passages_per_hop=3, max_hops=1):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.max_hops = max_hops

    @cot
    def generate_query(self, context: list[str], question) -> str:
        """Write a simple search query that will help answer a complex question."""
        pass

    @cot
    def generate_answer(self, context: list[str], question) -> str:
        """Answer questions with short factoid answers."""
        pass

    def forward(self, question):
        context = []

        for _ in range(self.max_hops):
            query = self.generate_query(context=context, question=question)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=answer)
```

## Optimizing Typed Predictors

Typed predictors can be optimized on the Signature instructions through the `optimize_signature` optimizer. Here is an example of this optimization on the `QASignature`:

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.teleprompt.signature_opt_typed import optimize_signature

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=4000)
gpt4 = dspy.OpenAI(model='gpt-4', max_tokens=4000)
dspy.settings.configure(lm=turbo)

evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=10, display_progress=True)

result = optimize_signature(
    student=dspy.TypedPredictor(QASignature),
    evaluator=evaluator,
    initial_prompts=6,
    n_iterations=100,
    max_examples=30,
    verbose=True,
    prompt_model=gpt4,
)
```
