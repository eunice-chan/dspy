 specifying multiple thread settings in the respective DSPy `optimizers` or within the `dspy.Evaluate` utility function.

- **How do freeze a module?**

Modules can be frozen by setting their `._compiled` attribute to be True, indicating the module has gone through optimizer compilation and should not have its parameters adjusted. This is handled internally in optimizers such as `dspy.BootstrapFewShot` where the student program is ensured to be frozen before the teacher propagates the gathered few-shot demonstrations in the bootstrapping process. 

- **How do I get JSON output?**

You can specify JSON-type descriptions in the `desc` field of the long-form signature `dspy.OutputField` (e.g. `output = dspy.OutputField(desc='key-value pairs')`).

If you notice outputs are still not conforming to JSON formatting, try Asserting this constraint! Check out [Assertions](https://dspy-docs.vercel.app/docs/building-blocks/assertions) (or the next question!)

- **How do I use DSPy assertions?**

    a) **How to Add Assertions to Your Program**:
    - **Define Constraints**: Use `dspy.Assert` and/or `dspy.Suggest` to define constraints within your DSPy program. These are based on boolean validation checks for the outcomes you want to enforce, which can simply be Python functions to validate the model outputs.
    - **Integrating Assertions**: Keep your Assertion statements following a model generations (hint: following a module layer)

    b) **How to Activate the Assertions**:
    1. **Using `assert_transform_module`**:
        - Wrap your DSPy module with assertions using the `assert_transform_module` function, along with a `backtrack_handler`. This function transforms your program to include internal assertions backtracking and retry logic, which can be customized as well:
        `program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)`
    2. **Activate Assertions**:
        - Directly call `activate_assertions` on your DSPy program with assertions: `program_with_assertions = ProgramWithAssertions().activate_assertions()`

    **Note**: To use Assertions properly, you must **activate** a DSPy program that includes `dspy.Assert` or `dspy.Suggest` statements from either of the methods above. 

## Errors

- **How do I deal with "context too long" errors?**

If you're dealing with "context too long" errors in DSPy, you're likely using DSPy optimizers to include demonstrations within your prompt, and this is exceeding your current context window. Try reducing these parameters (e.g. `max_bootstrapped_demos` and `max_labeled_demos`). Additionally, you can also reduce the number of retrieved passages/docs/embeddings to ensure your prompt is fitting within your model context length.

A more general fix is simply increasing the number of `max_tokens` specified to the LM request (e.g. `lm = dspy.OpenAI(model = ..., max_tokens = ...`).

- **How do I deal with timeouts or backoff errors?**

Firstly, please refer to your LM/RM provider to ensure stable status or sufficient rate limits for your use case!

Additionally, try reducing the number of threads you are testing on as the corresponding servers may get overloaded with requests and trigger a backoff + retry mechanism.

If all variables seem stable, you may be experiencing timeouts or backoff errors due to incorrect payload requests sent to the api providers. Please verify your arguments are compatible with the SDK you are interacting with. 

You can configure backoff times for your LM/RM provider by setting `dspy.settings.backoff_time` while configuring your DSPy workflow. 

```python
dspy.settings.configure(backoff_time = ...)
```

Additionally, if you'd like to set individual backoff times for specific providers, you can do so through the DSPy context manager: 

```python
with dspy.context(backoff_time = ..):
      dspy.OpenAI(...) # example

with dspy.context(backoff_time = ..):
      dspy.AzureOpenAI(...) # example
```

At times, DSPy may have hard-coded arguments that are not relevant for your compatible, in which case, please free to open a PR