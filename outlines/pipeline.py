import re
from typing import List, Union, Optional, Callable, Iterator, Any, Dict, Tuple
import torch

from outlines.fsm.guide import Guide
from outlines.processors.structured import GuideLogitsProcessor
from outlines.types.dsl import Term, String, Regex, to_regex
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models.tokenizer import Tokenizer
from outlines.samplers import Sampler, multinomial

class TokenTrigger:
    """A special token pattern that when generated signals to halt and move to the next step.
    
    This class allows defining special token sequences that, when encountered during 
    generation, will trigger a halt and allow moving to the next step in a pipeline.
    """
    def __init__(self, trigger_pattern: str):
        """
        Parameters
        ----------
        trigger_pattern : str
            The regex pattern that when matched will trigger a halt
        """
        self.trigger_pattern = trigger_pattern
        
    def is_triggered(self, generated_text: str) -> bool:
        """Check if the trigger pattern appears in the generated text.
        
        Parameters
        ----------
        generated_text : str
            The generated text to check for trigger patterns
            
        Returns
        -------
        bool
            True if trigger pattern is found, False otherwise
        """
        return bool(re.search(self.trigger_pattern, generated_text))


class PipelineStep:
    """A single step in a generation pipeline.
    
    Each step consists of a regex pattern that guides generation,
    and an optional token trigger that signals when to move to the next step.
    
    The regex pattern can be any outlines Term type, including String, Regex,
    and JsonSchema (which supports Pydantic models).
    """
    def __init__(
        self, 
        regex_pattern: Term,
        token_trigger: Optional[TokenTrigger] = None,
        name: Optional[str] = None
    ):
        """
        Parameters
        ----------
        regex_pattern : Term
            The regex pattern (as an outlines Term) that guides generation for this step
        token_trigger : Optional[TokenTrigger]
            An optional token trigger that signals when to move to the next step
        name : Optional[str]
            An optional name for this step for easier identification
        """
        self.regex_pattern = regex_pattern
        self.token_trigger = token_trigger
        self.name = name or f"Step-{id(self)}"
        
    def get_regex_string(self) -> str:
        """Get the regex string for this step's pattern.
        
        Returns
        -------
        str
            The regex string representing this step's pattern
        """
        return to_regex(self.regex_pattern)
    
    def is_complete(self, generated_text: str) -> bool:
        """Check if this step is complete based on token trigger.
        
        Parameters
        ----------
        generated_text : str
            The text generated so far
            
        Returns
        -------
        bool
            True if this step is complete, False otherwise
        """
        if self.token_trigger is None:
            return False  # Without a token trigger, step is never "complete" until generation stops
        return self.token_trigger.is_triggered(generated_text)


class Pipeline:
    """A pipeline for executing multiple regex-guided generation steps in sequence.
    
    This class allows defining a series of generation steps, each with its own
    regex pattern and optional token trigger, to be executed sequentially.
    """
    def __init__(self, steps: List[PipelineStep]):
        """
        Parameters
        ----------
        steps : List[PipelineStep]
            The sequence of steps to execute in this pipeline
        """
        self.steps = steps
        
    def execute(
        self,
        model: Any,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params
    ) -> str:
        """Execute the pipeline steps sequentially.
        
        Parameters
        ----------
        model : Any
            The model to use for generation
        prompts : Union[str, List[str]]
            The prompt(s) to use for generation
        max_tokens : Optional[int]
            Maximum number of tokens to generate
        stop_at : Optional[Union[str, List[str]]]
            Additional stop sequences
        seed : Optional[int]
            Random seed for generation
        **model_specific_params
            Model-specific parameters
            
        Returns
        -------
        str
            The combined generated text from all steps
        """
        result = []
        current_prompt = prompts
        
        for i, step in enumerate(self.steps):
            # Create a generator for this step
            from outlines.processors import RegexLogitsProcessor
            
            pattern = step.get_regex_string()
            logits_processor = RegexLogitsProcessor(pattern, tokenizer=model.tokenizer)
            generator = SequenceGeneratorAdapter(model, logits_processor, multinomial())
            
            # Generate text for this step
            generated = generator(
                current_prompt,
                max_tokens=max_tokens,
                stop_at=stop_at,
                seed=seed,
                **model_specific_params
            )
            
            generated_text = generated if isinstance(generated, str) else generated[0]
            result.append(generated_text)
            
            # Use the generated text as the prompt for the next step
            if i < len(self.steps) - 1:
                current_prompt = current_prompt + " " + generated_text
        
        return " ".join(result)
        
    def stream(
        self,
        model: Any,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params
    ) -> Iterator[str]:
        """Stream execution of the pipeline steps.
        
        Parameters
        ----------
        model : Any
            The model to use for generation
        prompts : Union[str, List[str]]
            The prompt(s) to use for generation
        max_tokens : Optional[int]
            Maximum number of tokens to generate
        stop_at : Optional[Union[str, List[str]]]
            Additional stop sequences
        seed : Optional[int]
            Random seed for generation
        **model_specific_params
            Model-specific parameters
            
        Yields
        ------
        str
            Streamed tokens from the generation process
        """
        current_prompt = prompts
        generated_so_far = ""
        
        for i, step in enumerate(self.steps):
            # Create a generator for this step
            from outlines.processors import RegexLogitsProcessor
            
            pattern = step.get_regex_string()
            logits_processor = RegexLogitsProcessor(pattern, tokenizer=model.tokenizer)
            generator = SequenceGeneratorAdapter(model, logits_processor, multinomial())
            
            # Generate and stream tokens for this step
            step_generated = ""
            for token in generator.stream(
                current_prompt,
                max_tokens=max_tokens,
                stop_at=stop_at,
                seed=seed,
                **model_specific_params
            ):
                if isinstance(token, list):
                    token = token[0] if token else ""
                
                step_generated += token
                yield token
                
                # Check if we should move to the next step
                if step.is_complete(step_generated):
                    break
            
            generated_so_far += step_generated
            
            # Use the generated text as the prompt for the next step
            if i < len(self.steps) - 1:
                current_prompt = current_prompt + " " + step_generated


class PipelineBuilder:
    """A helper class for constructing pipelines.
    
    This class provides a convenient interface for building pipelines by defining
    steps with regex patterns and token triggers.
    """
    def __init__(self):
        self.steps = []
        
    def add_step(
        self, 
        regex_pattern: Term,
        token_trigger: Optional[Union[str, TokenTrigger]] = None,
        name: Optional[str] = None
    ) -> 'PipelineBuilder':
        """Add a step to the pipeline.
        
        Parameters
        ----------
        regex_pattern : Term
            The regex pattern for this step. Can be any Term type including
            String, Regex, or a Pydantic model wrapped with JsonSchema.
        token_trigger : Optional[Union[str, TokenTrigger]]
            A token trigger or trigger pattern string
        name : Optional[str]
            A name for this step
            
        Returns
        -------
        PipelineBuilder
            The builder instance for method chaining
        """
        if isinstance(token_trigger, str):
            token_trigger = TokenTrigger(token_trigger)
            
        self.steps.append(PipelineStep(regex_pattern, token_trigger, name))
        return self
        
    def build(self) -> Pipeline:
        """Build and return the pipeline.
        
        Returns
        -------
        Pipeline
            The constructed pipeline
        """
        return Pipeline(self.steps)


def create_pipeline(
    regex_patterns: List[Term],
    token_triggers: Optional[List[Optional[str]]] = None,
    step_names: Optional[List[Optional[str]]] = None
) -> Pipeline:
    """Create a pipeline directly from lists of regex patterns and optional token triggers.
    
    Parameters
    ----------
    regex_patterns : List[Term]
        List of regex patterns for each step. These can be any Term type including
        String, Regex, or a Pydantic model wrapped with the json_schema() function.
    token_triggers : Optional[List[Optional[str]]]
        Optional list of token trigger patterns for each step. 
        Use None for steps that don't have triggers.
    step_names : Optional[List[Optional[str]]]
        Optional list of names for each step.
        
    Returns
    -------
    Pipeline
        The constructed pipeline
        
    Examples
    --------
    >>> from outlines.types.dsl import String, digit, json_schema
    >>> from pydantic import BaseModel
    >>> 
    >>> # Define a Pydantic model for structured output
    >>> class Conclusion(BaseModel):
    ...     conclusion: str
    ...     count: int
    ... 
    >>> pipeline = create_pipeline(
    ...     regex_patterns=[
    ...         String("Step 1: ") + digit, 
    ...         String("Step 2: ") + digit,
    ...         json_schema(Conclusion)
    ...     ],
    ...     token_triggers=[r"\n", r"\n", None]
    ... )
    """
    builder = PipelineBuilder()
    
    # Default empty lists if None
    if token_triggers is None:
        token_triggers = [None] * len(regex_patterns)
    if step_names is None:
        step_names = [None] * len(regex_patterns)
        
    # Ensure lists are the same length
    if len(token_triggers) != len(regex_patterns):
        raise ValueError("token_triggers list must have the same length as regex_patterns")
    if len(step_names) != len(regex_patterns):
        raise ValueError("step_names list must have the same length as regex_patterns")
    
    # Add each step to the builder
    for pattern, trigger, name in zip(regex_patterns, token_triggers, step_names):
        builder.add_step(regex_pattern=pattern, token_trigger=trigger, name=name)
        
    return builder.build()


def run_pipeline(
    model: Any,
    prompt: Union[str, List[str]],
    regex_patterns: List[Term],
    token_triggers: Optional[List[Optional[str]]] = None,
    step_names: Optional[List[Optional[str]]] = None,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
    **model_specific_params
) -> str:
    """Create and execute a pipeline in a single function call.
    
    Parameters
    ----------
    model : Any
        The model to use for generation
    prompt : Union[str, List[str]]
        The prompt(s) to use for generation
    regex_patterns : List[Term]
        List of regex patterns for each step. These can be any Term type including
        String, Regex, or Pydantic models wrapped with the json_schema() function.
    token_triggers : Optional[List[Optional[str]]]
        Optional list of token trigger patterns for each step.
        Use None for steps that don't have triggers.
    step_names : Optional[List[Optional[str]]]
        Optional list of names for each step.
    max_tokens : Optional[int]
        Maximum number of tokens to generate
    stop_at : Optional[Union[str, List[str]]]
        Additional stop sequences
    seed : Optional[int]
        Random seed for generation
    **model_specific_params
        Model-specific parameters
        
    Returns
    -------
    str
        The combined generated text from all steps
        
    Examples
    --------
    >>> from outlines.types.dsl import String, digit, json_schema
    >>> from pydantic import BaseModel
    >>> 
    >>> # Define a Pydantic model for structured output
    >>> class Conclusion(BaseModel):
    ...     conclusion: str
    ...     count: int
    ... 
    >>> model = outlines.models.transformers("gpt2")
    >>> output = run_pipeline(
    ...     model,
    ...     prompt="Count the letters",
    ...     regex_patterns=[
    ...         String("Thinking: ") + String("..."),
    ...         json_schema(Conclusion)
    ...     ],
    ...     token_triggers=[r"\n\n", None]
    ... )
    """
    pipeline = create_pipeline(regex_patterns, token_triggers, step_names)
    return pipeline.execute(model, prompt, max_tokens, stop_at, seed, **model_specific_params)


def stream_pipeline(
    model: Any,
    prompt: Union[str, List[str]],
    regex_patterns: List[Term],
    token_triggers: Optional[List[Optional[str]]] = None,
    step_names: Optional[List[Optional[str]]] = None,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
    **model_specific_params
) -> Iterator[str]:
    """Create and stream from a pipeline in a single function call.
    
    This function works like run_pipeline() but streams the output token by token.
    
    Parameters
    ----------
    model : Any
        The model to use for generation
    prompt : Union[str, List[str]]
        The prompt(s) to use for generation
    regex_patterns : List[Term]
        List of regex patterns for each step. These can be any Term type including
        String, Regex, or Pydantic models wrapped with the json_schema() function.
    token_triggers : Optional[List[Optional[str]]]
        Optional list of token trigger patterns for each step.
        Use None for steps that don't have triggers.
    step_names : Optional[List[Optional[str]]]
        Optional list of names for each step.
    max_tokens : Optional[int]
        Maximum number of tokens to generate
    stop_at : Optional[Union[str, List[str]]]
        Additional stop sequences
    seed : Optional[int]
        Random seed for generation
    **model_specific_params
        Model-specific parameters
        
    Returns
    -------
    Iterator[str]
        Iterator yielding tokens as they are generated
        
    Examples
    --------
    >>> from outlines.types.dsl import String, digit, json_schema
    >>> from pydantic import BaseModel
    >>> 
    >>> # Define a Pydantic model for structured output
    >>> class Conclusion(BaseModel):
    ...     conclusion: str
    ...     count: int
    ... 
    >>> model = outlines.models.transformers("gpt2")
    >>> for token in stream_pipeline(
    ...     model,
    ...     prompt="Count the letters",
    ...     regex_patterns=[
    ...         String("Thinking: ") + String("..."),
    ...         json_schema(Conclusion)
    ...     ],
    ...     token_triggers=[r"\n\n", None]
    ... ):
    ...     print(token, end="", flush=True)
    """
    pipeline = create_pipeline(regex_patterns, token_triggers, step_names)
    return pipeline.stream(model, prompt, max_tokens, stop_at, seed, **model_specific_params) 