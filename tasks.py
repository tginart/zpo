from typing import List, Callable, Dict

from datasets import load_dataset  # type: ignore

# ----------------------------------------------------------------------------
# Reward functions
# ----------------------------------------------------------------------------


def _length_reward_fn(completions: List[str], *, items: List[Dict] = None, **kwargs) -> List[float]:
    """Simple reward: number of whitespace-separated tokens in the completion."""
    return [float(len(text.split())) / 1024.0 for text in completions]


# ----------------------------------------------------------------------------
# Task loaders
# ----------------------------------------------------------------------------


def _length_task(num_prompts: int = 2, **kwargs):
    """Synthetic *length* task.

    Returns a small fixed set of prompts together with the length-based reward
    function defined above.  The *num_prompts* argument is kept only to keep a
    consistent signature with other task loaders and is otherwise ignored.
    
    Note: Prompts are returned in conversational format (list of message dicts)
    to ensure proper chat template handling with models that expect user/assistant
    message structure.
    """
    default_prompts = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "Write me a short poem about cats."}],
        [{"role": "user", "content": "Write me a short poem about dogs."}],
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "How is the weather in Tokyo?"}],
        [{"role": "user", "content": "Can you please write a short story about a cat?"}],
        [{"role": "user", "content": "Can you please make me feel good?"}],
        [{"role": "user", "content": "What do you think is the best way to learn programming?"}],
    ]
    items = [{"prompt": p} for p in default_prompts]
    return items, _length_reward_fn


def _aime_task(split: str = "train", **kwargs):
    """Task based on the *di-zhang-fdu/AIME_1983_2024* dataset (AIME math problems).

    This loader fetches all prompts from the chosen *split* of the dataset.
    Uses the "Question" column for prompts and "Answer" column for verification.
    
    Note: Prompts are returned in conversational format (list of message dicts)
    to ensure proper chat template handling with models that expect user/assistant
    message structure.
    """
    dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split=split)

    # aime math prompt
    aime_prompt = """
The answer to the question above is an integer between 0 and 999.
Please reason step by step, and put your final answer within \\boxed{}."""

    # Use the "Question" column for prompts
    if "Question" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'Question' column")
    
    # Use the "Answer" column for verification
    if "Answer" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'Answer' column")

    # Get all questions and answers
    questions = dataset["Question"]
    answers = dataset["Answer"]
    
    # Convert to lists if they're not already
    questions = list(questions) if not isinstance(questions, list) else questions
    answers = list(answers) if not isinstance(answers, list) else answers
    
    # Append the aime_prompt to each question
    prompts = []
    for question in questions:
        full_prompt = question + aime_prompt
        prompts.append([{"role": "user", "content": full_prompt}])
    
    items = [{"prompt": prompts[i], "answer": answers[i]} for i in range(len(prompts))]
    
    def aime_reward_fn(completions: List[str], *, items: List[Dict]) -> List[float]:
        """Reward function that checks if the AI's answer matches the expected answer."""
        rewards = []
        
        for i, completion in enumerate(completions):
            expected_answer = str(items[i]["answer"])
            
            # Extract answer from \boxed{} at the end of the response
            import re
            boxed_pattern = r'\\boxed\{([^}]+)\}'
            matches = re.findall(boxed_pattern, completion)
            
            if matches:
                # Take the last \boxed{} match
                extracted_answer = matches[-1].strip()
                # Check if the extracted answer matches the expected answer
                if extracted_answer == expected_answer:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # No \boxed{} found, return 0
                rewards.append(0.0)
        
        return rewards

    return items, aime_reward_fn


def _gsm8k_task(split: str = "train", **kwargs):
    """Task based on the *skrishna/gsm8k_only_answer* dataset (GSM8K math problems).

    This loader fetches all prompts from the chosen *split* of the dataset.
    Uses the "text" column for prompts and "label" column for verification.
    
    Note: Prompts are returned in conversational format (list of message dicts)
    to ensure proper chat template handling with models that expect user/assistant
    message structure.
    """
    dataset = load_dataset("skrishna/gsm8k_only_answer", split=split)

    # gsm8k math prompt
    gsm8k_prompt = """
Please solve this math problem step by step and provide the final answer."""

    # Use the "text" column for prompts
    if "text" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'text' column")
    
    # Use the "label" column for verification
    if "label" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'label' column")

    # Get all questions and answers
    questions = dataset["text"]
    answers = dataset["label"]
    
    # Convert to lists if they're not already
    questions = list(questions) if not isinstance(questions, list) else questions
    answers = list(answers) if not isinstance(answers, list) else answers
    
    # Append the gsm8k_prompt to each question
    prompts = []
    for question in questions:
        full_prompt = question + gsm8k_prompt
        prompts.append([{"role": "user", "content": full_prompt}])
    
    items = [{"prompt": prompts[i], "answer": answers[i]} for i in range(len(prompts))]

    def gsm8k_reward_fn(completions: List[str], *, items: List[Dict]) -> List[float]:
        """Reward function that checks if the AI's answer matches the expected answer."""
        rewards = []
        
        for i, completion in enumerate(completions):
            expected_answer = str(items[i]["answer"])
            
            # Extract the final numerical answer from the completion
            import re
            # Look for numbers at the end of the response, possibly preceded by "answer", "=", etc.
            number_pattern = r'(?:answer|result|solution|final answer|answer is|result is|solution is|final answer is|answer:|result:|solution:|final answer:)?\s*[=:]\s*(\d+(?:\.\d+)?)'
            matches = re.findall(number_pattern, completion.lower())
            
            if matches:
                # Take the last number match as the final answer
                extracted_answer = matches[-1].strip()
                # Check if the extracted answer matches the expected answer
                if extracted_answer == expected_answer:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # No number found, return 0
                rewards.append(0.0)
        
        return rewards

    return items, gsm8k_reward_fn


# ----------------------------------------------------------------------------
# Public registry
# ----------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Callable[..., tuple]] = {
    "length": _length_task,
    "aime": _aime_task,
    "gsm8k": _gsm8k_task,
}


def get_task(task_name: str, **kwargs):
    """Return *(prompts, reward_fn)* pair for the requested *task_name*."""
    if task_name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {', '.join(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[task_name](**kwargs)
