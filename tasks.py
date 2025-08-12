from typing import List, Callable, Dict, Tuple

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

    # aime math prompt (now uses ANSWER: format, matching gsm8k)
    aime_prompt = """
Please solve the above math problem step by step and provide the final answer at
the end of your response in the following format:

ANSWER: <answer>
"""

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

    items = [{"prompt": prompts[i], "answer": str(answers[i]).strip()} for i in range(len(prompts))]

    def aime_reward_fn(completions: List[str], *, items: List[Dict]) -> List[float]:
        """Reward function that checks if the AI's answer matches the expected answer using ANSWER: format."""
        import re
        rewards = []
        answer_pattern = r"ANSWER:\s*([^\n\r]+)"
        for i, completion in enumerate(completions):
            expected_answer = str(items[i]["answer"]).strip()
            matches = re.findall(answer_pattern, completion)
            if matches:
                # Take the last ANSWER: match
                extracted_answer = matches[-1].strip()
                # Remove any trailing punctuation or whitespace
                extracted_answer = extracted_answer.rstrip(". \n\r")
                # Compare as strings (could add more normalization if needed)
                if extracted_answer == expected_answer:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # No ANSWER: found, return 0
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
Please solve this math problem step by step and provide the final answer at
the end of your response in the following format:

ANSWER: <answer>
"""

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
            
            # Extract the answer by splitting on 'ANSWER:' and parsing as int
            parts = completion.split('ANSWER:')
            if len(parts) > 1:
                answer_str = parts[-1].strip().split()[0]  # Take first token after 'ANSWER:'
                try:
                    extracted_answer = str(int(answer_str))
                    matches = [extracted_answer]
                except Exception:
                    matches = []
            else:
                matches = []
            
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

def _gsm8k_calibrated_task(split: str = "train", **kwargs):
    '''
    Variant of gsm8k task that explicitly asks the model to use <think> </think> tags for thought,
    followed by a <confidence> </confidence> tag for confidence in the answer, finally followed by <answer> </answer> tag for
    user-facing answer.

    We extract the answer from the <answer> </answer> tag using "ANSWER: <answer>" format at the END of the answer.
    We also extract the confidence from the <confidence> </confidence> tag, using CONFIDENCE: <confidence> format at the END of the confidence.

    The confidence is a float between 0 and 1, with at most 2 decimal places.
    '''
    # Load the same dataset used by the regular GSM8K task
    dataset = load_dataset("skrishna/gsm8k_only_answer", split=split)

    # Calibrated prompt that enforces structure and explicit markers
    calibrated_prompt = """
You are solving a grade-school math word problem. Follow these rules strictly:

1) Think step by step inside <think> </think> tags. Do not include the final answer here.
2) Then state your confidence inside <confidence> </confidence> tags as a single number between 0 and 1 with at most 2 decimal places.
   - The last line inside <confidence> must be exactly: CONFIDENCE: <confidence>
3) Finally, provide the user-facing answer inside <answer> </answer> tags.
   - The last line inside <answer> must be exactly: ANSWER: <answer>

Example format:
<think>
...your reasoning...
</think>
<confidence>
... optional brief justification ...
CONFIDENCE: 0.85
</confidence>
<answer>
... succinct final answer ...
ANSWER: 42
</answer>
"""

    # Validate dataset schema
    if "text" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'text' column")
    if "label" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'label' column")

    # Extract questions and answers
    questions = dataset["text"]
    answers = dataset["label"]

    questions = list(questions) if not isinstance(questions, list) else questions
    answers = list(answers) if not isinstance(answers, list) else answers

    # Compose prompts in chat format
    prompts = []
    for question in questions:
        full_prompt = question + "\n\n" + calibrated_prompt
        prompts.append([{"role": "user", "content": full_prompt}])

    items = [{"prompt": prompts[i], "answer": str(answers[i])} for i in range(len(prompts))]

    def gsm8k_calibrated_reward_fn(completions: List[str], *, items: List[Dict]) -> Tuple[List[float], List[float]]:
        """Reward and confidence extractor.

        Reward = +1.0 if the answer extracted from the terminal 'ANSWER: <answer>' line inside <answer>...</answer> matches the dataset label,
        -1.0 otherwise, regardless of confidence.

        Returns:
            (rewards, confidences)
            - rewards: List[float] with values in {+1.0, -1.0}
            - confidences: List[float] with the parsed confidence value when valid, else NaN
        """
        import re

        rewards: List[float] = []
        confidences: List[float] = []

        # Regex patterns
        confidence_block_pattern = re.compile(r"<confidence>([\s\S]*?)</confidence>", re.IGNORECASE)
        answer_block_pattern = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
        terminal_conf_pattern = re.compile(r"CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE | re.MULTILINE)
        terminal_answer_pattern = re.compile(r"ANSWER:\s*([^\n\r]+)\s*$", re.IGNORECASE | re.MULTILINE)

        # Confidence format: 0-1 inclusive, at most 2 decimals
        def is_valid_confidence(conf_str: str) -> bool:
            # Accept 0, 1, 0.x, 0.xx, 1.0, 1.00
            if not re.fullmatch(r"(?:0(?:\.[0-9]{1,2})?|1(?:\.0{1,2})?)", conf_str):
                return False
            try:
                value = float(conf_str)
            except Exception:
                return False
            return 0.0 <= value <= 1.0

        for i, completion in enumerate(completions):
            expected_answer = str(items[i]["answer"]).strip()

            # Find confidence block and terminal CONFIDENCE line
            conf_block_match = confidence_block_pattern.search(completion)
            parsed_conf_value: float = float('nan')
            if conf_block_match:
                conf_block = conf_block_match.group(1)
                conf_lines = terminal_conf_pattern.findall(conf_block)
                if conf_lines:
                    last_conf = conf_lines[-1].strip()
                    if is_valid_confidence(last_conf):
                        try:
                            parsed_conf_value = float(last_conf)
                        except Exception:
                            parsed_conf_value = float('nan')

            # Find answer block and terminal ANSWER line
            answer_block_match = answer_block_pattern.search(completion)
            answer_ok = False
            if answer_block_match:
                ans_block = answer_block_match.group(1)
                ans_lines = terminal_answer_pattern.findall(ans_block)
                if ans_lines:
                    extracted = ans_lines[-1].strip()
                    # Align with gsm8k behavior: parse integer token if possible
                    token = extracted.split()[0]
                    try:
                        normalized = str(int(token))
                        answer_ok = (normalized == expected_answer)
                    except Exception:
                        answer_ok = False

            rewards.append(1.0 if answer_ok else -1.0)
            confidences.append(parsed_conf_value)

        return rewards, confidences

    return items, gsm8k_calibrated_reward_fn
# ----------------------------------------------------------------------------
# Public registry
# ----------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Callable[..., tuple]] = {
    "length": _length_task,
    "aime": _aime_task,
    "gsm8k": _gsm8k_task,
    'gsm8k-calibrated': _gsm8k_calibrated_task,
}


def get_task(task_name: str, **kwargs):
    """Return *(prompts, reward_fn)* pair for the requested *task_name*."""
    if task_name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {', '.join(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[task_name](**kwargs)
