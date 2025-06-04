"""Prompts and utilities used for training the ether0 model."""

import re
from enum import Enum, StrEnum
from typing import assert_never

# Tokens to surround reasoning and answer in XML format
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
ANSWER_START = "<|answer_start|>"
ANSWER_END = "<|answer_end|>"


# Keys: True (reasoning + answer), False (answer only)
# Use strict regex for ether0 models, as we can SFT or RL the models into compliance
STRICT_XML_ANSWER_SPLIT_PATTERNS: dict[bool, re.Pattern] = {
    True: re.compile(
        rf"^\s?{re.escape(THINK_START)}\s*([\s\S]*?)\s*{re.escape(THINK_END)}([\s\S]*?){re.escape(ANSWER_START)}\s*([\s\S]*?)\s*{re.escape(ANSWER_END)}$"
    ),
    False: re.compile(
        rf"^\s?{re.escape(ANSWER_START)}\s*(\S[\s\S]*?)\s*{re.escape(ANSWER_END)}$"
    ),
}
# Use loose regex for other models because:
# 1. <think> may be out-of-distribution from the model's training data,
#    so requiring thoughts may degrade performance.
# 2. We allow baseline models to add extra whitespace and/or preceding or trailing text
#    around answer XML, again to maximize performance.
# 3. Similarly, we allow models to ramble for a bit mentioning <answer>,
#    and then we just keep the last <answer> XML.
# 4. We want to avoid prompt engineering tricks to get around the previous items.
LOOSE_XML_ANSWER_LOOSE_PATTERN = r"<answer>\s*(\S[\s\S]*?)\s*<\/answer>"


class XMLAnswerPrompts(StrEnum):
    """Enum of prompts to use ."""

    REASONING_ANSWER = (
        "A conversation between User and Assistant."
        " The user asks a question, and the Assistant solves it."
        " The assistant first thinks about the reasoning process"
        " in the mind and then provides the user with the answer."
        " The reasoning process and answer are enclosed within"
        f" {THINK_START} {THINK_END} and {ANSWER_START} {ANSWER_END} tags,"
        " respectively, i.e.,"
        f" {THINK_START} reasoning process here {THINK_END}"
        f"{ANSWER_START} answer here {ANSWER_END}"
    )
    ANSWER_ONLY = (
        "A conversation between User and Assistant."
        " The user asks a question, and the Assistant solves it."
        " The assistant encloses its answer within"
        f" {ANSWER_START} {ANSWER_END} tags, i.e.,"
        f" {ANSWER_START} answer here {ANSWER_END}"
    )

    @property
    def pattern(self) -> re.Pattern:
        return STRICT_XML_ANSWER_SPLIT_PATTERNS[
            self == XMLAnswerPrompts.REASONING_ANSWER
        ]


class SysPrompt(Enum):  # Use Enum over StrEnum for trl.TrlParser compatibility
    """Possible system prompts for making a conversation to train upon."""

    SCIENTIFIC_AI = "scientific_ai"

    def get_sys_prompt(self) -> str:
        match self:
            case SysPrompt.SCIENTIFIC_AI:
                return "You are a scientific reasoning AI assistant."
            case _:
                assert_never(self)


class ProblemPrompt(Enum):  # Use Enum over StrEnum for trl.TrlParser compatibility
    """Possible user prompts for making a conversation to train upon."""

    NONE = "none"
    THINK_ANSWER = "think_answer"
    ANSWER = "answer"

    def get_prompt(self) -> str:
        match self:
            case ProblemPrompt.NONE:
                return ""
            case ProblemPrompt.THINK_ANSWER:
                return XMLAnswerPrompts.REASONING_ANSWER
            case ProblemPrompt.ANSWER:
                return XMLAnswerPrompts.ANSWER_ONLY
            case _:
                assert_never(self)


def extract_thought_answer_strict(
    text: str, reasoning: bool
) -> tuple[str | None, str | None]:
    """Extract thought and answer from text using a strict XML pattern."""
    # Use `maxsplit=1` to enforce just one match
    matches = STRICT_XML_ANSWER_SPLIT_PATTERNS[reasoning].split(text, maxsplit=1)
    try:
        _, *inner, suffix = matches
    except (IndexError, ValueError):
        return None, None  # Consider no answer or 2+ answers as a failure
    if reasoning:
        thought, inter, answer = inner
    else:
        thought, inter = None, None
        (answer,) = inner
    if (
        THINK_START not in (thought or "")
        and THINK_START not in (inter or "")
        and ANSWER_START not in answer
        and not suffix
    ):
        return thought, answer or None
    return None, None  # Consider nested answer as a failure


def extract_answer_loose(text: str | None) -> str:
    """
    Extract thought and answer from text using a loose XML pattern.

    SEE: LOOSE_XML_ANSWER_LOOSE_PATTERN for when to use this.
    """
    matches = re.findall(LOOSE_XML_ANSWER_LOOSE_PATTERN, text or "")
    try:
        last_answer = matches[-1]  # Last answer in the response
    except IndexError:
        return ""  # Consider no answer as a failure
    if "<answer>" not in last_answer:
        return last_answer
    return ""  # Consider nested answer as a failure
