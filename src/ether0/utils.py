import logging
import re
from http import HTTPStatus
from typing import TypeVar

import regex
from datasets import Dataset, DatasetDict, Version, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import HfHubHTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
)

logger = logging.getLogger(__name__)

# pylint: disable-next=invalid-name
TDataset = TypeVar("TDataset", bound=Dataset | DatasetDict)


@retry(
    retry=retry_if_exception(
        lambda x: (
            (
                # On 2/11/2025 James kept seeing on the g3 server cluster:
                # > huggingface_hub.errors.HfHubHTTPError: 504 Server Error: Gateway Time-out for
                # > url: https://huggingface.co/api/datasets/org/repo/paths-info/abc123
                # And on 3/14 James saw this on the g3 server cluster:
                # > huggingface_hub.errors.HfHubHTTPError: 502 Server Error: Bad Gateway for
                # > url: https://huggingface.co/api/datasets/org/repo/paths-info/abc123
                isinstance(x, HfHubHTTPError)
                and x.response.status_code
                in {HTTPStatus.BAD_GATEWAY.value, HTTPStatus.GATEWAY_TIMEOUT.value}
            )
            # On 4/14/2025 James kept seeing on the g5 server cluster:
            # > datasets.exceptions.DatasetNotFoundError:
            # > Dataset 'org/repo' doesn't exist on the Hub or cannot be accessed.
            or isinstance(x, DatasetNotFoundError)
        )
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
    wait=wait_fixed(5),
)
def load_dataset_retrying(
    path: str,
    revision: str | Version | None = None,
) -> DatasetDict:
    return load_dataset(path, revision=revision)


# SEE: https://www.compart.com/en/unicode/block/U+2070 for subscript letters
invalid_chars_regex = re.compile(
    r"[^A-Za-z0-9Α-Ωα-ωₐₑₒₓₔₕₖₗₘₙₚₛₜ⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉×\s!\"#$%&±⁻'´ʻ‘’ʼ“”()*+⁺,\-—–‐‑‒―−⏤./:;«<≤=≡≈≆≥>›»⇌?@[\\\]^_`{|}~←⇐→➔➞➛➡➟➧➭⇨⇒⇛⟺⇔⟶…]"  # noqa: RUF001
)
invalid_languages_regex = regex.compile(
    r"[\p{"
    + r"}\p{".join({
        # SEE: https://jrgraphix.net/r/Unicode/
        "Arabic",
        "Armenian",
        "Bengali",
        "Braille_Patterns",
        "Cyrillic",
        "Devanagari",
        "Ethiopic",
        "Georgian",
        "Gujarati",
        "Gurmukhi",
        "Han",
        "Hangul",
        "Hebrew",
        "Hiragana",
        "Kannada",
        "Katakana",
        "Khmer",
        "Latin_Extended_A",
        "Latin_Extended_Additional",
        "Latin_Extended_B",
        "Malayalam",
        "Myanmar",
        "Syriac",
        "Tamil",
        "Telugu",
        "Thaana",
        "Thai",
        "Tifinagh",
    })
    + r"}]"
)


def contains_invalid(
    text: str, chars: bool = False, languages: bool = False, threshold: int = 1
) -> tuple[bool, list[str]]:
    """Check if the text contains invalid characters or languages."""
    if chars:
        matches = invalid_chars_regex.findall(text)
        if len(matches) >= threshold:
            return True, sorted(matches)
    if languages:
        matches = invalid_languages_regex.findall(text)
        if len(matches) >= threshold:
            return True, sorted(matches)
    return False, []
