import pytest

from ether0.utils import contains_invalid


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("COC-C=O-C=NC(=O)", (False, []), id="smiles"),
        pytest.param("Normal text", (False, []), id="plain-english-1"),
        pytest.param("موعد", (True, ["د", "ع", "م", "و"]), id="has-arabic-1"),
        pytest.param(
            "having a methyl[,mحصلة نفيسدكم](=O)",
            (True, ["ة", "ح", "د", "س", "ص", "ف", "ك", "ل", "م", "ن", "ي"]),
            id="has-arabic-2",
        ),
    ],
)
def test_contains_invalid_languages(
    text: str, expected: tuple[bool, list[str]]
) -> None:
    assert contains_invalid(text, languages=True) == expected
