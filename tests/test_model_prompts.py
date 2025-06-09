import pytest

from ether0.model_prompts import (
    ANSWER_END,
    ANSWER_START,
    THINK_END,
    THINK_START,
    ProblemPrompt,
    extract_answer_loose,
    extract_thought_answer_strict,
)


def test_problem_prompt() -> None:
    none_prompt = ProblemPrompt.NONE.get_prompt()
    assert isinstance(none_prompt, str)
    assert "think" not in none_prompt
    assert "answer" not in none_prompt

    answer_prompt = ProblemPrompt.ANSWER.get_prompt()
    assert isinstance(answer_prompt, str)
    assert "think" not in answer_prompt
    assert "answer" in answer_prompt

    think_answer_prompt = ProblemPrompt.THINK_ANSWER.get_prompt()
    assert isinstance(think_answer_prompt, str)
    assert "think" in think_answer_prompt
    assert "answer" in think_answer_prompt


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("<answer>CCO</answer>", "CCO", id="base"),
        pytest.param("<answer></answer>", "", id="empty-answer"),
        pytest.param("<answer> </answer>", "", id="space-answer"),
        pytest.param("\n<answer>CCO</answer>", "CCO", id="base-extra-whitespace-1"),
        pytest.param("\n<answer>CCO</answer>\n", "CCO", id="base-extra-whitespace-2"),
        pytest.param(" <answer>CCO</answer> ", "CCO", id="base-extra-whitespace-3"),
        pytest.param("word<answer>CCO</answer> ", "CCO", id="base-extra-whitespace-4"),
        pytest.param("<answer>\nCCO\n</answer>", "CCO", id="base-w-newlines"),
        pytest.param(
            "<answer> \nCCO\n</answer>", "CCO", id="base-w-spaces-and-newlines"
        ),
        pytest.param(
            "<answer>\n\nCCO\n\n</answer>", "CCO", id="base-w-double-newlines"
        ),
        pytest.param("<answer> CCO </answer>", "CCO", id="base-w-spaces"),
        pytest.param(
            "<answer> < CCO</answer>", "< CCO", id="base-contains-potential-xml-1"
        ),
        pytest.param(
            "<answer> <stub CCO</answer>",
            "<stub CCO",
            id="base-contains-potential-xml-2",
        ),
        pytest.param("<answer><answer>CCO</answer></answer>", "", id="nested"),
        pytest.param(
            "<answer>\n<answer>\nCCO\n</answer>\n</answer>", "", id="nested-w-newlines"
        ),
        pytest.param(
            "<think>thought</think><answer>CCO</answer>", "CCO", id="thought-base"
        ),
        pytest.param(
            "<think>thought</think><answer>\nCCO\n</answer>",
            "CCO",
            id="thought-base-w-newlines",
        ),
        pytest.param(
            "<answer>CCO</answer><think>thought</think>", "CCO", id="thought-reversed"
        ),
        pytest.param(
            "<answer>\nCCO\n</answer><think>thought</think>",
            "CCO",
            id="thought-reversed-w-newlines",
        ),
        pytest.param(
            "<answer>echoing prompt</answer><answer>CCO2</answer>",
            "CCO2",
            id="multi-answer",
        ),
        pytest.param(
            "<answer>echoing prompt</answer><answer>\nCCO2\n</answer>",
            "CCO2",
            id="multi-answer-w-newlines",
        ),
    ],
)
def test_extract_answer_loose(content: str | None, expected: str) -> None:
    assert extract_answer_loose(content) == expected


@pytest.mark.parametrize(
    ("content", "expected_answer", "expected_thought"),
    [
        pytest.param(f"{ANSWER_START}CCO{ANSWER_END}", None, None, id="no-thought"),
        pytest.param(
            f"{ANSWER_START}{ANSWER_END}", None, None, id="no-thought-empty-answer"
        ),
        pytest.param(
            f"{ANSWER_START} {ANSWER_END}", None, None, id="no-thought-space-answer"
        ),
        pytest.param(
            f"{ANSWER_START}\nCCO\n{ANSWER_END}",
            None,
            None,
            id="no-thought-w-newlines",
        ),
        pytest.param(
            f"{ANSWER_START}{ANSWER_START}CCO{ANSWER_END}{ANSWER_END}",
            None,
            None,
            id="no-thought-nested",
        ),
        pytest.param(
            f"{ANSWER_START}\n{ANSWER_START}\nCCO\n{ANSWER_END}\n{ANSWER_END}",
            None,
            None,
            id="no-thought-nested-w-newlines",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought",
            id="base",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START}{ANSWER_END}",
            None,
            "thought",
            id="empty-answer",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START} {ANSWER_END}",
            None,
            "thought",
            id="space-answer",
        ),
        pytest.param(
            # Allow models to place up to one whitespace before the thought
            f"\n{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought",
            id="base-extra-whitespace-1",
        ),
        pytest.param(
            f"\n{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}\n",
            None,
            None,
            id="base-extra-whitespace-2",
        ),
        pytest.param(
            f" {THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END} ",
            None,
            None,
            id="base-extra-whitespace-3",
        ),
        pytest.param(
            f"word{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}\n",
            None,
            None,
            id="base-extra-whitespace-4",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START}\nCCO\n{ANSWER_END}",
            "CCO",
            "thought",
            id="base-w-newlines",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START} \nCCO\n{ANSWER_END}",
            "CCO",
            "thought",
            id="base-w-spaces-and-newlines",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START}\n\nCCO\n\n{ANSWER_END}",
            "CCO",
            "thought",
            id="base-w-double-newlines",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START} CCO {ANSWER_END}",
            "CCO",
            "thought",
            id="base-w-spaces",
        ),
        pytest.param(
            f"{THINK_START}\nthought\n{THINK_END}{ANSWER_START}\nCCO\n{ANSWER_END}",
            "CCO",
            "thought",
            id="base-w-newlines-both",
        ),
        pytest.param(
            f"{THINK_START}thought\nthought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought\nthought",
            id="base-inner-newline",
        ),
        pytest.param(
            f"{THINK_START}\nthought\nthought\n{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought\nthought",
            id="base-inner-newline-w-newlines",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}inter{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought",
            id="base-inter",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}inter\ninter{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought",
            id="base-inter-inner-newline",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}\ninter\ninter\n{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought",
            id="base-inter-inner-newline-w-newlines",
        ),
        pytest.param(
            f"{ANSWER_START}CCO{ANSWER_END}{THINK_START}thought{THINK_END}",
            None,
            None,
            id="base-reversed",
        ),
        pytest.param(
            f"{ANSWER_START}\nCCO\n{ANSWER_END}{THINK_START}thought{THINK_END}",
            None,
            None,
            id="base-reversed-w-newlines",
        ),
        pytest.param(
            f"{THINK_START}thought < thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought < thought",
            id="thought-contains-potential-xml-1",
        ),
        pytest.param(
            f"{THINK_START}thought <stub"
            f" thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought <stub thought",
            id="thought-contains-potential-xml-2",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}a",
            None,
            None,
            id="text-after-answer",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            None,
            None,
            id="multi-thought",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}<thought>thought</thought>{ANSWER_START}CCO{ANSWER_END}",
            "CCO",
            "thought",
            id="interleaved-think-thought",
        ),
        pytest.param(
            f"{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}Some"
            f" text{THINK_START}thought{THINK_END}{ANSWER_START}CCO{ANSWER_END}",
            None,
            None,
            id="continuation",
        ),
        pytest.param(
            f"{ANSWER_START}echoing prompt{ANSWER_END}{ANSWER_START}CCO2{ANSWER_END}",
            None,
            None,
            id="no-thought-multi-answer",
        ),
        pytest.param(
            f"{ANSWER_START}echoing"
            f" prompt{ANSWER_END}{ANSWER_START}\nCCO2\n{ANSWER_END}",
            None,
            None,
            id="no-thought-multi-answer-w-newlines",
        ),
        pytest.param(
            f"{THINK_START}\nAlright, so I need to figure out the IUPAC name for the"
            " molecule with the formula C1=CC(=CC=C1O)O. Let me start by trying to"
            " visualize the structure. The formula seems a bit complex, so breaking it"
            " down might help.\n\nFirst, I notice there's a ring structure because of"
            " the C1 notation, which suggests a cyclic compound. The presence of"
            " double bonds (the = signs) indicates that it's not just a simple alkane."
            " So, I'm thinking it's a cyclic diene or something similar.\n\nLooking"
            " closer at the formula, I see two oxygen atoms attached to the ring. The"
            " first O is attached to a carbon that's part of a double bond (C1=CC...),"
            " and the second O is attached to another carbon that's also part of a"
            " double bond. So, there are two ketone groups or possibly ester groups?"
            " Wait, no, the formula is C1=CC(=CC=C1O)O, which might imply that each"
            " carbon attached to the ring has an oxygen, but let me try to count the"
            " bonds properly.\n\nWait, perhaps I should draw this out. Let me imagine"
            " the ring. Carbon 1 (C1) is double-bonded to a carbon (C2). Then, C2 is"
            " connected to another carbon (C3) via a double bond, which is then"
            " connected to C4, and so on, until I come back to C1, forming a ring. But"
            " since there are multiple double bonds, it's probably a conjugated diene"
            " or something like that.\n\nWait, maybe it's a cyclopentadiene"
            " derivative. But the presence of two oxygen atoms makes it more likely to"
            " be a dienol ether or something similar. Alternatively, it could be a"
            " cyclohexene derivative with two ketone groups, but I'm not sure.\n\nLet"
            " me count the carbons. The formula is C1=CC(=CC=C1O)O. So, each 'C' is"
            " part of the ring. Let me see: C1 is double-bonded to C2, then C2 is"
            " connected via a single bond to another group, which is C3, which is"
            " double-bonded to C4, and so on until I close the ring back to C1. Hmm,"
            " maybe it's a cyclopentadiene ring with substituents.\n\nWait, perhaps"
            " it's a cyclopentadienyl ether. Let me think about how the substituents"
            " are arranged. If C1 is connected via a double bond to C2, then C2 has"
            " another substituent, which is C3 via a double bond, and so on. Maybe the"
            " structure is such that there are two adjacent double bonds, making it a"
            " conjugated diene.\n\nBut I'm getting a bit confused. Maybe I should"
            " approach this differently. Let's try to write out the structure step by"
            " step.\n\nStarting with C1, which is double-bonded to C2: C1=C2. Then, C2"
            " is connected to another carbon, which is part of a double bond to C3:"
            " C2=C3. So, now we have C1=C2-C3=... Then, C3 is connected to C4, which"
            " is connected back to C1, forming a ring. So, the ring would be a"
            " five-membered ring with two double bonds.\n\nWait, but a five-membered"
            " ring with two double bonds would be conjugated diene in a ring. So,"
            " that's cyclopentadiene. Now, each of the carbons in the double bonds has"
            " an oxygen substituent. So, C1 has an O attached, and C3 also has an O"
            " attached. So, it's cyclopentadiene-1,3-dione? But wait, the formula is"
            " C1=CC(=CC=C1O)O, which suggests that each double bond carbon has an O"
            " attached. So, both C1 and C3 have O substituents.\n\nBut"
            " cyclopentadiene-1,3-dione would have two ketone groups, which would make"
            " it a diketone. Alternatively, it could be a dienol ether, but with two"
            " oxygen atoms attached to the ring.\n\nWait, another possibility is that"
            " it's a cyclopentadienyl ether with two substituents. Alternatively,"
            " perhaps it's a substituted cyclopentadienone.\n\nWait, maybe I should"
            " count the number of atoms. The formula is C1=CC(=CC=C1O)O. Let's parse"
            " this:\n\n- C1 is connected via double bond to C2.\n- C2 is connected to"
            " another carbon (let's say C3) which is double-bonded to C4.\n- C4 is"
            " connected back to C1, forming a ring.\n- Each of C1 and C3 has an oxygen"
            " attached.\n\nWait, that would make a five-membered ring with two double"
            " bonds and two oxygen atoms. So, perhaps it's cyclopentadienone-1,3-dioic"
            " acid? No, that doesn't sound right because the formula doesn't indicate"
            " acid groups.\n\nAlternatively, maybe it's a dienol ether, where two of"
            " the double bond carbons have ether oxygen substituents.\n\nWait, perhaps"
            " the correct name is something like 1,3-dihydroxycyclopentadiene or"
            " similar. Alternatively, since the oxygens are on the double bond"
            " carbons, maybe it's a dienol ether.\n\nWait, I'm getting stuck. Maybe I"
            " should think about the structure again.\n\nThe formula is"
            " C1=CC(=CC=C1O)O. So, let's parse the SMILES notation:\n\n-"
            " C1=CC(=CC=C1O)O\nBreaking it down:\n- C1 is the first carbon in a"
            " ring.\n- C1 is double-bonded to C2: C1=C2\n- C2 is connected to another"
            " carbon, C3: C2-C3\n- C3 has a double bond to C4: C3=C4\n- C4 is"
            " connected back to C1, closing the ring: C4=C1\n- Additionally, C1 and C3"
            " each have an oxygen attached: C1=O and C3=O\n\nWait, that would mean"
            " both C1 and C3 are double-bonded to O. So, it's a cyclopentadiene ring"
            " with two ketone groups. So, the IUPAC name would be something like"
            " cyclopentadien-1,3-dione.\n\nWait, but isn't cyclopentadien-1,3-dione a"
            " thing? Let me check. Yes, in organic chemistry, cyclopentadien-1,3-dione"
            " is a known compound, also called 1,3-cyclopentadienedione.\n\nBut in the"
            " given formula, the oxygens are in positions 1 and 3, connected via"
            " double bonds. So, the IUPAC name would be"
            " cyclopentadien-1,3-dione.\n\nWait, but sometimes the numbering might be"
            " different. Let me think about the numbering in the ring. Starting at C1,"
            " moving to C2, C3, C4, C5, and back to C1.\n\nWait, no, in this case, the"
            " ring would be five-membered, with double bonds at C1-C2 and C3-C4, and"
            " oxygens attached to C1 and C3. So, the correct IUPAC name would be"
            " cyclopentadien-1,3-dione.\n\nAlternatively, perhaps it's named as"
            " 1,3-diketo-cyclopentadiene, but I think the more standard name would be"
            " cyclopentadien-1,3-dione.\n\nWait, but let me make sure. The correct"
            " IUPAC name would assign numbers starting from the first substituent. So,"
            " C1 has an oxygen, C3 has an oxygen, so the substituents are at positions"
            " 1 and 3.\n\nTherefore, the IUPAC name would be"
            " cyclopentadien-1,3-dione.\n\nBut wait, sometimes the parent structure is"
            " considered in a way that gives the lowest possible numbers."
            " Alternatively, it might be named as 1,3-dicyclohexa-1,3-diene-1,3-dione,"
            " but no, that's not right because it's a five-membered ring.\n\nI think"
            " I'm confident that the correct IUPAC name is"
            f" cyclopentadien-1,3-dione.\n{THINK_END}\n\nThe IUPAC name of the molecule"
            " with the formula C1=CC(=CC=C1O)O is **cyclopentadien-1,3-dione**."
            f" \n\n{ANSWER_START} cyclopentadien-1,3-dione {ANSWER_END}",
            "cyclopentadien-1,3-dione",
            None,
            id="actual-case-1",
        ),
        pytest.param(
            f"{THINK_START} I need to complete the molecule by adding a functional"
            " group or atom to it. To do this, I'll consider the existing structure"
            " and choose a suitable group that can be added."
            f" {THINK_END}\n{ANSWER_START} ClC1=CC(Cl)=CC(C2=C(/octet)2)"
            f" {ANSWER_END}\n\nWhat is a valid completion of this"
            f" molecule:\nClC1=CC(Cl)=CC(C2=C(/octet\n/octetassistant\n{THINK_START} To"
            " complete the molecule, I need to consider the existing structure and"
            " determine what type of functional group or atom can be added. I'll"
            " analyze the existing bonds and determine the most suitable option."
            f" {THINK_END}\n{ANSWER_START} ClC1=CC(Cl)=CC(C2=C(/octet)2) {ANSWER_END}",
            None,
            None,
            id="actual-case-2",
        ),
    ],
)
def test_extract_answer_thought_strict_reasoning(
    content: str, expected_answer: str | None, expected_thought: str | None
) -> None:
    thought, answer = extract_thought_answer_strict(content, reasoning=True)
    assert answer == expected_answer
    if expected_thought:
        assert thought == expected_thought


@pytest.mark.parametrize(
    ("content", "expected_answer"),
    [(
        "<|answer_start|>Clc1ccc(cc1)OCCOC(=O)COC(=O)CCNC(=O)COc2ccccc2<|answer_end|>",
        "Clc1ccc(cc1)OCCOC(=O)COC(=O)CCNC(=O)COc2ccccc2",
    )],
)
def test_extract_answer_thought_strict_no_reasoning(
    content: str, expected_answer: str | None
) -> None:
    thought, answer = extract_thought_answer_strict(content, reasoning=False)
    assert answer == expected_answer
    assert thought is None
