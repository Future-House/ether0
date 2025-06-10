from unittest.mock import patch

from lighteval.main_tasks import list as lighteval_list
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.tasks.requests import Doc

import ether0.lighteval_tasks


def test_task_list(capsys) -> None:
    """Integration test designed to test TASKS_TABLE and custom task creation."""
    with patch(  # Work around https://github.com/huggingface/lighteval/issues/805
        "lighteval.tasks.registry.create_custom_tasks_module",
        side_effect=[ether0.lighteval_tasks],
    ):
        lighteval_list(custom_tasks=ether0.lighteval_tasks.__file__)
    captured = capsys.readouterr()
    assert not captured.err
    tasks = [row for row in captured.out.splitlines() if "ether0" in row]
    assert len(tasks) > 1, "Expected some ether0 tasks"
    assert any(
        "functional-group" in row for row in tasks
    ), "Expected specific tasks to be listed"
    # TODO: after https://github.com/huggingface/lighteval/issues/806,
    # remove the .litellm_cache directory created by this test importing from LightEval


def test_accuracy_metric() -> None:
    accuracy_metric = getattr(
        Metrics, ether0.lighteval_tasks.ETHER0_ACCURACY_METRIC_NAME
    ).value
    assert isinstance(accuracy_metric, SampleLevelMetric)

    # NOTE: these inputs were taken from a gpt-4o baseline run
    doc_json = {
        "query": (
            "When answering, be sure to place the final answer as SMILES notation into"
            " XML tags <answer></answer>. An example is <answer>CCO</answer>.\n\nWhat"
            " is a valid completion of this molecule:\nO=C(OCC1=CC=CC=C1)N1CCCC1C(=O"
        ),
        "choices": [""],
        "gold_index": 0,
        "original_query": "",
        "specific": {
            "solution": (
                "valid_mol_eval!:!O=C(OCC1=CC=CC=C1)N1CCCC1C(=O!:!molecule-completion"
            ),
            "id": "e8b8bb34-731a-46e1-93a2-b6330a705148",
            "soft": False,
            "test": True,
            "reasoning": False,
        },
        "task_name": "community|ether0:loose:molecule-completion",
        "instruction": "",
        "ctx": [{
            "role": "user",
            "content": (
                "When answering, be sure to place the final answer as SMILES notation"
                " into XML tags <answer></answer>. An example is"
                " <answer>CCO</answer>.\n\nWhat is a valid completion of this"
                " molecule:\nO=C(OCC1=CC=CC=C1)N1CCCC1C(=O"
            ),
        }],
        "num_asked_few_shots": 0,
        "num_effective_few_shots": 0,
    }
    assert (
        accuracy_metric.sample_level_fn(
            predictions=[
                "The given fragment of the molecule O=C(OCC1=CC=CC=C1)N1CCCC1C(=O suggests"
                " a structure that indicates an amide linkage with a substituted"
                " cyclohexanone. A plausible completion of this structure is a standard"
                " cyclohexanone amide. Therefore, a valid SMILES notation for the completed"
                " structure is:\n\n<answer>O=C(OCC1=CC=CC=C1)N1CCCC1C(=O)C2CCCCC2</answer>"
            ],
            formatted_doc=Doc(**doc_json),
            golds=[""],
        )
        == 1.0
    )
