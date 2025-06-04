import logging
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("ETHER0_REMOTES_API_BASE_URL")
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('ETHER0_REMOTES_API_TOKEN')}",
    "Content-Type": "application/json",
}
SERVER_ERRORS_COUNTER = Counter({
    "fetch_solubility": 0,
    "fetch_purchasable": 0,
    "fetch_forward_rxn": 0,
    "fetch_rxn_info": 0,
})
THROW_500_ERROR_THRESHOLD = int(
    os.environ.get("ETHER0_REMOTES_THROW_500_ERROR_THRESHOLD", "100")
)
# If our server throws a 501, we don't retry
OUR_SERVER_DONT_RETRY_CODE = httpx.codes.NOT_IMPLEMENTED.value
REMOTE_WORKER_COLD_START_TIME = 180  # sec


class RetryableServerError(Exception):
    """Retryable server error."""

    @classmethod
    def check_raise(
        cls, response: httpx.Response, kwargs: Mapping[str, Any] | None = None
    ) -> None:
        if (
            response.is_server_error
            and response.status_code != OUR_SERVER_DONT_RETRY_CODE
        ):
            raise cls(
                f"Retryable server error with status code {response.status_code}"
                f" and inputs {kwargs or {}} and response {response=}."
            )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((
        httpx.ReadTimeout,
        httpx.ConnectError,
        RetryableServerError,
    )),
)
def fetch_solubility(query_smiles: str) -> dict:
    response = httpx.post(
        f"{BASE_URL}/compute_solubility",
        json={"smiles": query_smiles},
        headers=HEADERS,
        timeout=REMOTE_WORKER_COLD_START_TIME,
    )

    error_message = ""
    if response.is_success:
        result = response.json()
        if "error" in result:
            error_message = result["error"]
        else:
            solubility = result["mean"]
            return {"smiles": query_smiles, "solubility": solubility}
    if response.is_redirect or response.is_server_error:
        # We should not have redirect responses or server errors, so let's retry these
        error_message = response.text
        SERVER_ERRORS_COUNTER["fetch_solubility"] += 1
        if SERVER_ERRORS_COUNTER["fetch_solubility"] >= THROW_500_ERROR_THRESHOLD:
            response.raise_for_status()
        RetryableServerError.check_raise(
            response, kwargs={"query_smiles": query_smiles}
        )
    if error_message:
        logger.warning(
            f"fetch_solubility did not succeed on {query_smiles=} with"
            f" {response=} and {error_message=}."
        )
    return {
        "smiles": query_smiles,
        "error": f"API error: {response} - {error_message}",
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((
        httpx.ReadTimeout,
        httpx.ConnectError,
        RetryableServerError,
    )),
)
def fetch_purchasable(query_smiles_list: list[str] | str) -> dict[str, bool]:
    response = httpx.post(
        f"{BASE_URL}/is_purchasable",
        json={"smiles": query_smiles_list},
        headers=HEADERS,
        timeout=REMOTE_WORKER_COLD_START_TIME,
    )

    if response.is_success:
        return response.json()
    logger.warning(
        f"fetch_purchasable did not succeed on {query_smiles_list=} with"
        f" {response=} and {response.text=}."
    )
    if response.is_redirect or response.is_server_error:
        # We should not have redirect responses or server errors, so let's retry these
        SERVER_ERRORS_COUNTER["fetch_purchasable"] += 1
        if SERVER_ERRORS_COUNTER["fetch_purchasable"] >= THROW_500_ERROR_THRESHOLD:
            response.raise_for_status()
        RetryableServerError.check_raise(
            response, kwargs={"query_smiles_list": query_smiles_list}
        )
    return {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((
        httpx.ReadTimeout,
        httpx.ConnectError,
        RetryableServerError,
    )),
)
def fetch_forward_rxn(query_rxn_smiles: str) -> dict[str, str]:
    response = httpx.post(
        f"{BASE_URL}/translate",
        json={"reaction": query_rxn_smiles},
        headers=HEADERS,
        timeout=REMOTE_WORKER_COLD_START_TIME,
    )

    if response.is_success:
        result = response.json()
        product = result["product"]
        return {"smiles": query_rxn_smiles, "product": product}
    logger.warning(
        f"fetch_forward_rxn did not succeed on {query_rxn_smiles=} with"
        f" {response=} and {response.text=}."
    )
    if response.is_redirect or response.is_server_error:
        # We should not have redirect responses or server errors, so let's retry these
        SERVER_ERRORS_COUNTER["fetch_forward_rxn"] += 1
        if SERVER_ERRORS_COUNTER["fetch_forward_rxn"] >= THROW_500_ERROR_THRESHOLD:
            response.raise_for_status()
        RetryableServerError.check_raise(
            response, kwargs={"query_rxn_smiles": query_rxn_smiles}
        )
    return {
        "smiles": query_rxn_smiles,
        "error": f"API error: {response} - {response.text}",
    }
