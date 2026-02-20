import os
import unittest
from unittest import mock

import etl.tinvest_client as tinvest_client


class SandboxSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def _clear_relevant_env(self) -> None:
        for name in (
            "TINVEST_ENV",
            "TINVEST_SANDBOX_TOKEN",
            "INVEST_TOKEN",
            "TINVEST_TOKEN",
            "TOKEN",
        ):
            os.environ.pop(name, None)

    def test_non_sandbox_env_raises(self) -> None:
        self._clear_relevant_env()
        os.environ["TINVEST_ENV"] = "prod"
        os.environ["TINVEST_SANDBOX_TOKEN"] = "sandbox-token"

        with self.assertRaisesRegex(RuntimeError, "must be 'sandbox'"):
            tinvest_client.TInvestClient()

    def test_missing_sandbox_token_raises(self) -> None:
        self._clear_relevant_env()
        os.environ["TINVEST_ENV"] = "sandbox"

        with self.assertRaisesRegex(RuntimeError, "Missing required env var TINVEST_SANDBOX_TOKEN"):
            tinvest_client.TInvestClient()

    def test_legacy_token_env_is_rejected(self) -> None:
        self._clear_relevant_env()
        os.environ["TINVEST_ENV"] = "sandbox"
        os.environ["INVEST_TOKEN"] = "legacy-token"

        with self.assertRaisesRegex(RuntimeError, "Use TINVEST_SANDBOX_TOKEN only"):
            tinvest_client.TInvestClient()

    def test_target_is_forced_to_sandbox(self) -> None:
        self._clear_relevant_env()
        os.environ["TINVEST_ENV"] = "sandbox"
        os.environ["TINVEST_SANDBOX_TOKEN"] = "sandbox-token"

        class DummyRetryingClient:
            last_target = None

            def __init__(self, token, settings, **kwargs):
                DummyRetryingClient.last_target = kwargs.get("target")
                self._target = kwargs.get("target")

            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        with mock.patch.object(tinvest_client, "RetryingClient", DummyRetryingClient):
            client = tinvest_client.TInvestClient()

        self.assertEqual(client._target, tinvest_client.INVEST_GRPC_API_SANDBOX)
        self.assertEqual(DummyRetryingClient.last_target, tinvest_client.INVEST_GRPC_API_SANDBOX)


if __name__ == "__main__":
    unittest.main()
