import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch

from smartenergy.operations import (
    auto_train_loop as canonical_loop,
    detect_anomalies as canonical_detection,
    reset_db as canonical_reset,
    seed_db as canonical_seed,
    simulate_data as canonical_batch,
    stream_realtime as canonical_stream,
    train_model as canonical_training,
    wait_for_db as canonical_wait,
)


class OperationEntrypointTests(unittest.TestCase):
    def test_seed_entrypoint_creates_existing_metadata_only_when_invoked(self):
        engine = object()
        with (
            patch.object(canonical_seed, "get_engine", return_value=engine),
            patch.object(canonical_seed.Base.metadata, "create_all") as create_all,
            redirect_stdout(StringIO()),
        ):
            canonical_seed.main()
        create_all.assert_called_once_with(bind=engine)

    def test_wait_entrypoint_uses_explicit_environment_and_disposes_its_engine(self):
        settings = MagicMock(database_url="postgresql://fixture/db")
        wait_settings = MagicMock(max_attempts=1, sleep_seconds=0)
        engine = MagicMock()
        with (
            patch.object(canonical_wait, "load_settings", return_value=settings) as load_settings,
            patch.object(canonical_wait, "load_wait_settings", return_value=wait_settings),
            patch.object(canonical_wait, "create_engine", return_value=engine) as create_engine,
            redirect_stdout(StringIO()),
        ):
            canonical_wait.main()

        load_settings.assert_called_once_with(load_dotenv_file=False)
        create_engine.assert_called_once_with(settings.database_url, pool_pre_ping=True)
        engine.connect.assert_called_once()
        engine.dispose.assert_called_once()

    def test_wait_entrypoint_checks_database_url_before_retry_settings(self):
        with (
            patch.object(canonical_wait, "load_settings", return_value=MagicMock(database_url=None)),
            patch.object(
                canonical_wait,
                "load_wait_settings",
                side_effect=AssertionError("Retry settings are irrelevant without a URL"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL no esta configurada"):
                canonical_wait.main()

    def test_stream_entrypoint_preserves_cli_batch_size_and_closes_session_on_interrupt(self):
        session = MagicMock()
        session.query.return_value.count.return_value = 0
        with (
            patch.object(sys, "argv", ["stream_realtime", "--batch-size", "2", "--interval", "0.5"]),
            patch.object(canonical_stream, "get_engine", return_value=object()),
            patch.object(canonical_stream, "get_session_factory", return_value=lambda: session),
            patch.object(canonical_stream.Base.metadata, "create_all"),
            patch.object(canonical_stream.time, "sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(StringIO()),
        ):
            canonical_stream.main()

        self.assertEqual(len(session.bulk_save_objects.call_args.args[0]), 2)
        session.commit.assert_called_once()
        session.close.assert_called_once()

    def test_continuous_ml_loop_calls_each_canonical_job_once_per_cycle(self):
        calls = []
        with (
            patch.object(sys, "argv", ["auto_train_loop", "--interval", "5"]),
            patch.object(canonical_loop, "run_cmd", side_effect=lambda cmd: calls.append(cmd) or True),
            patch.object(canonical_loop.time, "sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(StringIO()),
        ):
            canonical_loop.main()

        self.assertEqual(
            calls,
            [
                [sys.executable, "-m", "smartenergy.operations.train_model"],
                [sys.executable, "-m", "smartenergy.operations.detect_anomalies"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
