import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from smartenergy import config
from smartenergy.config import Settings
from smartenergy.infrastructure import database


class ConfigurationAndLifecycleTests(unittest.TestCase):
    def tearDown(self):
        database.dispose_engine()

    def test_settings_load_dotenv_explicitly_and_resolve_paths_from_repository(self):
        with (
            patch("dotenv.load_dotenv") as load_dotenv,
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://example/db"}, clear=True),
        ):
            settings = config.load_settings()

        load_dotenv.assert_called_once_with(config.REPOSITORY_ROOT / ".env")
        self.assertEqual(settings.database_url, "postgresql://example/db")
        self.assertEqual(
            settings.model_path,
            config.REPOSITORY_ROOT / "models" / "anomaly_detector.pkl",
        )

    def test_settings_can_read_only_process_environment_without_loading_dotenv(self):
        with (
            patch("dotenv.load_dotenv") as load_dotenv,
            patch.dict(
                os.environ,
                {
                    "DATABASE_URL": "postgresql://example/monitor",
                    "DB_MAX_ATTEMPTS": "3",
                    "DB_SLEEP_SECONDS": "0.25",
                },
                clear=True,
            ),
        ):
            settings = config.load_settings(load_dotenv_file=False)
            wait_settings = config.load_wait_settings()

        load_dotenv.assert_not_called()
        self.assertEqual(settings.database_url, "postgresql://example/monitor")
        self.assertEqual(wait_settings.max_attempts, 3)
        self.assertEqual(wait_settings.sleep_seconds, 0.25)

    def test_wait_only_configuration_cannot_break_monitoring_configuration(self):
        with patch.dict(
            os.environ,
            {"DB_MAX_ATTEMPTS": "invalid", "DB_SLEEP_SECONDS": "invalid"},
            clear=True,
        ):
            settings = config.load_settings(load_dotenv_file=False)
            self.assertIsNone(settings.database_url)
            with self.assertRaises(ValueError):
                config.load_wait_settings()

    def test_engine_and_session_factory_are_created_once_and_disposed_explicitly(self):
        settings = Settings("sqlite://", Path("/tmp/model.pkl"))
        engine = MagicMock()
        with patch.object(database, "create_engine", return_value=engine) as create_engine:
            with ThreadPoolExecutor(max_workers=4) as pool:
                engines = list(pool.map(lambda _: database.get_engine(settings), range(8)))
            factories = [database.get_session_factory(settings) for _ in range(3)]
            self.assertTrue(all(value is engine for value in engines))
            self.assertTrue(all(value is factories[0] for value in factories))
            create_engine.assert_called_once_with("sqlite://", echo=False)
            with self.assertRaises(RuntimeError):
                database.get_engine(Settings("sqlite:///other.db", Path("/tmp/model.pkl")))

        database.dispose_engine()
        engine.dispose.assert_called_once()
        self.assertIsNone(database.get_engine(Settings(None, Path("/tmp/model.pkl"))))


if __name__ == "__main__":
    unittest.main()
