"""E2E R-F regression: setup_logging redirection via the ``force`` flag.

The D5 cleanup replaced the env-var sentinel with a module flag plus an
explicit ``force=`` parameter. Re-calling ``setup_logging`` without
``force`` must keep the original handlers; calling it with ``force=True``
must close the old file handler and route records to the new log file only.
"""

from __future__ import annotations

import logging

from beyond_backprop.utils.training_support import setup_logging


def test_force_redirects_records_to_new_log_file(tmp_path) -> None:
    log_a = tmp_path / "A.log"
    log_b = tmp_path / "B.log"
    logger = logging.getLogger("e2e-rf-probe")

    try:
        setup_logging(log_level="INFO", log_file=str(log_a))
        logger.info("to-A")
        assert "to-A" in log_a.read_text()

        # Without force the reconfiguration is a no-op: B is never created.
        setup_logging(log_level="INFO", log_file=str(log_b))
        logger.info("still-A")
        assert not log_b.exists()

        # With force the new file receives records; A stops receiving them.
        setup_logging(log_level="INFO", log_file=str(log_b), force=True)
        logger.info("to-B")
        b_text = log_b.read_text()
        assert "to-B" in b_text
        assert "to-A" not in b_text and "still-A" not in b_text
        assert "to-B" not in log_a.read_text()
    finally:
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()
