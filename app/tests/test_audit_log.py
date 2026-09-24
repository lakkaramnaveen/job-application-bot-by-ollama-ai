from job_bot.safety.audit_log import AuditLogger


def test_read_entries_empty_when_no_file_exists_yet(tmp_path):
    logger = AuditLogger(tmp_path / "audit.log")

    assert logger.read_entries() == []


def test_read_entries_returns_most_recent_first(tmp_path):
    logger = AuditLogger(tmp_path / "audit.log")
    logger.log("search", keywords="backend engineer")
    logger.log("scored", job_id="1", score=90)
    logger.log("applied", job_id="1", company="Acme")

    entries = logger.read_entries()

    assert [e["action"] for e in entries] == ["applied", "scored", "search"]


def test_read_entries_filters_by_exact_action(tmp_path):
    logger = AuditLogger(tmp_path / "audit.log")
    logger.log("scored", job_id="1", score=90)
    logger.log("applied", job_id="1", company="Acme")
    logger.log("scored", job_id="2", score=40)

    entries = logger.read_entries(action="applied")

    assert len(entries) == 1
    assert entries[0]["details"]["job_id"] == "1"


def test_read_entries_search_matches_action_name(tmp_path):
    logger = AuditLogger(tmp_path / "audit.log")
    logger.log("skip_blacklisted", job_id="1", company="Acme")
    logger.log("applied", job_id="2", company="Beta")

    entries = logger.read_entries(search="blacklisted")

    assert len(entries) == 1
    assert entries[0]["action"] == "skip_blacklisted"


def test_read_entries_search_matches_a_detail_value_case_insensitively(tmp_path):
    """A detail value (not just the action name) must be searchable too -
    the most common real use is finding every entry mentioning a specific
    company or job_id, whichever action logged it.
    """
    logger = AuditLogger(tmp_path / "audit.log")
    logger.log("applied", job_id="1", company="Acme Corp")
    logger.log("applied", job_id="2", company="Beta Inc")

    entries = logger.read_entries(search="acme")

    assert len(entries) == 1
    assert entries[0]["details"]["company"] == "Acme Corp"


def test_read_entries_search_and_action_combine(tmp_path):
    logger = AuditLogger(tmp_path / "audit.log")
    logger.log("applied", job_id="1", company="Acme Corp")
    logger.log("scored", job_id="2", score=40, error="Acme Corp mismatch")
    logger.log("applied", job_id="3", company="Beta Inc")

    entries = logger.read_entries(search="acme", action="applied")

    assert len(entries) == 1
    assert entries[0]["details"]["job_id"] == "1"


def test_read_entries_skips_a_corrupted_line_but_keeps_the_valid_ones(tmp_path):
    """A single truncated/corrupted line (e.g. a killed process mid-write)
    must not hide every other, valid entry - this is an append-only log of
    independent entries, unlike the JSON *stores* elsewhere where one
    corrupted file degrades the whole thing to empty.
    """
    path = tmp_path / "audit.log"
    logger = AuditLogger(path)
    logger.log("applied", job_id="1")
    with path.open("a", encoding="utf-8") as f:
        f.write("not valid json {{{\n")
        f.write("\n")  # a blank line must also be skipped, not crash
        f.write("42\n")  # valid JSON, but not an object - also skipped
    logger.log("applied", job_id="2")

    entries = logger.read_entries()

    assert [e["details"]["job_id"] for e in entries] == ["2", "1"]


def test_read_entries_returns_empty_for_a_non_utf8_file(tmp_path):
    """Same graceful-degrade reasoning the other JSON stores already use
    for a non-UTF-8 file - a corrupted audit.log must not crash `job-bot
    audit-log` (or anything else reading it) with a raw UnicodeDecodeError.
    """
    path = tmp_path / "audit.log"
    path.write_bytes("Isn’t sponsorship needed?".encode("cp1252"))
    logger = AuditLogger(path)

    assert logger.read_entries() == []
