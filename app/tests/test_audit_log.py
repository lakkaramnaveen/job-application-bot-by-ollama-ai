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


def test_read_entries_normalizes_a_null_details_field(tmp_path):
    """Real bug this guards against: log() always writes "details" as a
    dict, but a hand-edited or externally-authored line could have
    "details": null (valid JSON, a valid top-level entry object) - both
    cmd_audit_log's text output and render_audit_log_html previously
    called .items() directly on whatever entry["details"] came back,
    crashing with a raw AttributeError on this exact shape.
    """
    path = tmp_path / "audit.log"
    path.write_text('{"timestamp": "t", "action": "manual_note", "details": null}\n', encoding="utf-8")
    logger = AuditLogger(path)

    entries = logger.read_entries()

    assert entries == [{"timestamp": "t", "action": "manual_note", "details": {}}]


def test_read_entries_normalizes_a_non_dict_details_field(tmp_path):
    """Same reasoning as the null-details case above, for a "details" value
    that's valid JSON but not an object at all (a string, here) - also not
    something .items() can be called on directly.
    """
    path = tmp_path / "audit.log"
    path.write_text('{"timestamp": "t", "action": "manual_note", "details": "oops"}\n', encoding="utf-8")
    logger = AuditLogger(path)

    entries = logger.read_entries()

    assert entries == [{"timestamp": "t", "action": "manual_note", "details": {}}]


def test_read_entries_returns_empty_for_a_non_utf8_file(tmp_path):
    """Same graceful-degrade reasoning the other JSON stores already use
    for a non-UTF-8 file - a corrupted audit.log must not crash `job-bot
    audit-log` (or anything else reading it) with a raw UnicodeDecodeError.
    """
    path = tmp_path / "audit.log"
    path.write_bytes("Isn’t sponsorship needed?".encode("cp1252"))
    logger = AuditLogger(path)

    assert logger.read_entries() == []


def test_read_entries_empty_when_the_path_is_unreadable(tmp_path):
    """A directory where the log should be (or permission denied) used to
    escape as a raw IsADirectoryError from `job-bot audit-log`, the
    dashboard's Audit Log view, and doctor - same empty fallback as a
    missing or non-UTF-8 file now.
    """
    path = tmp_path / "audit.log"
    path.mkdir()
    assert AuditLogger(path).read_entries() == []


def test_last_search_signed_out_at_is_the_latest_search_outcome_only(tmp_path):
    logger = AuditLogger(tmp_path / "audit.log")
    assert logger.last_search_signed_out_at() is None  # no searches yet

    logger.log("search_error", keywords="x", location="y", error="not signed in", signed_out=True)
    logger.log("scored", job_id="1")  # not a search outcome - ignored
    signed_out_at = logger.last_search_signed_out_at()
    assert signed_out_at is not None and signed_out_at.startswith("20")

    logger.log("search", keywords="x", location="y", results=3)  # after `job-bot login`
    assert logger.last_search_signed_out_at() is None
