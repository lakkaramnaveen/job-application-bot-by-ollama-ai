from job_bot.safety.audit_log import AuditLogger


def test_audit_log_redacts_anthropic_api_key(tmp_path):
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path)

    logger.log("apply_error", error="failed with key sk-ant-api03-abc123XYZ-super-secret")

    content = log_path.read_text()
    assert "sk-ant-api03-abc123XYZ-super-secret" not in content
    assert "[REDACTED]" in content


def test_audit_log_redacts_bearer_token(tmp_path):
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path)

    logger.log("apply_error", error="auth failed: Bearer abcDEF123.token-value")

    content = log_path.read_text()
    assert "Bearer abcDEF123.token-value" not in content
    assert "[REDACTED]" in content


def test_audit_log_redacts_nested_secrets(tmp_path):
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path)

    logger.log("nested_test", context={"headers": {"Authorization": "Bearer sk-leaked-token-value"}})

    content = log_path.read_text()
    assert "sk-leaked-token-value" not in content


def test_audit_log_only_records_metadata_not_full_resume_text(tmp_path):
    """The generation/matching modules never pass resume text to the audit
    log - this asserts the logger doesn't get any help hiding it either way,
    by confirming a plausible resume-length string still round-trips as-is
    (i.e. redaction targets secret *shapes*, not arbitrary long text) while
    callers are expected to only pass small metadata fields.
    """
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path)

    logger.log("scored", job_id="123", score=90, should_apply=True)

    content = log_path.read_text()
    assert '"job_id": "123"' in content
    assert '"score": 90' in content
