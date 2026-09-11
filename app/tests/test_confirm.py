from job_bot.safety.confirm import SubmitConfirmer


def test_confirm_returns_true_when_not_required():
    confirmer = SubmitConfirmer(required=False, ask=lambda summary: False)

    assert confirmer.confirm("Apply to Acme?") is True


def test_confirm_delegates_to_ask_when_required():
    confirmer = SubmitConfirmer(required=True, ask=lambda summary: summary == "yes please")

    assert confirmer.confirm("yes please") is True
    assert confirmer.confirm("no thanks") is False


def test_default_ask_accepts_y_and_yes_case_insensitively(monkeypatch):
    confirmer = SubmitConfirmer(required=True)
    for reply in ("y", "Y", "yes", "YES", " Yes  "):
        monkeypatch.setattr("builtins.input", lambda _prompt, reply=reply: reply)
        assert confirmer.confirm("Apply?") is True


def test_default_ask_rejects_anything_else(monkeypatch):
    confirmer = SubmitConfirmer(required=True)
    for reply in ("n", "no", "", "sure", "maybe"):
        monkeypatch.setattr("builtins.input", lambda _prompt, reply=reply: reply)
        assert confirmer.confirm("Apply?") is False


def test_default_ask_treats_eof_as_no_instead_of_crashing(monkeypatch, capsys):
    """input() raises EOFError when there's no terminal to answer from (cron,
    a pipe, CI without --yes-i-understand-the-risk). That used to propagate
    straight out of cmd_run uncaught - not in EXPECTED_ERRORS - crashing the
    whole run with a raw traceback instead of the same "declined" outcome a
    literal "no" produces.
    """
    confirmer = SubmitConfirmer(required=True)

    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert confirmer.confirm("Apply to Acme?") is False
    assert "treating as 'no'" in capsys.readouterr().out
