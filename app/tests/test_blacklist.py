import json

from job_bot.safety.blacklist import CompanyBlacklist


def test_missing_file_means_nothing_blocked(tmp_path):
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    assert blacklist.is_blocked("Acme Corp") is False


def test_loads_existing_blacklist_case_insensitively(tmp_path):
    path = tmp_path / "blacklist.json"
    path.write_text(json.dumps(["Old Employer Inc"]))

    blacklist = CompanyBlacklist(path)

    assert blacklist.is_blocked("old employer inc") is True
    assert blacklist.is_blocked("  OLD EMPLOYER INC  ") is True
    assert blacklist.is_blocked("Unrelated Co") is False


def test_malformed_json_file_means_nothing_blocked(tmp_path):
    """A hand-edited or corrupted blacklist.json must not crash startup -
    degrade to an empty blacklist rather than raising.
    """
    path = tmp_path / "blacklist.json"
    path.write_text("{not valid json", encoding="utf-8")

    blacklist = CompanyBlacklist(path)

    assert blacklist.is_blocked("Anything") is False


def test_non_utf8_file_means_nothing_blocked(tmp_path):
    """Same graceful-degrade reasoning as test_malformed_json_file_means_
    nothing_blocked above - a blacklist.json saved with a non-UTF-8
    encoding previously crashed CompanyBlacklist.__init__() (and every
    command that constructs one) with a raw UnicodeDecodeError instead of
    degrading to an empty blacklist the same way invalid JSON already does.
    """
    path = tmp_path / "blacklist.json"
    path.write_bytes("Acme ’s Corp".encode("cp1252"))

    blacklist = CompanyBlacklist(path)

    assert blacklist.is_blocked("Anything") is False


def test_non_list_json_file_means_nothing_blocked(tmp_path):
    """Valid JSON but the wrong shape (e.g. a dict, from an old or
    hand-edited format) is treated the same as no blacklist at all, not a
    crash.
    """
    path = tmp_path / "blacklist.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    blacklist = CompanyBlacklist(path)

    assert blacklist.is_blocked("Anything") is False


def test_add_persists_to_disk(tmp_path):
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)

    blacklist.add("Bad Company")

    reloaded = CompanyBlacklist(path)
    assert reloaded.is_blocked("bad company") is True


def test_remove_returns_true_and_unblocks(tmp_path):
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)
    blacklist.add("Bad Company")

    removed = blacklist.remove("bad company")

    assert removed is True
    assert blacklist.is_blocked("Bad Company") is False
    reloaded = CompanyBlacklist(path)
    assert reloaded.is_blocked("Bad Company") is False


def test_remove_returns_false_when_not_present(tmp_path):
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    assert blacklist.remove("Never Added Inc") is False


def test_list_companies_returns_sorted_display_names(tmp_path):
    """Real bug this guards against: list_companies() (and `job-bot
    blacklist list`) used to return the normalized/casefolded form used
    internally for matching - a company added as "Acme Corp" showed up as
    "acme corp" forever. The name as the user actually typed it must be
    preserved for display, while matching (is_blocked/remove) still goes
    through the normalized form so casing/whitespace differences don't
    matter for those.
    """
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    blacklist.add("Zebra Corp")
    blacklist.add("Acme Corp")

    assert blacklist.list_companies() == ["Acme Corp", "Zebra Corp"]


def test_list_companies_persists_display_casing_across_reload(tmp_path):
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)
    blacklist.add("Acme Corp")

    reloaded = CompanyBlacklist(path)

    assert reloaded.list_companies() == ["Acme Corp"]


def test_add_with_reason_persists_and_reloads(tmp_path):
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)

    blacklist.add("Acme Corp", reason="no H1B sponsorship")

    reloaded = CompanyBlacklist(path)
    assert reloaded.list_entries() == [{"name": "Acme Corp", "reason": "no H1B sponsorship"}]


def test_add_without_reason_defaults_to_empty_string(tmp_path):
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")

    blacklist.add("Acme Corp")

    assert blacklist.list_entries() == [{"name": "Acme Corp", "reason": ""}]


def test_list_companies_unaffected_by_reasons(tmp_path):
    """list_companies() stays name-only regardless of whether entries have
    a reason - existing callers (is_blocked-adjacent code, the dashboard's
    blacklist-list/remove flow) never needed to change for this feature.
    """
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    blacklist.add("Acme Corp", reason="no H1B sponsorship")
    blacklist.add("Beta Inc")

    assert blacklist.list_companies() == ["Acme Corp", "Beta Inc"]


def test_re_adding_a_company_updates_its_reason(tmp_path):
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    blacklist.add("Acme Corp", reason="typo reason")

    blacklist.add("Acme Corp", reason="correct reason")

    assert blacklist.list_entries() == [{"name": "Acme Corp", "reason": "correct reason"}]


def test_an_entry_with_no_reason_is_saved_as_a_plain_string(tmp_path):
    """The on-disk format for an entry with no reason must stay exactly
    what it was before this feature (a bare string, not {"name": ...,
    "reason": ""}) - so a blacklist where nobody has used --reason yet
    round-trips through _save() with the same shape it always had, and
    stays readable by any external tool that only ever expected a plain
    list of strings.
    """
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)

    blacklist.add("Acme Corp")

    assert json.loads(path.read_text(encoding="utf-8")) == ["Acme Corp"]


def test_an_entry_with_a_reason_is_saved_as_an_object(tmp_path):
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)

    blacklist.add("Acme Corp", reason="no H1B sponsorship")

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"name": "Acme Corp", "reason": "no H1B sponsorship"}
    ]


def test_loads_the_pre_reason_plain_string_format(tmp_path):
    """Backward compatibility: a blacklist.json written before this feature
    existed (or hand-edited/imported as plain strings) must still load
    correctly, with an empty reason for every entry.
    """
    path = tmp_path / "blacklist.json"
    path.write_text(json.dumps(["Old Employer Inc"]), encoding="utf-8")

    blacklist = CompanyBlacklist(path)

    assert blacklist.list_entries() == [{"name": "Old Employer Inc", "reason": ""}]


def test_loads_a_mix_of_plain_strings_and_reason_objects(tmp_path):
    path = tmp_path / "blacklist.json"
    path.write_text(
        json.dumps(["Beta Inc", {"name": "Acme Corp", "reason": "no H1B sponsorship"}]),
        encoding="utf-8",
    )

    blacklist = CompanyBlacklist(path)

    assert blacklist.list_entries() == [
        {"name": "Acme Corp", "reason": "no H1B sponsorship"},
        {"name": "Beta Inc", "reason": ""},
    ]


def test_skips_a_list_item_that_is_neither_a_string_nor_an_object(tmp_path):
    """A hand-edited file with a stray non-string, non-object item (a
    number, null, a nested list, ...) must not crash loading - skip just
    that one malformed entry, the same tolerance a plain non-string item
    already had before {"name", "reason"} objects were a valid shape too.
    """
    path = tmp_path / "blacklist.json"
    path.write_text(json.dumps([123, None, ["nested"], "Acme Corp"]), encoding="utf-8")

    blacklist = CompanyBlacklist(path)

    assert blacklist.list_entries() == [{"name": "Acme Corp", "reason": ""}]


def test_skips_an_entry_with_a_blank_or_missing_name(tmp_path):
    """A whitespace-only string entry, and an object entry with no "name"
    key (or a blank one) - both must be skipped rather than adding a
    company with an empty display name.
    """
    path = tmp_path / "blacklist.json"
    path.write_text(
        json.dumps(["   ", {"reason": "no name given"}, {"name": "  ", "reason": "blank"}, "Acme Corp"]),
        encoding="utf-8",
    )

    blacklist = CompanyBlacklist(path)

    assert blacklist.list_entries() == [{"name": "Acme Corp", "reason": ""}]


def test_is_blocked_collapses_internal_whitespace_like_gmail_sync_does(tmp_path):
    """blacklist.py and gmail_sync.py both normalize company names to decide
    whether two strings mean the same company - they must agree, or a
    company blocked here could still slip past the other subsystem's
    matching on the exact same pair of strings. See job_bot/text_utils.py.
    """
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    blacklist.add("Foo   Bar Inc")

    assert blacklist.is_blocked("Foo Bar Inc") is True
    assert blacklist.is_blocked("foo\tbar inc") is True
