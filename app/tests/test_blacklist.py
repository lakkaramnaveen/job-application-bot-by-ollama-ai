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


def test_add_persists_to_disk(tmp_path):
    path = tmp_path / "blacklist.json"
    blacklist = CompanyBlacklist(path)

    blacklist.add("Bad Company")

    reloaded = CompanyBlacklist(path)
    assert reloaded.is_blocked("bad company") is True
