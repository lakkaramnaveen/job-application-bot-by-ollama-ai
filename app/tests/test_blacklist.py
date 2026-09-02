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


def test_list_companies_returns_sorted_normalized_names(tmp_path):
    blacklist = CompanyBlacklist(tmp_path / "blacklist.json")
    blacklist.add("Zebra Corp")
    blacklist.add("Acme Corp")

    assert blacklist.list_companies() == ["acme corp", "zebra corp"]
