from job_bot.pipeline.search_terms import SearchRotation, split_search_terms


def test_a_comma_separated_list_becomes_trimmed_distinct_titles():
    assert split_search_terms(" java full stack,  MERN  stack ,, Java Full Stack,junior engineer ") == [
        "java full stack",
        "MERN stack",
        "junior engineer",
    ]
    assert split_search_terms("software engineer") == ["software engineer"]


def test_each_cycle_searches_the_next_title_in_turn():
    rotation = SearchRotation("a, b, c")
    assert [rotation.next_term() for _ in range(5)] == ["a", "b", "c", "a", "b"]
    assert rotation.position("b") == " (2 of 3)"
    assert SearchRotation("only one").position("only one") == ""
