from job_bot.llm.base import GenerationStats


def test_generation_stats_summarizes_and_resets():
    stats = GenerationStats()
    stats.record(seconds=6.0, prompt_tokens=3000, output_tokens=300, load_seconds=2.0)
    stats.record(seconds=4.0, prompt_tokens=1000, output_tokens=100)

    line = stats.summary()

    assert "Model: 2 call(s), 10.0s total (5.0s avg)" in line
    assert "avg prompt 2,000 tokens" in line
    assert "400 tokens generated (40/s)" in line
    assert "2.0s loading the model" in line
    stats.reset()
    assert stats == GenerationStats()
