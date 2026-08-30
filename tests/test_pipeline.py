"""Tests for the subtitle pipeline. Pure logic: no GPU, no ffmpeg, no network."""

import pytest

import subtitle_core as core


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


# --- Turkish-aware casing ------------------------------------------------------------

def test_turkish_lower_maps_dotless_i():
    # Python's str.lower() maps "I" to "i"; Turkish maps it to "ı".
    assert core.turkish_lower("IIIİ") == "ıııi"


def test_turkish_upper_maps_dotted_i():
    assert core.turkish_upper("istanbul") == "İSTANBUL"


def test_turkish_upper_first():
    assert core.turkish_upper_first("istanbul") == "İstanbul"


def test_normalize_strips_punctuation():
    assert core.normalize_text("Eee...") == "eee"


def test_uppercase_filler_is_recognised():
    assert core.is_filler_only(core.normalize_text("III")) is True


# --- Duplicate word collapsing -------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("ben ben gidiyorum", "ben gidiyorum"),
        ("ben ben ben gidiyorum", "ben gidiyorum"),
        ("Ben ben geldim", "Ben geldim"),
    ],
)
def test_stutters_collapse(text, expected):
    assert core.collapse_duplicate_words(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "çok çok güzel",       # Turkish reduplication intensifies, it is not a stutter
        "yavaş yavaş yürüdü",
        "Hayır. Hayır.",       # a sentence boundary is not a stutter either
    ],
)
def test_legitimate_repetition_is_kept(text):
    assert core.collapse_duplicate_words(text) == text


# --- terms.json dictionary -----------------------------------------------------------

def test_replacements_do_not_cascade():
    # Sequential replaces would turn "a" into "c"; a single pass must not.
    replacer = core.TermReplacer({"a": "b", "b": "c"}, "tr")
    assert replacer.apply("a b") == "b c"


def test_multi_word_key():
    replacer = core.TermReplacer({"chat gpt": "ChatGPT"}, "en")
    assert replacer.apply("I used chat gpt today") == "I used ChatGPT today"


def test_word_boundary_is_respected():
    replacer = core.TermReplacer({"chat gpt": "ChatGPT"}, "en")
    assert replacer.apply("chatgpt") == "chatgpt"


def test_explicit_casing_is_used_verbatim():
    replacer = core.TermReplacer({"istanbul": "İstanbul"}, "tr")
    assert replacer.apply("istanbul güzel") == "İstanbul güzel"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ISTANBUL çok güzel", "İstanbul çok güzel"),  # dotless capital I
        ("İSTANBUL çok güzel", "İstanbul çok güzel"),  # dotted capital İ
        ("Istanbul çok güzel", "İstanbul çok güzel"),
    ],
)
def test_both_turkish_capital_i_forms_match(text, expected):
    replacer = core.TermReplacer({"istanbul": "İstanbul"}, "tr")
    assert replacer.apply(text) == expected


def test_lowercase_replacement_mirrors_case():
    replacer = core.TermReplacer({"yanlis": "doğru"}, "tr")
    assert replacer.apply("Yanlis kelime") == "Doğru kelime"
    assert replacer.apply("YANLIS") == "DOĞRU"


def test_key_with_punctuation_edges():
    replacer = core.TermReplacer({"c++": "C++"}, "en")
    assert replacer.apply("i like c++ a lot") == "i like C++ a lot"


def test_empty_dictionary_is_a_no_op():
    assert core.TermReplacer({}, "tr").apply("değişmemeli") == "değişmemeli"


def test_broken_terms_file_warns_but_does_not_raise(tmp_path):
    bad = tmp_path / "terms.json"
    bad.write_text("{not valid json", encoding="utf-8")
    terms, warning = core.load_terms(bad)
    assert terms == {}
    assert warning


def test_terms_file_must_be_an_object(tmp_path):
    bad = tmp_path / "terms.json"
    bad.write_text('["a", "b"]', encoding="utf-8")
    terms, warning = core.load_terms(bad)
    assert terms == {} and warning


def test_missing_terms_file_is_created(tmp_path):
    path = tmp_path / "terms.json"
    terms, warning = core.load_terms(path)
    assert terms == {} and warning is None
    assert path.exists()


def test_non_string_entries_are_ignored(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text('{"ok": "fine", "bad": 5}', encoding="utf-8")
    terms, warning = core.load_terms(path)
    assert terms == {"ok": "fine"}
    assert "bad" in warning


# --- Line wrapping and cue layout ----------------------------------------------------

def test_short_text_stays_on_one_line():
    text = "Bu kısa bir cümle."
    assert core.wrap_text(text) == [text]


def test_long_text_wraps_to_two_balanced_lines():
    text = "Bu cümle otuz iki karakterden uzun olduğu için bölünmeli"
    lines = core.wrap_text(text)
    assert len(lines) == 2
    assert all(len(line) <= core.MAX_LINE_CHARS for line in lines)
    assert " ".join(lines) == text


def test_single_over_long_word_is_never_truncated():
    assert core.wrap_text("A" * 50) == ["A" * 50]


def test_over_long_segment_becomes_several_cues():
    text = (
        "Bu oldukça uzun bir cümledir ve otuz iki karakterlik satır sınırıyla "
        "iki satıra sığmayacak kadar fazla yer kaplamaktadır elbette"
    )
    cues = core.layout_cues(0.0, 10.0, text)

    assert len(cues) > 1
    assert all(len(c["text"].split("\n")) <= core.MAX_LINES_PER_CUE for c in cues)
    assert all(
        len(line) <= core.MAX_LINE_CHARS
        for c in cues
        for line in c["text"].split("\n")
    )
    # No word may be lost or reordered by the split.
    assert " ".join(l for c in cues for l in c["text"].split("\n")) == text
    # The cues must tile the original span exactly.
    assert cues[0]["start"] == 0.0 and cues[-1]["end"] == 10.0
    assert all(
        cues[i]["end"] == pytest.approx(cues[i + 1]["start"])
        for i in range(len(cues) - 1)
    )


def test_cue_duration_is_split_by_character_count():
    text = "kısa " * 4 + "uzun " * 20
    cues = core.layout_cues(0.0, 10.0, text.strip())
    total_chars = sum(len(c["text"].replace("\n", " ")) for c in cues)
    for cue in cues:
        share = len(cue["text"].replace("\n", " ")) / total_chars
        assert (cue["end"] - cue["start"]) / 10.0 == pytest.approx(share, abs=0.06)


# --- Drop rules ----------------------------------------------------------------------

def test_full_pipeline_drop_rules():
    segments = [
        seg(0.0, 1.0, "Eee"),                          # filler only
        seg(1.0, 3.0, "Eee, tamam o zaman"),           # filler inside a sentence: keep
        seg(3.0, 4.0, "ııı..."),                       # filler only
        seg(4.0, 6.0, "ben ben gidiyorum"),            # stutter, collapsed
        seg(6.0, 8.0, "çok çok güzel"),                # reduplication, untouched
        seg(8.0, 13.0, "Teşekkür ederim"),             # long but almost no words
        seg(13.0, 15.0, "Bu bir tekrar cümlesidir"),   # hallucination loop
        seg(15.0, 17.0, "Bu bir tekrar cümlesidir"),
        seg(17.0, 19.0, "Bu bir tekrar cümlesidir"),
        seg(19.0, 21.0, "   "),                        # empty
        seg(21.0, 23.0, "Hayır. Hayır."),
        seg(23.0, 24.0, "Kısa ama hızlı"),             # few words but short: keep
    ]
    assert [c["text"] for c in core.postprocess_segments(segments)] == [
        "Eee, tamam o zaman",
        "ben gidiyorum",
        "çok çok güzel",
        "Hayır. Hayır.",
        "Kısa ama hızlı",
    ]


def test_repeat_run_shorter_than_the_threshold_is_kept():
    segments = [seg(0.0, 1.0, "Aynı cümle burada"), seg(1.0, 2.0, "Aynı cümle burada")]
    assert len(core.postprocess_segments(segments)) == 2


def test_terms_are_applied_before_the_drop_rules():
    segments = [seg(0.0, 2.0, "reyki çok faydalı")]
    replacer = core.TermReplacer({"reyki": "reiki"}, "tr")
    assert core.postprocess_segments(segments, replacer)[0]["text"] == "reiki çok faydalı"


# --- SRT rendering -------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "00:00:00,000"), (3661.5, "01:01:01,500"), (-5, "00:00:00,000")],
)
def test_timestamp_format(seconds, expected):
    assert core.format_timestamp(seconds) == expected


@pytest.mark.parametrize("seconds,expected", [(83, "01:23"), (3723, "1:02:03")])
def test_clock_format(seconds, expected):
    assert core.format_clock(seconds) == expected


def test_srt_is_renumbered_without_gaps():
    segments = [seg(0.0, 1.0, "Eee"), seg(1.0, 2.0, "Bir"), seg(2.0, 3.0, "İki")]
    srt = core.build_srt(core.postprocess_segments(segments))
    numbers = [int(l) for l in srt.split("\n") if l.isdigit()]
    assert numbers == list(range(1, len(numbers) + 1))
    assert srt.count(" --> ") == len(numbers)


def test_empty_input_yields_empty_srt():
    assert core.build_srt(core.postprocess_segments([])) == ""


def test_truncate_middle_keeps_both_ends():
    out = core.truncate_middle("a" * 40 + "MIDDLE" + "b" * 40, limit=20)
    assert len(out) == 20 and out.startswith("a") and out.endswith("b") and "..." in out
