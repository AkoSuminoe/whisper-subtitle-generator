"""Shared subtitle pipeline, used by both the desktop app and the HTTP API."""

import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import types
import unicodedata
from contextlib import contextmanager
from pathlib import Path


def configure_utf8_console():
    """Force UTF-8 on stdout/stderr; the cp1252 console dies on Turkish text."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


APP_DIR = Path(__file__).resolve().parent
TERMS_PATH = APP_DIR / "terms.json"

# Hide the console window that ffprobe would otherwise flash on screen. The flag
# is Windows-only: passing it elsewhere raises ValueError.
CREATE_NO_WINDOW = 0x08000000
_SUBPROCESS_FLAGS = (
    {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}
)

# --- Post-processing tuning knobs -------------------------------------------------
MAX_LINE_CHARS = 32  # maximum characters on a single subtitle line
MAX_LINES_PER_CUE = 2  # longer text becomes additional cues rather than a taller one
LONG_SEGMENT_SEC = 3.0  # a segment longer than this with too few words is a hallucination
MIN_WORDS_FOR_LONG = 4
REPEAT_RUN_THRESHOLD = 3  # this many identical consecutive segments is a loop
KEEP_FIRST_OF_REPEAT_RUN = False  # set True to keep one instance of a detected loop

FILLER_WORDS = {
    # Turkish
    "eee", "ııı", "hıı", "ee", "ıı", "hmm", "ıhı",
    # English
    "uh", "um", "uhh", "umm", "mhm",
}

# Turkish reduplication is grammatical (it intensifies), not a stutter - never collapse.
REDUPLICATION_ALLOWLIST = {
    "çok", "yavaş", "ağır", "tek", "sık", "ara", "az", "uzun", "yeni",
    "iyi", "güzel", "bazı", "teker", "birer", "yer",
}

SENTENCE_END_CHARS = (".", "!", "?", "…")

MODEL_CHOICES = ["medium", "large-v3-turbo", "large-v3"]
DEFAULT_MODEL = "large-v3"
MODEL_SIZES = {"medium": "~1.5 GB", "large-v3-turbo": "~1.6 GB", "large-v3": "~2.9 GB"}

LANGUAGES = {"Turkish": "tr", "English": "en"}



# ======================================================================================
# Text helpers
# ======================================================================================

def turkish_lower(text):
    """Lowercase Turkish-correctly: plain str.lower() maps "I" to "i", not to "ı"."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def turkish_upper(text):
    """Uppercase Turkish-correctly: "i" becomes "İ" and "ı" becomes "I"."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def turkish_upper_first(text):
    """Capitalize the first letter, mapping "i" to "İ" the way Turkish requires."""
    if not text:
        return text
    first = "İ" if text[0] == "i" else text[0].upper()
    return first + text[1:]


def case_variants(word):
    """Every casing of a term. Turkish and Python disagree on i/I, so list both."""
    return {v for v in (
        word, word.lower(), word.upper(), turkish_lower(word), turkish_upper(word)
    ) if v}


def strip_punctuation(text):
    """Replace Unicode punctuation with spaces so tokens stay separated."""
    return "".join(
        " " if unicodedata.category(ch).startswith("P") else ch for ch in text
    )


def normalize_text(text):
    """Canonical form used by every comparison rule: no punctuation, Turkish-lowercased."""
    return " ".join(turkish_lower(strip_punctuation(text)).split())


def format_timestamp(seconds):
    """Format seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_clock(seconds):
    """Format seconds as MM:SS (or HH:MM:SS past an hour) for the status line."""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def truncate_middle(text, limit=58):
    """Shorten a long path for display, keeping both ends readable."""
    if len(text) <= limit:
        return text
    head = (limit - 3) // 2
    tail = limit - 3 - head
    return text[:head] + "..." + text[-tail:]


# ======================================================================================
# terms.json custom dictionary
# ======================================================================================

def load_terms(path=None):
    """Read the terms dictionary. Returns (mapping, warning); never raises."""
    terms_path = Path(path) if path else TERMS_PATH

    if not terms_path.exists():
        try:
            terms_path.write_text("{}\n", encoding="utf-8")
        except OSError as exc:
            return {}, f"Could not create terms.json: {exc}"
        return {}, None

    try:
        raw = json.loads(terms_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, f"terms.json could not be read, so replacements were skipped.\n\n{exc}"

    return validate_terms(raw)


def validate_terms(raw):
    """Coerce a parsed JSON object into a {str: str} mapping.

    Shared by the file loader and the per-request dictionary the API accepts, so
    both apply the same rules. Returns (mapping, warning); never raises.
    """
    if not isinstance(raw, dict):
        return {}, 'Terms must be a flat {"wrong": "correct"} object. Replacements were skipped.'

    terms, ignored = {}, []
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str) and key.strip():
            terms[key] = value
        else:
            ignored.append(str(key))

    warning = None
    if ignored:
        warning = "Ignored non-text entries: " + ", ".join(ignored[:5])
    return terms, warning


class TermReplacer:
    """Applies terms.json substitutions in one pass, so replacements never cascade."""

    def __init__(self, terms, lang_code="tr"):
        self._lang = lang_code
        self._lookup = {}
        variants = set()

        for key, value in terms.items():
            for variant in case_variants(key):
                variants.add(variant)
                # Index under both folding schemes - Python's default (which is what
                # re.IGNORECASE matches with) and the Turkish one - so that "ISTANBUL"
                # and "İSTANBUL" both resolve back to the "istanbul" entry.
                self._lookup.setdefault(variant.lower(), value)
                self._lookup.setdefault(turkish_lower(variant), value)

        self._pattern = None
        if variants:
            # Longest key first so "chat gpt" wins over "chat". Lookarounds rather than
            # \b so keys that begin or end with punctuation still behave.
            ordered = sorted(variants, key=len, reverse=True)
            alternation = "|".join(re.escape(variant) for variant in ordered)
            self._pattern = re.compile(rf"(?<!\w)({alternation})(?!\w)", re.IGNORECASE)

    def apply(self, text):
        if not self._pattern or not text:
            return text
        return self._pattern.sub(self._substitute, text)

    def _substitute(self, match):
        matched = match.group(0)
        replacement = self._lookup.get(matched.lower())
        if replacement is None:
            replacement = self._lookup.get(turkish_lower(matched))
        if replacement is None:
            return matched
        # An uppercase character in the replacement means the user chose that casing.
        if any(ch.isupper() for ch in replacement):
            return replacement
        return self._match_case(matched, replacement)

    def _match_case(self, matched, replacement):
        if len(matched) > 1 and matched.isupper():
            return turkish_upper(replacement) if self._lang == "tr" else replacement.upper()
        if matched[:1].isupper():
            if self._lang == "tr":
                return turkish_upper_first(replacement)
            return replacement[:1].upper() + replacement[1:]
        return replacement


# ======================================================================================
# Segment post-processing (pure functions - testable without Tk or Whisper)
# ======================================================================================

def collapse_duplicate_words(text):
    """Collapse stutters like "ben ben gidiyorum" into "ben gidiyorum"."""
    tokens = text.split()
    if len(tokens) < 2:
        return text

    kept = [tokens[0]]
    for token in tokens[1:]:
        previous = kept[-1]
        previous_norm = normalize_text(previous)
        current_norm = normalize_text(token)
        is_repeat = bool(previous_norm) and previous_norm == current_norm
        crosses_sentence = previous.endswith(SENTENCE_END_CHARS)
        if is_repeat and not crosses_sentence and previous_norm not in REDUPLICATION_ALLOWLIST:
            continue
        kept.append(token)
    return " ".join(kept)


def is_filler_only(normalized):
    """True when every token is a filler - never used to strip fillers mid-sentence."""
    if not normalized:
        return False
    return all(token in FILLER_WORDS for token in normalized.split())


def balanced_split_index(words):
    """Word index that divides `words` into the two most even halves by length."""
    best_index, best_delta = 1, None
    for index in range(1, len(words)):
        first = " ".join(words[:index])
        second = " ".join(words[index:])
        delta = abs(len(first) - len(second))
        if best_delta is None or delta < best_delta:
            best_index, best_delta = index, delta
    return best_index


def wrap_text(text, width=MAX_LINE_CHARS):
    """Wrap text into lines of at most `width` chars, preferring two balanced lines."""
    if len(text) <= width:
        return [text]

    words = text.split()
    if len(words) < 2:
        return [text]  # a single over-long word: never truncate it

    index = balanced_split_index(words)
    first = " ".join(words[:index])
    second = " ".join(words[index:])
    if len(first) <= width and len(second) <= width:
        return [first, second]

    # Needs more than two lines: greedy wrap, and the caller splits it into cues.
    return textwrap.wrap(text, width=width) or [text]


def wrap_line(text, width=MAX_LINE_CHARS):
    """Wrap over-long text at word boundaries, joined with newlines."""
    return "\n".join(wrap_text(text, width))


def layout_cues(start, end, text, width=MAX_LINE_CHARS, max_lines=MAX_LINES_PER_CUE):
    """Split a segment into cues of at most `max_lines` lines.

    Duration is divided in proportion to character count. Recurses because one
    split is not always enough at a narrow width.
    """
    lines = wrap_text(text, width)
    if len(lines) <= max_lines:
        return [{"start": start, "end": end, "text": "\n".join(lines)}]

    words = text.split()
    if len(words) < 2:
        return [{"start": start, "end": end, "text": "\n".join(lines)}]

    index = balanced_split_index(words)
    first = " ".join(words[:index])
    second = " ".join(words[index:])
    total_chars = len(first) + len(second)
    if not total_chars:
        return [{"start": start, "end": end, "text": "\n".join(lines)}]

    # Each recursion strictly reduces the word count, so this always terminates.
    midpoint = start + (end - start) * (len(first) / total_chars)
    return layout_cues(start, midpoint, first, width, max_lines) + layout_cues(
        midpoint, end, second, width, max_lines
    )


def postprocess_segments(segments, replacer=None):
    """Clean Whisper's raw segments: transforms, then drops, then wrapping last."""
    replacer = replacer or TermReplacer({})

    prepared = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        text = replacer.apply(text)
        text = collapse_duplicate_words(text)
        prepared.append(
            {
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
                "text": text,
                "norm": normalize_text(text),
            }
        )

    # Every drop decision is computed against the same sequence, then applied in one
    # pass, so no rule's outcome depends on another rule having already removed a item.
    drop = [False] * len(prepared)
    for index, item in enumerate(prepared):
        normalized = item["norm"]
        duration = item["end"] - item["start"]
        if not normalized:
            drop[index] = True
        elif is_filler_only(normalized):
            drop[index] = True
        elif duration > LONG_SEGMENT_SEC and len(normalized.split()) < MIN_WORDS_FOR_LONG:
            drop[index] = True

    # Hallucination loops: maximal runs of consecutive segments with identical text.
    start = 0
    while start < len(prepared):
        end = start + 1
        while end < len(prepared) and prepared[end]["norm"] == prepared[start]["norm"]:
            end += 1
        if prepared[start]["norm"] and (end - start) >= REPEAT_RUN_THRESHOLD:
            first_dropped = start + 1 if KEEP_FIRST_OF_REPEAT_RUN else start
            for index in range(first_dropped, end):
                drop[index] = True
        start = end

    kept = [item for item, dropped in zip(prepared, drop) if not dropped]

    # Wrapping happens last, and may turn one segment into several cues.
    cues = []
    for item in kept:
        cues.extend(layout_cues(item["start"], item["end"], item["text"]))
    return cues


def build_srt(segments):
    """Render cleaned segments as SRT text, renumbered from 1 with no gaps."""
    blocks = []
    for number, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        blocks.append(f"{number}\n{start} --> {end}\n{segment['text']}\n")
    return "\n".join(blocks)


# ======================================================================================
# Media + Whisper plumbing
# ======================================================================================

def probe_duration(path):
    """Return media duration in seconds via ffprobe, or None if it cannot be read."""
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            **_SUBPROCESS_FLAGS,
        )
        if completed.returncode != 0:
            return None
        duration = json.loads(completed.stdout)["format"]["duration"]
        value = float(duration)
        return value if value > 0 else None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def _make_tqdm_shim(callback):
    """Build a stand-in for tqdm.tqdm that forwards Whisper's frame progress."""

    class ProgressTqdm:
        def __init__(self, total=None, unit=None, disable=False, **kwargs):
            self.total = total or 0
            self.n = 0

        def update(self, n=1):
            self.n += n
            if self.total:
                callback(min(1.0, self.n / self.total))

        def close(self):
            pass

        def set_description(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    return ProgressTqdm


@contextmanager
def whisper_progress(callback):
    """Route Whisper's internal tqdm bar into `callback` for real progress."""
    try:
        # Not `import whisper.transcribe as m`: __init__.py rebinds that name to
        # the function, which has no tqdm to patch. import_module gives the module.
        transcribe_module = importlib.import_module("whisper.transcribe")
    except ImportError:
        yield
        return

    original = getattr(transcribe_module, "tqdm", None)
    if original is None:
        # Unknown Whisper version: transcribe normally, just without progress.
        yield
        return

    transcribe_module.tqdm = types.SimpleNamespace(tqdm=_make_tqdm_shim(callback))
    try:
        yield
    finally:
        transcribe_module.tqdm = original


def whisper_cache_dir():
    """Mirror Whisper's own download_root logic so we can detect a missing model."""
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

