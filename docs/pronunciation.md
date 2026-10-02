# Hindi pronunciation layer (V7 track T1)

Code: `app/services/pronunciation/`. Evidence: `bench/results/pronunciation_probe_v7.tsv` (regenerate with
`python bench/pronunciation_probe.py`; `--check` fails if the committed table is stale). Tests: `tests/test_pronunciation_rules.py`.

**Status: implemented, tested, OFF by default.** Nothing here has been heard by a native speaker, and the corpus in
`tests/data/pronunciation_corpus.tsv` pins today's text output (it keeps `WhatsApp`, `sorry`, `यह` as written), so turning the
layer on changes 39 of its 384 rows. Enable per deployment with `PRONUNCIATION_RULES=all` after listening.

## Architecture

```
raw text
  -> text_normalizer.normalize():  stash URLs/emails -> hinglish.convert (romanized Hindi/names -> Devanagari)
                                   -> numbers, currency, dates, units, abbreviations -> numbers as words
  -> pronunciation.apply(text, rules)        (last step, so it sees the final words)
        Devanagari token -> data/lexical.tsv  -> else phonological.word()   (grammar, schwa)
        Latin token      -> data/names.tsv, data/english.tsv                  (names, english)
        number / punctuation -> never touched
  -> chunking -> TTS (any engine)
```

| Part | File | What it does |
|---|---|---|
| Lexical rules | `data/lexical.tsv` | Word-level respellings of Hindi words. Columns `word, respelling, category, reason`. |
| Phonological rules | `phonological.py` | Three rules, each with measured evidence (below). Pure function of one NFC word, `lru_cache`d. |
| Brand / name lexicon | `data/names.tsv` | Latin-script company, product, person and city names espeak reads wrongly. The 400+ names in `lexicon_hi.tsv` stay there (hinglish.py reads them; they win on conflict). |
| English exceptions | `data/english.tsv` | Latin loanwords the Indian-English path gets wrong (+ the letter name एफ in `lexical.tsv`). |
| Code-switch detection | `detect.py` `tag()` | Per-token `hi`, `hi-Latn`, `en`, `num`, `punct`; a Latin word is `hi-Latn` exactly when `hinglish.convert` rewrites it. Diagnostic API: the hot path needs no tags because hinglish has already run, so every Latin token `apply` sees is English or a name. |

Rule groups (`settings.pronunciation_rules`, env `PRONUNCIATION_RULES`; comma list, `all`, or `off`; unknown names raise):
`grammar` (यह, वह), `schwa` (halant respellings + the H/KAR/FINAL rules), `names`, `english`. Default `off`.
`normalize(text, rules="all")` overrides the setting for one call.

### Phoneme-safe text: respelling, not phonemes
Rules emit Devanagari that espeak reads correctly, so they work for every engine. The installed Piper (piper-tts 1.8.0)
**does** support inline phonemes: `PiperVoice.phonemize` splits `[[ ... ]]` and feeds the inside as raw IPA
(`piper/voice.py`), and `indian_english.mark()` already uses it for English runs. Only that raw-phoneme path
(Piper-only, IPA must be in the voice's phoneme map) can express a sound a respelling cannot, e.g. कृपया as kri-pa-yaa. It is not used
here because `normalize()` is engine-agnostic (Kokoro and Supertonic would read the brackets); that fix belongs in
`PiperEngine.synth` (track T4) and is left open. Note the bare espeak call (`indian_english.espeak`) does NOT understand `[[ ]]`:
probing with it shows spelled-out garbage, which is a harness mistake, not a Piper bug.

The halant respellings (`रह्ता`, `चाह्ता`) are tricks for espeak-based engines (Piper, Kokoro). They are not correct orthography,
which is why the `schwa` group is separate and off by default; Supertonic reads graphemes and may handle them worse.

## What espeak-ng `hi` gets wrong (probe, 197 items)

Method: for each item, text -> `normalize` -> `indian_english.mark` -> Piper phonemize (exactly what `PiperEngine.synth` does),
once with rules off and once with `all`. "Wrong" means the broad form differs from the expected standard (Delhi/Hindustani
colloquial): stress, vowel length, dental vs retroflex, nasal vowel vs vowel + nasal, gemination and ʌ/ə, h/ɦ notation are
ignored, so a flagged item differs in syllables or segments. The expected forms are the author's reading, not native-reviewed.

| category | items | wrong before | wrong after |
|---|--:|--:|--:|
| grammar words (है ... इसके) | 47 | 4 | 0 |
| schwa (compounds, verbs, names of months/numbers incl.) | 32 | 22 | 0 |
| conjuncts / final clusters | 12 | 5 | 0 |
| explicit halant, anusvara, chandrabindu | 16 | 0 | 0 |
| nukta (NFC and precomposed) | 12 | 2 | 0 |
| ऋ | 4 | 1 | 1 |
| Perso-Arabic | 12 | 2 | 0 |
| Devanagari loanwords | 9 | 1 | 1 |
| Devanagari names | 10 | 6 | 0 |
| Latin names / brands | 20 | 5 | 0 |
| cities | 11 | 4 | 2 |
| abbreviations | 9 | 1 | 0 |
| English words | 3 | 3 | 0 |
| **total** | **197** | **56** | **4** |

Findings (all in the TSV, with IPA):
- **Correct already, no rule:** every grammar form except यह/वह (है/हैं/हूँ/हो, था..., करता..., रहा..., गया/गई/गए, चाहता* , सकता, जाएगा, करेगा, मिलेगा, देगा, ने/को/से/में/पर, ये/वे, मेरा..., आपका..., इसका...), nukta letters (and espeak reads the precomposed U+095B... and base + nukta identically), anusvara before stops (homorganic: संबंध səmbʌndʰ, गंगा ɡʌŋɡaː; before d/t it becomes a nasal vowel, e.g. हिंदी hĩdi, which is normal Hindi), chandrabindu, ऋ as ɾɪ.
- **यह / वह** are read letter by letter (jəh, ʋəh); colloquial is ये / वो. Lexical, group `grammar`.
- **Schwa kept where Hindi drops it:** ह before a syllable (रहता rahataa, पहचान, मेहनत, चाहता, मेहता, कोहली, नेहरू, शाहरुख़, पढ़ता), stem + -कर (देखकर, उठकर, बजकर, तेंदुलकर, अंबेडकर), final nasal clusters (प्रश्न prashna, कृष्ण, फ़िल्म, ब्रह्म), and a list of single words (मतलब, बचपन, एकदम, जनवरी, फ़रवरी, दशमलव, उनसठ... all words the number/date normalizer itself writes).
- **Schwa dropped where Hindi keeps it:** नमस्कार, पुरस्कार, राजस्थान, ज़िंदगी, इंतज़ार, पासवर्ड.
- **एफ** (the letter F in PDF, HDFC, FD...) is read ep (फ as aspirated p); एफ़ with nukta is ef.
- **Latin names the English path garbles:** Zerodha, Groww, WhatsApp, Aadhaar, Priyanka, Mysuru, Tiruchirappalli, Thiruvananthapuram, Ernakulam.
- **Not fixed, no respelling can express it** (still wrong after): कृपया (espeak kri-pya, Hindi kri-pa-yaa; needs the Piper `[[ ]]` path), बेंगलुरु/Bengaluru (espeak drops the schwa after ग), डिलीवरी (3 syllables, Hindi 4). Also not flagged but audible: कंप्यूटर has a long p (pː) before ्य, and अकाउंट reads aːʊ, not the diphthong.
- **Left alone on purpose:** a general schwa-deletion algorithm. A directional Ohala-style pass disagrees with both espeak and native usage on जनवरी, दशमलव, नमस्कार (a scratch reference implementation, not shipped, disagreed with espeak on 91 of the repo's 1288 Devanagari word types, and on the cases checked by hand the reference was wrong about as often as espeak), so only the three narrow rules above and a word list are used. `पहला/पहले` additionally have ɛ for the schwa before ह (not modelled; the probe compares broad forms).

## Rules in detail
- **H** (`schwa`): bare ह (or ढ़) that follows a vowelled syllable and precedes a syllable with a spoken vowel gets a virama:
  रहता -> रह्ता, पहचान -> पह्चान, मेहनत -> मेह्नत. Final ह (बाहर, शहर, दोपहर), ह with a vowel sign (बहुत, सहारा), and ह before a final bare consonant (बहन, रोहन) are untouched.
- **KAR** (`schwa`): `<vowelled syllable><bare consonant>कर` -> halant on the consonant: देखकर -> देख्कर. नौकर, शंकर, भास्कर, मुकर are untouched.
- **FINAL** (`schwa`): consonant + virama + final bare न/म/ण, excluding र्/न्/म् (espeak is right for कर्म, जन्म): प्रश्न -> प्रश्न्.
- **Nukta policy:** words are matched on their NFC form (which decomposes ज़ to ज + nukta), so a lexicon key matches either spelling; unmatched words are never rewritten.
- **Anusvara / chandrabindu / ऋ / conjuncts:** no rule; the probe shows espeak correct, and rules without an error to fix are not added.

## Adding an entry
1. Run the word through the real path (`python bench/pronunciation_probe.py` shows how; or `indian_english.espeak("hi", word)`),
   write the expected form, and confirm espeak is wrong. Add it as a row in `bench/pronunciation_probe_items.tsv`.
2. Add a row to the right TSV: `word, respelling, category, reason`. The reason states the espeak error (its IPA) and the correct form. A row without both fails import.
   - Hindi word -> `lexical.tsv`, category `grammar` (colloquial grammar words), `schwa` (halant/schwa fixes) or `english` (letter names).
   - Latin name or brand -> `names.tsv` (category person/city/company/product). English word -> `english.tsv`.
3. Check the respelling with espeak (it must read right), run `python bench/pronunciation_probe.py`, commit the regenerated table.
4. Add the word to a test in `tests/test_pronunciation_rules.py`.

When **not** to add: espeak already reads it correctly (grammar words are tested to pass through unchanged); the word is only
disputed between speakers (Hyderabad, पत्र with or without the final vowel); a pattern covers it (extend the rule, with probe words, not the list); it is a name already in `lexicon_hi.tsv`.

## Measured cost
`normalize()` on a 162-character Hindi + English sentence: 0.35 ms with rules off (indistinguishable from the V6 code, 0.35-0.43 ms,
run to run noise), +0.04 ms with every group on (0.39 ms), cold or warm word cache. Regexes and TSVs load once at import.

## Limits
- **No native-listener review.** Expected forms are the author's; corpus rows and probe rows are machine-drafted.
- **IPA correctness is not audio correctness.** A Piper voice trained on espeak labels may have learned to render a "wrong" label correctly (the IndicTTS speaker really said `rahtaa` while the label read `rahataa`). A respelling moves the input off the training distribution, so it can sound worse. That is why the `schwa` rules are off until someone listens, and why the probe is the only claim made.
- Probe items are isolated words; connected-speech stress and phrase effects are not measured.
- Dental/retroflex, ɛ-raising before ह, and h/ɦ differences are not compared.
- Enabling the layer changes `normalize()` output for 39 corpus rows (11 names/english, 2 grammar, 26 schwa); those rows would need a native reviewer's sign-off, not an edit to match the code.
- Shared files: `training/kaggle/make_bundle.py` now ships the package inside the Kaggle eval bundle (the bundle test needs it); `text_normalizer.py` imports the setting guardedly because the bundle has no `app.core`.
