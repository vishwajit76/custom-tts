import numpy as np

from app.services.text_normalizer import chunk, cut_at_phrase, normalize, num_to_words


def test_numbers():
    assert num_to_words(0) == "शून्य"
    assert num_to_words(45) == "पैंतालीस"
    assert num_to_words(105) == "एक सौ पाँच"
    assert num_to_words(12500) == "बारह हज़ार पाँच सौ"
    assert num_to_words(2_50_00_000) == "दो करोड़ पचास लाख"


def test_normalize():
    assert normalize("₹12,500.50") == "बारह हज़ार पाँच सौ रुपये पचास पैसे"
    assert normalize("500 रुपये") == "पाँच सौ रुपये"
    assert normalize("05/10/2026") == "पाँच अक्टूबर दो हज़ार छब्बीस"
    assert normalize("18% ब्याज") == "अठारह प्रतिशत ब्याज"
    assert normalize("नंबर 9876543210") == "नंबर नौ आठ सात छह पाँच, चार तीन दो एक शून्य"
    # OTP/PIN codes are read digit by digit (Phase 6); the old quantity reading "एक सौ तेईस" was wrong for a code
    assert normalize("आपका OTP १२३ है") == "आपका ओ टी पी एक दो तीन है"
    assert normalize("Please call करें") == "Please call करें"


def test_times():
    assert normalize("10:30") == "साढ़े दस बजे"
    assert normalize("1:30") == "डेढ़ बजे"
    assert normalize("2:30") == "ढाई बजे"
    assert normalize("9:15") == "सवा नौ बजे"
    assert normalize("9:45") == "पौने दस बजे"
    assert normalize("12:45") == "पौने एक बजे"
    assert normalize("9:00") == "नौ बजे"
    assert normalize("9:20") == "नौ बजकर बीस मिनट"
    assert normalize("9:05") == "नौ बजकर पाँच मिनट"
    assert normalize("10:30 बजे") == "साढ़े दस बजे"
    assert normalize("10:30pm") == "रात साढ़े दस बजे"  # explicit am/pm keeps its day period
    assert normalize("10:30 PM") == "रात साढ़े दस बजे"


def test_phones():
    expected = "प्लस नौ एक, नौ आठ सात छह पाँच, चार तीन दो एक शून्य"  # groups: a comma is a short pause
    assert normalize("+91 98765 43210") == expected
    assert normalize("+91-9876543210") == expected
    assert normalize("098765-43210") == "शून्य नौ आठ सात छह पाँच, चार तीन दो एक शून्य"
    assert normalize("1800 123 4567") == "एक आठ शून्य शून्य, एक दो तीन, चार पाँच छह सात"
    assert normalize("मोबाइल +91 98765 43210 पर कॉल करें") == (
        "मोबाइल प्लस नौ एक, नौ आठ सात छह पाँच, चार तीन दो एक शून्य पर कॉल करें"
    )
    # must not be treated as a phone number
    assert normalize("12,500") == "बारह हज़ार पाँच सौ"


def test_ordinals():
    assert normalize("1st") == "पहला"
    assert normalize("2nd") == "दूसरा"
    assert normalize("3rd") == "तीसरा"
    assert normalize("4th") == "चौथा"
    assert normalize("5th") == "पाँचवाँ"
    assert normalize("6th") == "छठा"
    assert normalize("7th") == "सातवाँ"
    assert normalize("10th") == "दसवाँ"
    assert normalize("21st") == "इक्कीसवाँ"
    assert normalize("5TH") == "पाँचवाँ"


def test_currencies():
    assert normalize("$50") == "पचास डॉलर"
    assert normalize("50 USD") == "पचास डॉलर"
    assert normalize("Rs 500") == "पाँच सौ रुपये"
    assert normalize("Rs. 500") == "पाँच सौ रुपये"
    assert normalize("INR 500") == "पाँच सौ रुपये"
    assert normalize("500 INR") == "पाँच सौ रुपये"
    assert normalize("5 lakh") == "पाँच लाख"
    assert normalize("5 lac") == "पाँच लाख"
    assert normalize("2 crore") == "दो करोड़"
    assert normalize("2.5 crore") == "ढाई करोड़"
    assert normalize("₹5 lakh") == "पाँच लाख रुपये"
    assert normalize("₹12,500.50") == "बारह हज़ार पाँच सौ रुपये पचास पैसे"


def test_units():
    assert normalize("5 kg") == "पाँच किलो"
    assert normalize("10 km") == "दस किलोमीटर"
    assert normalize("5 gm") == "पाँच ग्राम"
    assert normalize("5 g") == "पाँच ग्राम"
    assert normalize("5 ml") == "पाँच मिलीलीटर"
    assert normalize("5 l") == "पाँच लीटर"
    assert normalize("5 ltr") == "पाँच लीटर"
    assert normalize("5 GB") == "पाँच जीबी"
    assert normalize("5 MB") == "पाँच एमबी"
    assert normalize("5 min") == "पाँच मिनट"
    assert normalize("5 mins") == "पाँच मिनट"
    assert normalize("5 hr") == "पाँच घंटे"
    assert normalize("5 hrs") == "पाँच घंटे"
    assert normalize("5 sec") == "पाँच सेकंड"


def test_ranges():
    assert normalize("10-15") == "दस से पंद्रह"
    assert normalize("10–15") == "दस से पंद्रह"


def test_symbols():
    assert normalize("A & B") == "A और B"
    assert normalize("user@example.com") == "user एट example डॉट कॉम"
    assert normalize("कल+आज") == "कल प्लस आज"
    assert normalize("x = y") == "x बराबर y"
    assert normalize("हाँ/नहीं") == "हाँ या नहीं"


def test_abbreviations():
    assert normalize("Ltd.") == "लिमिटेड"
    assert normalize("Pvt. Ltd.") == "प्राइवेट लिमिटेड"
    assert normalize("No. 5") == "नंबर पाँच"
    assert normalize("Sr. Dev") == "सीनियर Dev"
    assert normalize("Jr. Dev") == "जूनियर Dev"
    assert normalize("Smt. Dev") == "श्रीमती Dev"
    assert normalize("Prof. Dev") == "प्रोफ़ेसर Dev"


def test_edge_cases():
    assert normalize("05/10/2026") == "पाँच अक्टूबर दो हज़ार छब्बीस"
    assert normalize("₹12,500.50") == "बारह हज़ार पाँच सौ रुपये पचास पैसे"
    assert normalize("10:30 बजे") == "साढ़े दस बजे"
    assert normalize("12,500") == "बारह हज़ार पाँच सौ"


def test_chunk():
    text = "पहला वाक्य। दूसरा वाक्य है? " + "शब्द " * 100
    parts = chunk(text, 50)
    assert all(len(p) <= 50 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_long_clauses_are_cut_at_phrase_breaks_not_mid_phrase():
    # "commercial | project" and "shops | available हैं" were heard as hiccups: each chunk is its own utterance
    s = "अहमदाबाद में हमारे पास Sindhu Bhavan Road par एक excellent commercial project है जहाँ office spaces और retail shops available हैं."
    assert cut_at_phrase(s, 71) == ["अहमदाबाद में हमारे पास Sindhu Bhavan Road par",
                                    "एक excellent commercial project है जहाँ office spaces और retail shops available हैं."]
    assert chunk(s, 120)[-1] == "और retail shops available हैं."  # before the conjunction, not before "available"
    assert cut_at_phrase("मैं श्रेया बोल रही हूँ और आपके loan के बारे में", 40)[0] == "मैं श्रेया बोल रही हूँ"
    assert cut_at_phrase("छोटा", 40) == ["छोटा"] and cut_at_phrase("x" * 50, 40) == ["x" * 50]
