import re
from collections import Counter

def clean_text(text: str) -> str:
    """Clean text extracted from PDF/DOCX/TXT by removing boilerplate,
    headers, footers, page numbers, repeated lines, and formatting noise.
    """

    # ------------------------------
    # 1. Normalize Unicode/invisible chars
    # ------------------------------
    text = text.replace("\xa0", " ")      # non-breaking space
    text = text.replace("\u200b", "")     # zero-width space
    text = text.replace("\ufeff", "")     # BOM

    # ------------------------------
    # 2. Split into lines
    # ------------------------------
    lines = [line.strip() for line in text.split("\n")]

    # ------------------------------
    # 3. Remove empty or noisy lines
    # ------------------------------
    def is_noise(line):
        return (
            len(line) <= 1 or
            re.fullmatch(r"[_\-=]{3,}", line) is not None or
            re.fullmatch(r"[|]{3,}", line) is not None
        )

    lines = [line for line in lines if not is_noise(line)]

    # ------------------------------
    # 4. Remove obvious page numbers
    # ------------------------------
    page_num_patterns = [
        r"^page\s*\d+\s*of\s*\d+$",
        r"^page\s*\d+$",
        r"^\d+\s*/\s*\d+$",
        r"^-\s*\d+\s*-$",
        r"^\d+$",
    ]

    def is_page_number(line):
        return any(re.match(p, line, flags=re.IGNORECASE) for p in page_num_patterns)

    lines = [line for line in lines if not is_page_number(line)]

    # ------------------------------
    # 5. Remove watermark/footer patterns
    # ------------------------------
    footer_patterns = [
        r"confidential",
        r"all rights reserved",
        r"internal use only",
        r"do not distribute",
        r"printed on.*",
        r"copyright.*",
        r"company name.*",
    ]

    def is_footer(line):
        return any(re.search(p, line, flags=re.IGNORECASE) for p in footer_patterns)

    lines = [line for line in lines if not is_footer(line)]

    # ------------------------------
    # 6. Detect repeated boilerplate (headers/footers)
    #     Lines appearing in > 20% of pages are removed.
    # ------------------------------
    counts = Counter(lines)
    threshold = max(3, int(len(lines) * 0.20))

    lines = [line for line in lines if counts[line] < threshold]

    # ------------------------------
    # 7. Final whitespace normalization
    # ------------------------------
    text = " ".join(lines)          # collapse lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()
