
import bleach


def clean_text(text: str) -> str:
    return bleach.clean(text, tags=[], strip=True).strip()
