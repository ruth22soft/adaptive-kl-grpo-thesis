import re
from difflib import SequenceMatcher


AMHARIC_PATTERN = re.compile(r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]')


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\\n", " ")
    text = re.sub(r"<\|.*?\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _tokenize(text: str):
    text = _normalize_text(text)
    if not text:
        return []
    return re.findall(r"[\w\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]+", text)


def _char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    if not pred_set or not ref_set:
        return 0.0

    inter = len(pred_set & ref_set)
    union = len(pred_set | ref_set)
    jaccard = inter / union if union else 0.0

    common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in ref_set)
    total = max(len(pred_tokens), len(ref_tokens))
    coverage = common / total if total else 0.0

    return max(jaccard, coverage)


def compute_score(solution_str, ground_truth, method='text'):
    """Reward for Amharic text generation.

    This is intentionally different from the math-answer reward. It evaluates whether
    the generated Amharic text is relevant to the target/reference text rather than
    trying to parse a boxed math answer.
    """
    generated = _normalize_text(solution_str)
    target = _normalize_text(ground_truth)

    if not generated:
        return {'score': -1.0, 'correctness': 0.0}

    if target and target in generated:
        text_score = 1.0
    else:
        text_score = 0.5 * _char_similarity(generated, target) + 0.5 * _token_overlap_score(generated, target)

    amharic_ratio = len(AMHARIC_PATTERN.findall(generated)) / max(len(generated.replace(' ', '')), 1)
    if amharic_ratio < 0.15:
        text_score *= 0.5

    # Soft reward for general Amharic text quality.
    score = max(-1.0, min(1.0, text_score))
    correctness = 1.0 if score >= 0.7 else 0.0
    return {'score': float(score), 'correctness': float(correctness)}
