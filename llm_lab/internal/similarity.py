from llm_lab.settings import settings


def jaccard_similarity(query: str, document: str) -> float:
    query = set(query.lower().split(" "))
    document = set(document.lower().split())
    intersection = query & document
    union = query | document
    return len(intersection) / len(union)


def return_response(query: str) -> str:
    similarities = [jaccard_similarity(query, c) for c in settings.CORPUS]
    return settings.CORPUS[similarities.index(max(similarities))]
