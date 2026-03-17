path = "D:/Projects/Research2Text-main/src/query_rag.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

target = """    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )"""

replacement = """    try:
        from embeddings import get_embedder
        embedder = get_embedder(prefer_gemini=True)
        class HybridEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, texts: List[str]) -> List[List[float]]:
                return embedder.embed(texts)
            def __init__(self):
                pass
        embed_fn = HybridEmbeddingFunction()
    except Exception as e:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )"""

if target in code:
    code = code.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched query_rag.py")
else:
    print("Failed to find target in query_rag.py")
