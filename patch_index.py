path = "D:/Projects/Research2Text-main/src/index_documents.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

target = """    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )"""

replacement = """    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    except ValueError as e:
        if "Embedding function conflict" in str(e):
            print("Embedding function conflict detected. Recreating the collection...")
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            raise"""

if target in code:
    code = code.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched index_documents.py")
else:
    print("Failed to find target in index_documents.py")
