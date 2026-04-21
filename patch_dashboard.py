path = "D:/Projects/Research2Text-main/src/app_streamlit.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

target = """                for i, chunk in enumerate(chunks[:5]):
                    with st.expander(f"Chunk {i+1}", expanded=False):
                        st.text(chunk.get("text", "")[:500])"""

replacement = """                for i, chunk in enumerate(chunks[:5]):
                    with st.expander(f"Chunk {i+1}", expanded=False):
                        if isinstance(chunk, str):
                            st.text(chunk[:500])
                        else:
                            st.text(chunk.get("text", "")[:500])"""

if target in code:
    code = code.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched app_streamlit.py")
else:
    print("Target not found in app_streamlit.py")
