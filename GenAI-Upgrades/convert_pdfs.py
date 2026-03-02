import pymupdf4llm
import glob
import os

def main():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf in pdf_files:
        print(f"Converting {pdf}...")
        try:
            md_text = pymupdf4llm.to_markdown(pdf)
            md_filename = os.path.splitext(pdf)[0] + ".md"
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"Successfully created {md_filename}")
        except Exception as e:
            print(f"Failed to convert {pdf}: {e}")

if __name__ == "__main__":
    main()
