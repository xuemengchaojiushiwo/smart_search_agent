import os
import sys

# 确保本地源代码包可被导入
sys.path.insert(0, os.path.abspath('.'))
try:
    from mypymupdf4llm.helpers.pymupdf_rag import to_markdown
except Exception:
    from PYMUPDF4LLM.helpers.pymupdf_rag import to_markdown


def main():
    pdf_path = r"python_service/file/安联美元.pdf"
    out_dir = "out_pdfllm_allianz"
    os.makedirs(out_dir, exist_ok=True)

    md_text = to_markdown(pdf_path, emit_positions=True)
    out_path = os.path.join(out_dir, "pdfllm_document_with_pos.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(out_path)


if __name__ == "__main__":
    main()


