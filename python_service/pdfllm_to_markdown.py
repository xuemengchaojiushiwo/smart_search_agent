#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 PyMuPDF4LLM 将 PDF 导出为 Markdown，并生成简要统计报告。
用法：
  python pdfllm_to_markdown.py --pdf path/to/file.pdf --out out_dir
输出：
  out_dir/pdfllm_document.md
  out_dir/pdfllm_report.json
"""
import os
import json
import argparse


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def try_import_reader():
    try:
        from mypymupdf4llm import LlamaMarkdownReader  # type: ignore
        return LlamaMarkdownReader
    except Exception as e:
        raise SystemExit(f"❌ 未找到 PyMuPDF4LLM，请先安装并确保可导入：{e}")


def summarize_md(md_path: str) -> dict:
    h1 = h2 = h3 = imgs = 0
    lines = 0
    with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            lines += 1
            s = line.lstrip()
            if s.startswith('# '):
                h1 += 1
            elif s.startswith('## '):
                h2 += 1
            elif s.startswith('### '):
                h3 += 1
            if '![ ' in s or s.startswith('!['):
                imgs += 1
    return {"lines": lines, "h1": h1, "h2": h2, "h3": h3, "images": imgs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, help='PDF 路径')
    parser.add_argument('--out', required=True, help='输出目录')
    args = parser.parse_args()

    pdf = args.pdf
    out_dir = args.out
    if not os.path.exists(pdf):
        raise SystemExit(f"❌ PDF不存在: {pdf}")
    ensure_dir(out_dir)

    LlamaMarkdownReader = try_import_reader()
    reader = LlamaMarkdownReader()
    try:
        md_nodes = reader.load_data(pdf)
    except Exception as e:
        raise SystemExit(f"❌ PDFLLM 导出失败: {e}")

    # 兼容 string / list
    if isinstance(md_nodes, (list, tuple)):
        parts = []
        for x in md_nodes:
            # LlamaIndex Document 节点通常有 text 属性
            txt = getattr(x, 'text', None)
            if txt is None:
                # 有些实现返回 dict-like
                try:
                    txt = x.get('text')  # type: ignore
                except Exception:
                    txt = None
            parts.append(txt if isinstance(txt, str) else str(x))
        md_text = '\n\n'.join(parts)
    else:
        # 单对象：优先取 text
        md_text = getattr(md_nodes, 'text', None) or str(md_nodes)

    md_path = os.path.join(out_dir, 'pdfllm_document.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

    report = summarize_md(md_path)
    with open(os.path.join(out_dir, 'pdfllm_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, "md_path": md_path, "report": report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
