#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键验证：
1) 用本地 mypymupdf4llm 生成带 <sub>pos: ...</sub> 的 Markdown
2) 转换为 aligned_positions.json（页和 bbox）
3) 生成每页带框 PNG 预览，并输出报告

用法：
  python -u python_service/validate_blocks_preview.py --pdf python_service/file/安联美元.pdf --out out_pdfllm_allianz
"""
import os
import sys
import argparse
from subprocess import run, PIPE

sys.path.insert(0, os.path.abspath('.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) 生成带位置的 MD
    from mypymupdf4llm.helpers.pymupdf_rag import to_markdown as to_md
    md_text = to_md(args.pdf, emit_positions=True)
    md_pos_path = os.path.join(args.out, 'pdfllm_document_with_pos.md')
    with open(md_pos_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

    # 2) 转 aligned_positions.json
    aligned_path = os.path.join(args.out, 'aligned_positions.json')
    run([sys.executable, '-u', 'python_service/md_pos_to_aligned.py', '--md', md_pos_path, '--out', aligned_path], check=True)

    # 3) 预览
    preview_dir = os.path.join(args.out, 'preview')
    os.makedirs(preview_dir, exist_ok=True)
    p = run([sys.executable, '-u', 'python_service/preview_alignment.py', '--pdf', args.pdf, '--aligned', aligned_path, '--out', preview_dir], check=False, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    print(p.stdout.strip() or p.stderr.strip())
    print(preview_dir)


if __name__ == '__main__':
    main()


