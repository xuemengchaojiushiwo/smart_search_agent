#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将目录中的非PDF文件转换为PDF，输出到: python_service/static/converted/{knowledge_id}/
- 支持: doc/docx/xls/xlsx/ppt/pptx/txt
- 默认扫描: python_service/file
- 默认knowledge_id: 29
- 依赖: 已安装LibreOffice(soffice)，以及 PyMuPDF

用法（PowerShell，逐条执行）：
  python scripts/batch_convert_to_pdf.py
  # 或者自定义参数：
  python scripts/batch_convert_to_pdf.py --input-dir python_service/file --knowledge-id 29 --out-root python_service/static/converted --soffice "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:
    print("缺少依赖: PyMuPDF，请先安装: pip install pymupdf")
    sys.exit(1)

COMMON_SOFFICE_CANDIDATES = [
    r"C:\\Program Files\\LibreOffice\\program\\soffice.exe",
    r"C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
]


def find_soffice(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        print(f"指定的 soffice 不存在: {explicit}")
    for cand in COMMON_SOFFICE_CANDIDATES:
        if Path(cand).exists():
            return cand
    from shutil import which
    w = which("soffice") or which("soffice.exe")
    if w:
        return w
    raise FileNotFoundError("未找到soffice，请安装LibreOffice或通过 --soffice 指定路径。可用 winget 安装: winget install TheDocumentFoundation.LibreOffice")


def convert_with_libreoffice(src_path: str, soffice_path: str, out_dir: str, timeout_sec: int = 300) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        soffice_path,
        "--headless",
        "--norestore",
        "--nolockcheck",
        "--convert-to", "pdf",
        "--outdir", out_dir,
        os.path.abspath(src_path),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    if cp.returncode != 0:
        raise RuntimeError(f"LibreOffice 转换失败: rc={cp.returncode}\nstdout={cp.stdout}\nstderr={cp.stderr}")
    expected = os.path.join(out_dir, Path(src_path).with_suffix('.pdf').name)
    if os.path.exists(expected):
        return expected
    # 兜底找最新pdf
    pdfs = [p for p in os.listdir(out_dir) if p.lower().endswith('.pdf')]
    if not pdfs:
        raise RuntimeError(f"未找到输出PDF。stdout={cp.stdout}\nstderr={cp.stderr}")
    pdfs.sort(key=lambda n: os.path.getmtime(os.path.join(out_dir, n)), reverse=True)
    return os.path.join(out_dir, pdfs[0])


def simple_txt_to_pdf(src_path: str, out_path: str) -> str:
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    rect = fitz.Rect(40, 50, 555, 800)
    page.insert_textbox(rect, text[:50000], fontsize=11, fontname="helv", align=0)
    doc.save(out_path)
    doc.close()
    return out_path


def validate_pages(pdf_path: str) -> int:
    d = fitz.open(pdf_path)
    try:
        return len(d)
    finally:
        d.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='python_service/file', help='输入目录（扫描非PDF文件）')
    ap.add_argument('--knowledge-id', type=int, default=29)
    ap.add_argument('--out-root', default='python_service/static/converted', help='输出根目录')
    ap.add_argument('--soffice', default=None, help='LibreOffice soffice.exe 路径，留空自动探测')
    args = ap.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    out_dir = os.path.join(args.out_root, str(args.knowledge_id))
    os.makedirs(out_dir, exist_ok=True)

    # 尝试寻找soffice（txt可不需要）
    try:
        soffice = find_soffice(args.soffice)
    except Exception as e:
        soffice = None
        print(f"警告: 未检测到LibreOffice，将仅对TXT进行降级转换。详情: {e}")

    support_exts = {'.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt'}

    print(f"开始转换: {input_dir} → {out_dir}")
    converted = []
    failed = []

    for name in os.listdir(input_dir):
        src = os.path.join(input_dir, name)
        if not os.path.isfile(src):
            continue
        ext = Path(name).suffix.lower()
        if ext == '.pdf':
            continue
        if ext not in support_exts:
            continue

        try:
            if ext == '.txt':
                target = os.path.join(out_dir, Path(name).with_suffix('.pdf').name)
                pdf_path = simple_txt_to_pdf(src, target)
            else:
                if not soffice:
                    raise RuntimeError("未检测到LibreOffice，无法转换此类型。")
                pdf_path = convert_with_libreoffice(src, soffice, out_dir)

            pages = validate_pages(pdf_path)
            converted.append((name, pdf_path, pages))
            print(f"✅ {name} → {pdf_path}  页数: {pages}")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"❌ {name} 转换失败: {e}")

    print("\n转换完成：")
    print(f"成功: {len(converted)} 个，失败: {len(failed)} 个")
    if converted:
        print("成功列表：")
        for name, pdf_path, pages in converted:
            print(f"  - {name} → {pdf_path}  页数: {pages}")
    if failed:
        print("失败列表：")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == '__main__':
    main()

