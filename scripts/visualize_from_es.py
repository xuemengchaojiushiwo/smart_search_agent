#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ES 命中的 chunks 可视化 bbox：
- 按 knowledge_id + source_file 查询 ES 命中
- 自动定位/生成 PDF（优先使用提供的 --pdf-path；否则：
  - 若 source_file 为 .pdf，且在 python_service/file/ 下存在则直接使用
  - 若为 .xlsx/.xls，尝试使用 LibreOffice 转为 PDF
  - 若为 .txt，生成简易 PDF 以便标注
- 在对应页绘制红框与标签，输出到 rag_visualization/

Windows 运行示例（同一终端顺序执行）：
  python scripts/visualize_from_es.py --knowledge-id 101 --source-file 小红书选品.xlsx
  python scripts/visualize_from_es.py --knowledge-id 102 --source-file temp_test1.txt
  python scripts/visualize_from_es.py --knowledge-id 34  --source-file 安联美元.pdf --pdf-path python_service/file/安联美元.pdf
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import shutil


def add_python_service_to_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_dir = os.path.join(repo_root, 'python_service')
    if os.path.isdir(py_dir) and py_dir not in sys.path:
        sys.path.insert(0, py_dir)


def build_default_source_path(source_file: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(repo_root, 'python_service', 'file', source_file)
    return cand


def soffice_path() -> str:
    candidates = [
        r"C:\\Program Files\\LibreOffice\\program\\soffice.exe",
        r"C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 尝试 PATH 中的可执行
    for name in ("soffice", "soffice.exe"):
        p = shutil.which(name)
        if p:
            return p
    return ""


def convert_office_to_pdf(src_path: str, out_dir: str) -> str:
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    exe = soffice_path()
    if not exe:
        raise RuntimeError("未找到 LibreOffice (soffice)，无法将 Office 转为 PDF。请安装 LibreOffice 后重试。")
    cp = subprocess.run([
        exe, "--headless", "--norestore", "--nolockcheck",
        "--convert-to", "pdf", "--outdir", out_dir, os.path.abspath(src_path)
    ], capture_output=True, text=True, timeout=180)
    if cp.returncode != 0:
        raise RuntimeError(f"LibreOffice 转换失败: rc={cp.returncode}\nstdout={cp.stdout}\nstderr={cp.stderr}")
    # 期望同名 pdf
    base_pdf = Path(src_path).with_suffix('.pdf').name
    candidate = os.path.join(out_dir, base_pdf)
    if os.path.exists(candidate):
        return candidate
    # 兜底取 out_dir 最新的 pdf
    pdfs = [p for p in os.listdir(out_dir) if p.lower().endswith('.pdf')]
    if not pdfs:
        raise RuntimeError("未生成 PDF 文件")
    pdfs.sort(key=lambda n: os.path.getmtime(os.path.join(out_dir, n)), reverse=True)
    return os.path.join(out_dir, pdfs[0])


def create_simple_pdf_from_txt(src_path: str, out_path: str) -> str:
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


def ensure_pdf(source_file: str, pdf_path_arg: str | None, knowledge_id: int | None = None) -> str:
    if pdf_path_arg:
        if not os.path.exists(pdf_path_arg):
            raise FileNotFoundError(f"指定的 PDF 不存在: {pdf_path_arg}")
        return pdf_path_arg

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_default = build_default_source_path(source_file)
    src_alt = os.path.join(repo_root, source_file)
    ext = Path(source_file).suffix.lower()
    if ext == '.pdf':
        if os.path.exists(src_default):
            return src_default
        if os.path.exists(src_alt):
            return src_alt
        raise FileNotFoundError(f"未找到 PDF: {src_default} 或 {src_alt}，请通过 --pdf-path 指定")

    if ext in {'.xlsx', '.xls'}:
        out_dir = os.path.join(repo_root, 'python_service', 'static', 'converted', 'vis')
        src = src_default if os.path.exists(src_default) else src_alt
        if not os.path.exists(src):
            raise FileNotFoundError(f"未找到源文件: {src_default} 或 {src_alt}")
        return convert_office_to_pdf(src, out_dir)

    if ext == '.txt':
        out_path = os.path.join(repo_root, 'python_service', 'static', 'converted', 'vis', Path(source_file).with_suffix('.pdf').name)
        src = src_default if os.path.exists(src_default) else src_alt
        if not os.path.exists(src):
            raise FileNotFoundError(f"未找到源文件: {src_default} 或 {src_alt}")
        return create_simple_pdf_from_txt(src, out_path)

    if ext in {'.doc', '.docx', '.ppt', '.pptx'}:
        # 走已处理后生成的持久化PDF路径（由后端处理时创建）
        if knowledge_id is None:
            raise ValueError("未提供 knowledge_id，无法定位已生成的PDF。请使用 --pdf-path 指定PDF路径")
        stem_pdf = Path(source_file).with_suffix('.pdf').name
        target_pdf = os.path.join(repo_root, 'python_service', 'static', 'converted', str(knowledge_id), stem_pdf)
        if os.path.exists(target_pdf):
            return target_pdf
        raise FileNotFoundError(f"未找到已生成PDF: {target_pdf}。请先调用 /api/document/process 处理该文件，或通过 --pdf-path 指定")

    raise ValueError(f"暂不支持的可视化类型: {ext}")


def query_es_hits(knowledge_id: int, source_file: str) -> List[Dict[str, Any]]:
    add_python_service_to_path()
    from elasticsearch import Elasticsearch
    try:
        from config import ES_CONFIG
    except Exception as e:
        raise RuntimeError(f"无法导入 ES 配置: {e}")

    es = Elasticsearch(
        [f"http://{ES_CONFIG['host']}:{ES_CONFIG['port']}"] ,
        basic_auth=(ES_CONFIG.get('username') or None, ES_CONFIG.get('password') or None) if ES_CONFIG.get('username') else None,
        verify_certs=ES_CONFIG.get('verify_certs', False)
    )
    index = ES_CONFIG['index']
    try:
        es.indices.refresh(index=index)
    except Exception:
        pass

    base_source = [
        "content", "knowledge_id", "knowledge_name", "description", "tags", "effective_time",
        "source_file", "page_num", "chunk_index", "bbox", "positions"
    ]

    def run(q):
        return es.search(index=index, body=q).get('hits', {}).get('hits', [])

    q1 = {
        "size": 10,
        "query": {"bool": {"must": [{"term": {"knowledge_id": knowledge_id}}, {"term": {"source_file.keyword": source_file}}]}},
        "_source": base_source,
    }
    hits = run(q1)
    if hits:
        return hits

    q2 = {
        "size": 10,
        "query": {"bool": {"must": [{"term": {"knowledge_id": knowledge_id}}, {"match_phrase": {"source_file": source_file}}]}},
        "_source": base_source,
    }
    hits = run(q2)
    if hits:
        return hits

    q3 = {"size": 10, "query": {"term": {"knowledge_id": knowledge_id}}, "_source": base_source}
    return run(q3)


def draw_hits(pdf_path: str, hits: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    out_dir = Path('rag_visualization')
    out_dir.mkdir(exist_ok=True)
    base_doc = fitz.open(pdf_path)
    outputs: List[str] = []

    for i, h in enumerate(hits[:limit], start=1):
        src = h.get('_source', {})
        bbox = src.get('bbox') or []
        page_num = int(src.get('page_num') or 1)
        content = (src.get('content') or '')

        # 新建独立文档用于绘制，避免“source document must not equal target”错误
        page = base_doc.load_page(page_num - 1)
        new_doc = fitz.open()
        # 注册中文字体，避免覆盖文字出现问号
        def _register_cjk_font(_doc: fitz.Document) -> str:
            try:
                candidates = [
                    r"C:\\Windows\\Fonts\\msyh.ttc",
                    r"C:\\Windows\\Fonts\\msyh.ttf",
                    r"C:\\Windows\\Fonts\\simsun.ttc",
                    r"C:\\Windows\\Fonts\\simhei.ttf",
                ]
                for p in candidates:
                    if os.path.exists(p):
                        _doc.insert_font(fontname="CJK", fontfile=p)
                        return "CJK"
            except Exception:
                pass
            return "helv"
        cjk_font = _register_cjk_font(new_doc)
        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(new_page.rect, base_doc, page_num - 1)

        if isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
            new_page.draw_rect(rect, color=(1, 0, 0), width=3)
            label = f"Hit {i}: page={page_num} idx={src.get('chunk_index')}"
            new_page.insert_text((rect.x0, max(10, rect.y0 - 12)), label, fontsize=10, fontname=cjk_font, color=(1, 0, 0))
            preview = content[:60].replace('\n', ' ')
            new_page.insert_text((rect.x0, rect.y1 + 14), f"{preview}", fontsize=9, fontname=cjk_font, color=(0, 0, 1))

        out_png = out_dir / f"es_hit_{i}_page_{page_num}.png"
        pix = new_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(str(out_png))
        outputs.append(str(out_png))
        new_doc.close()
    base_doc.close()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge-id', type=int, required=True)
    parser.add_argument('--source-file', required=True, help='ES 中的 source_file（原始文件名）')
    parser.add_argument('--pdf-path', default=None, help='若已知 PDF 路径可直接提供，优先使用')
    args = parser.parse_args()

    try:
        pdf_path = ensure_pdf(args.source_file, args.pdf_path, args.knowledge_id)
        hits = query_es_hits(args.knowledge_id, args.source_file)
        if not hits:
            print('❌ ES 未命中，无法可视化')
            sys.exit(2)
        outputs = draw_hits(pdf_path, hits)
        print('\n✅ 已生成可视化图片:')
        for p in outputs:
            print('  -', p)
    except Exception as e:
        print('❌ 失败:', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()


