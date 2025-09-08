#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ES中存储的chunk bbox，在PDF上可视化RAG引用框
用法示例：
  python scripts/visualize_rag_refs.py \
    --pdf python_service/file/安联美元.pdf \
    --index knowledge_base_new \
    --source-file 安联美元.pdf \
    --chunks 4,6,2 \
    --outdir rag_refs_visual

说明：
- 从ES读取指定source_file的chunks，按chunk_index筛选
- 使用每个chunk的bbox绘制矩形
- 按页输出图片：page_1.png, page_2.png, ...
"""

import argparse
import os
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from elasticsearch import Elasticsearch

# 复用python服务的配置
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_service'))
    from config import ES_CONFIG
except Exception:
    ES_CONFIG = {
        'host': 'localhost',
        'port': 9200,
        'index': 'knowledge_base_new',
        'username': 'elastic',
        'password': 'password',
        'verify_certs': False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在PDF上可视化RAG引用bbox")
    parser.add_argument('--pdf', required=True, help='PDF文件路径')
    parser.add_argument('--index', default=ES_CONFIG.get('index', 'knowledge_base_new'), help='ES索引名')
    parser.add_argument('--source-file', required=True, help='ES中文件名(source_file)')
    parser.add_argument('--chunks', required=True, help='以逗号分隔的chunk_index列表，如 4,6,2')
    parser.add_argument('--outdir', default='rag_refs_visual', help='输出目录')
    return parser.parse_args()


def connect_es() -> Elasticsearch:
    hosts = [f"http://{ES_CONFIG['host']}:{ES_CONFIG['port']}"]
    es = Elasticsearch(
        hosts,
        basic_auth=(ES_CONFIG.get('username'), ES_CONFIG.get('password')) if ES_CONFIG.get('username') else None,
        verify_certs=ES_CONFIG.get('verify_certs', False),
    )
    return es


def fetch_chunks_by_indices(es: Elasticsearch, index: str, source_file: str, chunk_indices: List[int]) -> List[Dict]:
    # 拉取该索引最多1000条，内存过滤，避免字段.keyword带来的不匹配
    resp = es.search(
        index=index,
        size=1000,
        query={"match_all": {}},
        _source=["chunk_index", "bbox", "page_num", "source_file"],
    )
    hits = resp.get('hits', {}).get('hits', [])
    all_docs = [h.get('_source', {}) for h in hits]

    # 调试：统计可用的source_file集合
    available_sources = sorted({d.get('source_file') for d in all_docs if d.get('source_file')})
    if not any(d.get('source_file') == source_file for d in all_docs):
        print(f"未在索引 {index} 中找到指定source_file: {source_file}")
        print(f"可用的source_file有: {available_sources}")
        return []

    # 先按source_file过滤
    docs_by_file = [d for d in all_docs if d.get('source_file') == source_file]
    # 建立chunk_index到文档映射
    by_index: Dict[int, Dict] = {}
    for d in docs_by_file:
        ci = d.get('chunk_index')
        if isinstance(ci, int):
            by_index[ci] = d

    # 收集需要的索引
    result = []
    for ci in chunk_indices:
        if ci in by_index:
            result.append(by_index[ci])
        else:
            print(f"警告: 未找到 chunk_index={ci} 的文档")
    return result


def draw_bboxes_on_pdf(pdf_path: str, refs: List[Dict], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # 按页分组
    page_to_boxes: Dict[int, List[Tuple[List[float], int]]] = {}
    for r in refs:
        page_num = int(r.get('page_num', 1))
        bbox = r.get('bbox') or []
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        page_to_boxes.setdefault(page_num, []).append((bbox, r.get('chunk_index', -1)))

    # 逐页渲染：使用单独的新doc承载原始页面绘制，避免“source must not equal target”
    for page_num, items in page_to_boxes.items():
        src_page = doc.load_page(page_num - 1)
        src_rect = src_page.rect

        # 构造承载页面
        vis_doc = fitz.open()
        vis_page = vis_doc.new_page(width=src_rect.width, height=src_rect.height)
        vis_page.show_pdf_page(vis_page.rect, doc, page_num - 1)

        # 颜色循环
        colors = [
            (0.9, 0.2, 0.2), (0.9, 0.8, 0.2), (0.4, 0.9, 0.2),
            (0.2, 0.9, 0.6), (0.2, 0.6, 0.9), (0.4, 0.2, 0.9), (0.9, 0.2, 0.8)
        ]

        for i, (bbox, chunk_index) in enumerate(items):
            x0, y0, x1, y1 = bbox
            rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
            color = colors[i % len(colors)]
            vis_page.draw_rect(rect, color=color, width=1.2)
            # 标注chunk index
            vis_page.insert_text(
                fitz.Point(rect.x0 + 3, rect.y0 + 12),
                f"chunk {chunk_index}",
                fontsize=10,
                color=color,
            )

        # 输出png
        png_path = os.path.join(outdir, f"page_{page_num}.png")
        pix = vis_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(png_path)
        print(f"保存: {png_path}")

    doc.close()


def main():
    args = parse_args()
    chunk_indices = [int(x) for x in args.chunks.split(',') if x.strip().isdigit()]

    es = connect_es()
    refs = fetch_chunks_by_indices(
        es=es,
        index=args.index,
        source_file=args.source_file,
        chunk_indices=chunk_indices,
    )

    if not refs:
        print("未找到匹配的引用chunk，请检查参数 --source-file 与 --chunks")
        return

    print(f"将可视化 {len(refs)} 个引用chunk: {[r.get('chunk_index') for r in refs]}")
    draw_bboxes_on_pdf(args.pdf, refs, args.outdir)


if __name__ == '__main__':
    main()
