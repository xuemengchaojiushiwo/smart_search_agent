#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PDFLLM 生成的 Markdown 文本进行页内定位：为每个段落计算
- page_num
- char_start / char_end （在该页纯文本上的字符区间）
- bbox_union （并集框，便于前端高亮）

用法：
  python align_md_positions.py --pdf path/to.pdf --md path/to/pdfllm_document.md --out aligned_positions.json

说明：
- 不依赖 PyMuPDF Pro，仅用开源 PyMuPDF（fitz）
- 锚点匹配策略：优先用段落开头/结尾锚点在页文本中查找；找不到再做模糊匹配兜底
- 对很短或噪声段落（< 30 字）跳过
"""
import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
import difflib

import fitz  # PyMuPDF (开源)


def read_file_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def strip_markdown(md: str) -> str:
    # 去掉图片/链接/标题/列表的标记，保留纯文本
    s = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", md)  # ![alt](url)
    s = re.sub(r"\[[^\]]*\]\([^\)]*\)", " ", s)    # [text](url)
    s = re.sub(r"^[#>*\-\s]+", "", s, flags=re.MULTILINE)  # 标题/列表/引用符
    s = re.sub(r"[`*_]{1,3}", "", s)  # 行内强调
    s = re.sub(r"<[^>]+>", " ", s)  # 简单去标签
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_paragraphs(md_text: str) -> List[str]:
    """以空行切段，做基础清洗（不做去重，尽量完整保留 PDFLLM 段落）。"""
    raw_paras = re.split(r"\n\s*\n", md_text)
    paras: List[str] = []
    for p in raw_paras:
        p2 = strip_markdown(p)
        if len(p2) >= 30 and re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", p2):
            paras.append(p2)
    return paras


def build_page_text_and_word_index(page: fitz.Page) -> Tuple[str, List[Tuple[int, int, Tuple[float, float, float, float]]]]:
    """返回 (page_text, word_entries)，word_entries: [(start,end,(x0,y0,x1,y1)), ...]"""
    try:
        words = page.get_text("words")  # (x0,y0,x1,y1,word,block_no,line_no,word_no)
        words_sorted = sorted(words, key=lambda w: (int(w[5]), int(w[6]), int(w[7])))
        parts: List[str] = []
        entries: List[Tuple[int, int, Tuple[float, float, float, float]]] = []
        last_key = None
        current = 0
        for x0, y0, x1, y1, token, bno, lno, wno in words_sorted:
            key = (bno, lno)
            if last_key is None:
                parts.append(token)
                start = current
                end = start + len(token)
                entries.append((start, end, (float(x0), float(y0), float(x1), float(y1))))
                current = end
            else:
                if key != last_key:
                    parts.append("\n")
                    current += 1
                    parts.append(token)
                    start = current
                    end = start + len(token)
                    entries.append((start, end, (float(x0), float(y0), float(x1), float(y1))))
                    current = end
                else:
                    parts.append(" ")
                    current += 1
                    parts.append(token)
                    start = current
                    end = start + len(token)
                    entries.append((start, end, (float(x0), float(y0), float(x1), float(y1))))
                    current = end
            last_key = key
        return "".join(parts), entries
    except Exception:
        txt = page.get_text() or ""
        return txt, []


def compute_bbox_union(entries: List[Tuple[int, int, Tuple[float, float, float, float]]], start: int, end: int) -> List[float]:
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    found = False
    for s, e, (a, b, c, d) in entries:
        if e <= start or s >= end:
            continue
        found = True
        x0 = min(x0, a)
        y0 = min(y0, b)
        x1 = max(x1, c)
        y1 = max(y1, d)
    return [x0, y0, x1, y1] if found else []


def find_positions_in_page(para: str, page_text: str) -> Tuple[int, int]:
    """先用开头/结尾锚点直接查找；不成再用模糊匹配，返回 (start,end) 或 (-1,-1)。"""
    pt_low = page_text.lower()
    p = para.strip()
    p_low = p.lower()
    # 首段/尾段锚点
    head = re.sub(r"\s+", " ", p_low)[:60]
    tail = re.sub(r"\s+", " ", p_low)[-60:]
    hs = pt_low.find(head)
    if hs >= 0:
        # 尝试尾锚
        te = pt_low.find(tail, hs)
        if te >= 0:
            return hs, te + len(tail)
        return hs, hs + min(len(p_low), 4000)
    # 模糊：取 page 与段落的最长公共子序列比例
    try:
        m = difflib.SequenceMatcher(None, p_low[:1000], pt_low)
        a, b, size = m.find_longest_match(0, len(p_low[:1000]), 0, len(pt_low))
        if size >= 30:
            return b, min(b + len(p_low), b + size + 1000)
    except Exception:
        pass
    return -1, -1


def align_md_to_pdf(pdf_path: str, md_path: str) -> List[Dict[str, Any]]:
    md_raw = read_file_text(md_path)
    paras = split_paragraphs(md_raw)
    doc = fitz.open(pdf_path)
    pages: List[Tuple[str, List[Tuple[int, int, Tuple[float, float, float, float]]]]] = []
    page_texts_nrm: List[str] = []
    def _normalize(s: str) -> str:
        s2 = (s or "").lower().strip()
        s2 = re.sub(r"[\s\u3000]+", "", s2)
        s2 = re.sub(r"[\-—–·•\.,;:!\?\(\)\[\]\{\}<>\|\/\\\*\^\$\#\+_=~`'\"]+", "", s2)
        return s2
    for page in doc:
        ptxt, entries = build_page_text_and_word_index(page)
        pages.append((ptxt, entries))
        page_texts_nrm.append(_normalize(ptxt))

    results: List[Dict[str, Any]] = []
    for idx, para in enumerate(paras):
        placed = False
        best = None
        for pno, (ptxt, entries) in enumerate(pages, start=1):
            s, e = find_positions_in_page(para, ptxt)
            if s >= 0:
                bbox = compute_bbox_union(entries, s, e) if entries else []
                best = {
                    "para_index": idx,
                    "page_num": pno,
                    "char_start": s,
                    "char_end": e,
                    "bbox_union": bbox,
                    "text": para[:200]
                }
                placed = True
                break
        if not placed:
            # Fallback：按页面相似度选择最可能的页码，暂不标注具体字符位置
            para_nrm = _normalize(para)
            best_pno = -1
            best_ratio = 0.0
            for pno, pt_nrm in enumerate(page_texts_nrm, start=1):
                try:
                    r = difflib.SequenceMatcher(None, para_nrm[:1000], pt_nrm).ratio()
                except Exception:
                    r = 0.0
                if r > best_ratio:
                    best_ratio = r
                    best_pno = pno
            # 如果相似度尚可（阈值可根据材料调整），先赋页码，位置后续优化
            if best_ratio >= 0.12:
                best = {"para_index": idx, "page_num": best_pno, "char_start": -1, "char_end": -1, "bbox_union": [], "text": para[:200]}
            else:
                best = {"para_index": idx, "page_num": -1, "char_start": -1, "char_end": -1, "bbox_union": [], "text": para[:200]}
        results.append(best)

    # 后处理：同页字符区间高度重叠的结果合并/去重，仅保留首个
    def overlap_ratio(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        s1, e1 = a; s2, e2 = b
        inter = max(0, min(e1, e2) - max(s1, s2))
        denom = max(1, min(e1 - s1, e2 - s2))
        return inter / denom

    filtered: List[Dict[str, Any]] = []
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for it in results:
        by_page.setdefault(int(it.get('page_num', -1)), []).append(it)

    for pno, items in by_page.items():
        # 按起始位置排序
        items_sorted = sorted(items, key=lambda x: (int(x.get('char_start', -1)), int(x.get('char_end', -1))))
        kept: List[Dict[str, Any]] = []
        for it in items_sorted:
            s = int(it.get('char_start', -1)); e = int(it.get('char_end', -1))
            if s < 0 or e <= s:
                kept.append(it)
                continue
            dup = False
            for prev in kept[-10:]:
                ps = int(prev.get('char_start', -1)); pe = int(prev.get('char_end', -1))
                if ps < 0 or pe <= ps:
                    continue
                if overlap_ratio((s, e), (ps, pe)) >= 0.85:
                    dup = True
                    break
            if not dup:
                kept.append(it)
        filtered.extend(kept)

    # 还原为 para_index 顺序
    filtered_sorted = sorted(filtered, key=lambda x: int(x.get('para_index', 0)))
    return filtered_sorted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True, help='PDF 路径')
    ap.add_argument('--md', required=True, help='PDFLLM 生成的 Markdown 路径')
    ap.add_argument('--out', required=True, help='输出 JSON 路径')
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"❌ PDF不存在: {args.pdf}")
    if not os.path.exists(args.md):
        raise SystemExit(f"❌ MD不存在: {args.md}")

    results = align_md_to_pdf(args.pdf, args.md)
    ensure_dir(os.path.dirname(args.out) or '.')
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"ok": True, "count": len(results), "out": args.out}, ensure_ascii=False))


if __name__ == '__main__':
    main()
