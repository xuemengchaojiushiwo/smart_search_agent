#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PDF 解析为 Markdown（含：标题层级、文本块、图片、页码与粗定位），用于验证接近 PDFLLM 的效果。
- 仅使用开源 PyMuPDF（fitz），无需 Pro
- 按页面顺序提取 block（文本/图片），保持版式顺序
- 通过字体大小分级，启发式生成 #/##/### 标题
- 图片抽取并在 Markdown 中就地引用
- 输出：
  - <out_dir>/document.md         结构化 Markdown
  - <out_dir>/images/...          页内图片资源
  - <out_dir>/layout_report.json  统计报告（页数/图片数/标题数等）
用法：
  python pdf_layout_to_markdown.py --pdf path/to/file.pdf --out out_dir
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple
import re
import math
import difflib
from collections import deque

import fitz  # PyMuPDF（开源版）

try:
    import pdfplumber  # 表格抽取补充
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_font_sizes(doc: fitz.Document) -> List[float]:
    sizes = []
    for page in doc:
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            if block.get("type") == 0:  # text
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = float(span.get("size", 0) or 0)
                        if size > 0:
                            sizes.append(size)
    return sizes


def size_to_heading_levels(sizes: List[float]) -> List[float]:
    """返回从大到小的3个阈值，用于H1/H2/H3切分（简单分位数法）。"""
    if not sizes:
        return []
    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    # 90/75/60 分位点作阈值
    q90 = sizes_sorted[int(0.9 * (n - 1))]
    q75 = sizes_sorted[int(0.75 * (n - 1))]
    q60 = sizes_sorted[int(0.6 * (n - 1))]
    return [q90, q75, q60]


def infer_heading(size: float, thresholds: List[float], font_name: str) -> int:
    """根据字体大小与字体名（含Bold）估计标题级别。返回 0 表示正文。"""
    if not thresholds:
        return 0
    bold = "bold" in (font_name or "").lower()
    if size >= thresholds[0] or (bold and size >= thresholds[1]):
        return 1
    if size >= thresholds[1]:
        return 2
    if size >= thresholds[2]:
        return 3
    return 0


def extract_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    """从页面获得按版式顺序的block列表（文本/图片），带 bbox 等基础信息。
    包含多重回退：rawdict → blocks(sort) → 整页text。
    """
    blocks: List[Dict[str, Any]] = []

    # 1) 首选 rawdict（可拿到字体大小用于标题启发）
    try:
        raw = page.get_text("rawdict")
        total_chars = 0
        for bi, block in enumerate(raw.get("blocks", [])):
            btype = block.get("type")  # 0=text, 1=image
            bbox = block.get("bbox")
            if btype == 0:
                lines_joined: List[str] = []
                spans_meta = []
                for line in block.get("lines", []):
                    line_parts = []
                    for span in line.get("spans", []):
                        stext = span.get("text", "")
                        if stext:
                            line_parts.append(stext)
                            spans_meta.append({
                                "size": float(span.get("size", 0) or 0),
                                "font": span.get("font", ""),
                            })
                    if line_parts:
                        lines_joined.append("".join(line_parts))
                text = "\n".join(lines_joined).strip()
                # 连字符换行修复：将 "-\n" 合并为 ""
                text = re.sub(r"-\n(?=\w)", "", text)
                total_chars += len(text)
                max_size = max([m["size"] for m in spans_meta], default=0.0)
                any_font = spans_meta[0]["font"] if spans_meta else ""
                if text:
                    blocks.append({
                        "type": "text",
                        "bbox": bbox,
                        "text": text,
                        "max_size": max_size,
                        "font": any_font,
                        "block_index": bi
                    })
            elif btype == 1:
                blocks.append({
                    "type": "image",
                    "bbox": bbox,
                    "image_name": block.get("image"),
                    "block_index": bi
                })
        if any(b.get("type") == "text" for b in blocks) and total_chars > 0:
            return blocks
    except Exception:
        pass

    # 2) 回退到 blocks（简化，只能得到文本与bbox）
    try:
        blk = page.get_text("blocks", sort=True)
        for bi, b in enumerate(blk):
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type == 0 and text and text.strip():
                fixed = re.sub(r"-\n(?=\w)", "", text.strip())
                blocks.append({
                    "type": "text",
                    "bbox": [x0, y0, x1, y1],
                    "text": fixed,
                    "max_size": 0.0,
                    "font": "",
                    "block_index": bi
                })
        if any(b.get("type") == "text" for b in blocks):
            return blocks
    except Exception:
        pass

    # 3) 最后回退整页 text（无版式，保证至少有可读文本）
    try:
        text = page.get_text("text") or ""
        if text.strip():
            fixed = re.sub(r"-\n(?=\w)", "", text.strip())
            blocks.append({
                "type": "text",
                "bbox": list(page.rect),
                "text": fixed,
                "max_size": 0.0,
                "font": "",
                "block_index": 0
            })
            return blocks
    except Exception:
        pass

    # 4) 若确为扫描件或无可提取文本，仅返回图片信息（若有）
    try:
        images = page.get_images(full=True)
        for bi, img in enumerate(images):
            blocks.append({
                "type": "image",
                "bbox": [],
                "image_name": f"xref:{img[0]}",
                "block_index": bi
            })
    except Exception:
        pass
    return blocks


def export_image(page: fitz.Page, image_dir: str, bbox: List[float], index_hint: int) -> str:
    """导出页面中与 bbox 匹配度最高的图像（近似法）。返回相对路径。"""
    ensure_dir(image_dir)
    # 简化：遍历 page.get_images，取第一张像素图导出
    # 更好的做法是根据 bbox 与图像矩阵做几何匹配，这里以近似为主用于验证
    images = page.get_images(full=True)
    if not images:
        return ""
    idx = max(0, min(index_hint, len(images) - 1))
    xref = images[idx][0]
    pix = fitz.Pixmap(page.parent, xref)
    # 转成RGB避免带alpha的CMYK等
    if pix.n >= 5:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    out_name = f"img_p{page.number+1}_{index_hint}.png"
    out_path = os.path.join(image_dir, out_name)
    pix.save(out_path)
    return os.path.join("images", out_name)


def extract_tables_by_pdfplumber(pdf_path: str) -> Dict[int, List[List[str]]]:
    """用 pdfplumber 抽取表格，返回 {page_number: [table_rows]}。失败则返回空。"""
    results: Dict[int, List[List[str]]] = {}
    if not PDFPLUMBER_AVAILABLE:
        return results
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                    merged: List[List[str]] = []
                    for table in tables:
                        # 规范化为字符串
                        for row in table:
                            merged.append([(cell if cell is not None else '').strip() for cell in row])
                    if merged:
                        results[i + 1] = merged
                except Exception:
                    continue
    except Exception:
        return {}
    return results


def normalize_text(s: str) -> str:
    """归一化文本：用于近重复抑制、表格指纹与页眉页脚清洗。
    - 小写
    - 移除空白与常见标点
    - 保留中英文与数字
    """
    s2 = (s or "").lower().strip()
    s2 = re.sub(r"[\s\u3000]+", "", s2)
    s2 = re.sub(r"[\-—–·•\.,;:!\?\(\)\[\]\{\}<>\|\/\\\*\^\$\#\+_=~`'\"]+", "", s2)
    return s2


def _looks_like_char_grid(table: List[List[str]]) -> bool:
    """判断表格是否为逐字切分的“字符网格”（其实是段落被切成单字）。"""
    if not table or not table[0]:
        return False
    # 统计单元格长度分布
    cells = [c for row in table for c in row if c is not None]
    if not cells:
        return False
    lengths = [len((c or '').strip()) for c in cells]
    cols = max(len(row) for row in table)
    # 放宽列阈值，适配较窄字符网格
    if cols < 6:
        return False
    short_ratio = sum(1 for L in lengths if L <= 1) / max(1, len(lengths))
    # 列较多且大多数单元格为单字符，基本可判定为字符网格
    return short_ratio >= 0.6


def _char_grid_to_paragraph(table: List[List[str]]) -> str:
    """将字符网格还原为连续段落文本。"""
    rows = []
    for row in table:
        cells = [(c or '').strip() for c in row]
        rows.append(''.join(cells))
    text = ''.join(rows)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_repeating_headers_footers(doc: fitz.Document) -> Tuple[set, set]:
    """启发式检测跨页重复的页眉/页脚文本（用于清洗）。
    基于页面上/下 15% 高度区域聚合后做归一化计数，阈值按页数的 50%。
    返回的是归一化后的 key 集合，调用方需同样归一化对比。
    """
    headers: Dict[str, int] = {}
    footers: Dict[str, int] = {}
    total_pages = len(doc)

    for page in doc:
        blocks = extract_blocks(page)
        height = float(page.rect.height or 0.0)
        top_limit = (page.rect.y0 or 0.0) + 0.15 * height
        bottom_limit = (page.rect.y1 or height) - 0.15 * height

        top_texts: List[str] = []
        bottom_texts: List[str] = []

        for b in blocks:
            if b.get("type") != "text":
                continue
            bbox = b.get("bbox") or [0, 0, 0, 0]
            y0 = float(bbox[1] or 0.0)
            y1 = float(bbox[3] or 0.0)
            txt = (b.get("text") or "").strip()
            if not txt:
                continue
            # 顶部区域
            if y0 <= top_limit:
                top_texts.append(txt)
            # 底部区域
            if y1 >= bottom_limit:
                bottom_texts.append(txt)

        if top_texts:
            head_txt = " ".join(top_texts)[:160]
            head_key = normalize_text(head_txt)
            if len(head_key) >= 10:
                headers[head_key] = headers.get(head_key, 0) + 1
        if bottom_texts:
            foot_txt = " ".join(bottom_texts)[:160]
            foot_key = normalize_text(foot_txt)
            if len(foot_key) >= 10:
                footers[foot_key] = footers.get(foot_key, 0) + 1

    # 选择出现频率 >= 50% 页的文本作为页眉/页脚
    threshold = max(2, int(0.5 * total_pages))
    header_set = {t for t, c in headers.items() if c >= threshold}
    footer_set = {t for t, c in footers.items() if c >= threshold}
    return header_set, footer_set


def _decide_columns_by_histogram(page: fitz.Page, blocks: List[Dict[str, Any]]) -> Tuple[bool, float]:
    """基于x中心的直方图判断是否双栏，并返回分割中线。"""
    width = float(page.rect.width or 0)
    centers = []
    for b in blocks:
        bbox = b.get("bbox") or [0, 0, 0, 0]
        cx = (bbox[0] + bbox[2]) / 2.0
        if cx > 0:
            centers.append(cx)
    if len(centers) < 8:
        return False, width / 2.0
    # 10 桶直方图
    bins = 10
    hist = [0] * bins
    for cx in centers:
        idx = min(bins - 1, max(0, int(cx / max(1.0, width) * bins)))
        hist[idx] += 1
    # 找两个主峰
    peaks = sorted(range(bins), key=lambda i: hist[i], reverse=True)[:2]
    if len(peaks) < 2:
        return False, width / 2.0
    p1, p2 = sorted(peaks)
    sep = abs(p2 - p1) / bins
    # 两峰间距至少 0.2 宽度，且峰值够大
    if sep >= 0.2 and hist[p1] >= 2 and hist[p2] >= 2:
        # 中线取两峰中点
        mid = width * ((p1 + p2 + 1) / 2.0 / bins)
        return True, mid
    return False, width / 2.0


def make_markdown(doc: fitz.Document, out_dir: str, pdf_path: str) -> Dict[str, Any]:
    ensure_dir(out_dir)
    image_dir = os.path.join(out_dir, "images")
    ensure_dir(image_dir)

    sizes = collect_font_sizes(doc)
    thresholds = size_to_heading_levels(sizes)

    md_lines: List[str] = []
    report = {
        "pages": len(doc),
        "images": 0,
        "text_blocks": 0,
        "headings": {"h1": 0, "h2": 0, "h3": 0},
        "tables": 0
    }

    # 预抽取表格
    tables_map: Dict[int, List[List[str]]] = extract_tables_by_pdfplumber(pdf_path)

    # 页眉/页脚候选，用于清洗
    header_set, footer_set = detect_repeating_headers_footers(doc)

    seen_norm = set()
    recent_norm = deque(maxlen=500)

    for page in doc:
        pno = page.number + 1
        md_lines.append(f"\n\n<!-- Page {pno} -->\n")
        blocks = extract_blocks(page)
        img_idx = 0
        # 双栏判定（直方图法）
        two_cols, mid = _decide_columns_by_histogram(page, blocks)

        def sort_key(b: Dict[str, Any]):
            bbox = b.get("bbox") or [0, 0, 0, 0]
            y = round(bbox[1], 1)
            x = round(bbox[0], 1)
            col = 0
            if two_cols:
                cx = (bbox[0] + bbox[2]) / 2.0
                col = 0 if cx < mid else 1
            return (col, y, x, b.get("block_index", 0))

        blocks_sorted = sorted(blocks, key=sort_key)

        # 文本段落合并：同列、y间隔较小且左边界接近则合并为一个段
        merged_blocks: List[Dict[str, Any]] = []
        # 放宽阈值，减少被切碎的相邻段落
        GAP_Y = 18.0  # 可调阈值（像素）
        X_TOL = 12.0
        def block_column(b: Dict[str, Any]) -> int:
            if not two_cols:
                return 0
            bbox = b.get("bbox") or [0, 0, 0, 0]
            cx = (bbox[0] + bbox[2]) / 2.0
            return 0 if cx < mid else 1

        for b in blocks_sorted:
            if b.get("type") != "text" or not b.get("text"):
                merged_blocks.append(b)
                continue
            if not merged_blocks or merged_blocks[-1].get("type") != "text":
                merged_blocks.append(b)
                continue
            prev = merged_blocks[-1]
            # 列一致且位置相邻
            col_ok = block_column(prev) == block_column(b)
            by = (b.get("bbox") or [0, 0, 0, 0])[1]
            py = (prev.get("bbox") or [0, 0, 0, 0])[3]
            bx = (b.get("bbox") or [0, 0, 0, 0])[0]
            px = (prev.get("bbox") or [0, 0, 0, 0])[0]
            if col_ok and (by - py) <= GAP_Y and abs(bx - px) <= X_TOL:
                # 合并文本，处理标点与空格
                joiner = "\n" if prev["text"].endswith(('.', '。', '！', '？')) else " "
                prev["text"] = (prev["text"].rstrip() + joiner + b["text"].lstrip()).strip()
                # 扩展bbox下边界
                pb = prev.get("bbox") or [0, 0, 0, 0]
                bb = b.get("bbox") or [0, 0, 0, 0]
                prev["bbox"] = [min(pb[0], bb[0]), min(pb[1], bb[1]), max(pb[2], bb[2]), max(pb[3], bb[3])]
            else:
                merged_blocks.append(b)

        blocks_sorted = merged_blocks
        for b in blocks_sorted:
            if b["type"] == "text":
                txt = b["text"]
                if not txt:
                    continue
                # 页眉/页脚清洗（使用归一化匹配，包含子串判断）
                head_key = normalize_text(txt.strip()[:160])
                if head_key in header_set or head_key in footer_set:
                    continue
                # 若该段是页眉/页脚归一化key的子串（或反过来）也跳过
                def _is_sub_in_sets(k: str, keys: set) -> bool:
                    if not k:
                        return False
                    for kk in keys:
                        if len(k) >= 8 and (k in kk or kk in k):
                            return True
                    return False
                if _is_sub_in_sets(head_key, header_set) or _is_sub_in_sets(head_key, footer_set):
                    continue
                # 近重复抑制：对较长文本做规范化去重
                norm = normalize_text(txt)
                if len(txt) >= 20:
                    if norm in seen_norm:
                        continue
                    # 与最近窗口内条目做相似度近重复判定
                    skip = False
                    for prev in recent_norm:
                        if len(prev) < 15:
                            continue
                        if difflib.SequenceMatcher(None, norm[:200], prev[:200]).ratio() >= 0.94:
                            skip = True
                            break
                    if skip:
                        continue
                    seen_norm.add(norm)
                    recent_norm.append(norm)
                h = infer_heading(b.get("max_size", 0.0), thresholds, b.get("font", ""))
                # 合并过短行，避免标题被切碎
                if len(txt) <= 2 and md_lines and md_lines[-1] and not md_lines[-1].startswith('#'):
                    md_lines[-1] = md_lines[-1].rstrip() + txt
                    continue
                if h == 1:
                    md_lines.append(f"# {txt}")
                    report["headings"]["h1"] += 1
                elif h == 2:
                    md_lines.append(f"## {txt}")
                    report["headings"]["h2"] += 1
                elif h == 3:
                    md_lines.append(f"### {txt}")
                    report["headings"]["h3"] += 1
                else:
                    md_lines.append(txt)
                report["text_blocks"] += 1
                bbox = b.get("bbox") or []
                if bbox:
                    md_lines.append(f"\n<sub>pos: page={pno}, bbox={','.join([str(round(x,1)) for x in bbox])}</sub>")
            else:
                rel_path = export_image(page, image_dir, b.get("bbox") or [], img_idx)
                if rel_path:
                    md_lines.append(f"\n![page {pno} image]({rel_path})\n")
                    report["images"] += 1
                img_idx += 1

        # 在页面尾部追加该页的表格（如有）
        page_tables = tables_map.get(pno) or []
        if page_tables:
            table_seen = set()
            table_lines: List[str] = []
            any_output_for_page = False
            for table in page_tables:
                if not table:
                    continue
                # 如果表格看起来是“逐字切分”的字符网格，则按段落输出，避免误判为表格
                if _looks_like_char_grid(table):
                    para = _char_grid_to_paragraph(table)
                    if para:
                        # 对字符网格段落做全局去重，避免与页面正文/页眉脚重复
                        para_norm = normalize_text(para)
                        # 命中页眉/页脚（包含子串）直接跳过
                        def _contains_in_sets(k: str, keys: set) -> bool:
                            for kk in keys:
                                if len(kk) >= 8 and (kk in k or k in kk):
                                    return True
                            return False
                        if _contains_in_sets(para_norm, header_set) or _contains_in_sets(para_norm, footer_set):
                            continue
                        if len(para) >= 40:
                            # 已出现或高度相似则跳过
                            skip = False
                            if para_norm in seen_norm:
                                skip = True
                            else:
                                for prev in recent_norm:
                                    if len(prev) < 15:
                                        continue
                                    if difflib.SequenceMatcher(None, para_norm[:300], prev[:300]).ratio() >= 0.94:
                                        skip = True
                                        break
                            if skip:
                                continue
                            seen_norm.add(para_norm)
                            recent_norm.append(para_norm)
                        table_lines.append(para)
                        any_output_for_page = True
                    continue
                # 生成 Markdown 表格
                header = table[0]
                sig = normalize_text("|".join(header[:6])) if header else ""
                if sig and sig in table_seen:
                    continue
                if sig:
                    table_seen.add(sig)
                table_lines.append("| " + " | ".join([c or '' for c in header]) + " |")
                table_lines.append("| " + " | ".join(["---" for _ in header]) + " |")
                for row in table[1:]:
                    table_lines.append("| " + " | ".join([c or '' for c in row]) + " |")
                table_lines.append("")
                report["tables"] += 1
                any_output_for_page = True
            if any_output_for_page and table_lines:
                md_lines.append(f"\n**表格（第 {pno} 页）**\n")
                md_lines.extend(table_lines)

    md_path = os.path.join(out_dir, "document.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    with open(os.path.join(out_dir, "layout_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {"md_path": md_path, "report": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--out", required=True, help="输出目录")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"❌ PDF不存在: {args.pdf}")

    doc = fitz.open(args.pdf)
    res = make_markdown(doc, args.out, args.pdf)
    print(json.dumps({"ok": True, **res}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
