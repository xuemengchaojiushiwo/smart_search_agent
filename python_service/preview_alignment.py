#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将对齐结果（aligned_positions.json）可视化：
- 为每页生成一张PNG，绘制每个段落的 bbox_union（绿=高置信，黄=中，红=低/无）
- 输出质量报告（JSON/CSV）：覆盖率、相似度、未定位段落等

用法：
  python preview_alignment.py --pdf path/to.pdf --aligned out_dir/aligned_positions.json --out out_dir/preview
"""
import os
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple
import difflib
import fitz  # PyMuPDF


def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)


def load_aligned(path:str) -> List[Dict[str,Any]]:
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data.get('items', data)


def page_text(doc: fitz.Document, pno:int) -> str:
    return doc[pno-1].get_text() if 1 <= pno <= len(doc) else ""


def sim_ratio(a:str,b:str) -> float:
    try:
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def draw_preview(pdf_path:str, aligned_path:str, out_dir:str):
    ensure_dir(out_dir)
    doc = fitz.open(pdf_path)
    items = load_aligned(aligned_path)

    # 预先分组到每页
    by_page: Dict[int, List[Dict[str,Any]]] = {}
    for it in items:
        by_page.setdefault(int(it.get('page_num',-1)), []).append(it)

    summary_rows = []
    ok_cnt = mid_cnt = bad_cnt = 0

    for pno in range(1, len(doc)+1):
        page = doc[pno-1]
        pix = page.get_pixmap(dpi=180)
        img = fitz.Pixmap(pix, 0) if pix.alpha else pix
        # 使用 shape 在矢量层画框
        shape = page.new_shape()

        page_items = by_page.get(pno, [])
        ptxt = page.get_text() or ""
        for it in page_items:
            s = int(it.get('char_start',-1)); e = int(it.get('char_end',-1))
            bbox = it.get('bbox_union') or []
            para_text = it.get('text','')
            found_sub = ptxt[s:e] if 0<=s<e<=len(ptxt) else ''
            ratio = sim_ratio(para_text[:200].lower(), found_sub.lower()) if found_sub else 0.0
            # 颜色阈值
            # 优先依据 bbox 是否存在作为“页级已定位”判断
            if bbox and ratio >= 0.8:
                color = (0, 1, 0)  # 绿
                ok_cnt += 1
            elif bbox and (ratio >= 0.5 or (s < 0 or e < 0)):
                color = (1, 1, 0)  # 黄
                mid_cnt += 1
            else:
                color = (1, 0, 0)  # 红
                bad_cnt += 1
            if bbox and len(bbox)==4:
                rect = fitz.Rect(*bbox)
                shape.draw_rect(rect)
                shape.finish(color=color, fill=None, width=1.5)
                # 标注索引
                shape.insert_text(rect.tl + fitz.Point(1, -8), str(it.get('para_index', 0)), color=color, fontsize=6)
            summary_rows.append({
                'page': pno,
                'para_index': it.get('para_index',-1),
                'ratio': round(ratio,3),
                'has_bbox': 1 if bbox else 0,
                'char_start': s,
                'char_end': e
            })
        # 将形状绘制到页面，导出PNG
        shape.commit()
        out_png = os.path.join(out_dir, f'page_{pno:03d}.png')
        page.get_pixmap(dpi=180).save(out_png)

    # 汇总报告
    report = {
        'pages': len(doc),
        'items': len(items),
        'ok': ok_cnt,
        'mid': mid_cnt,
        'bad': bad_cnt,
        'ok_rate': round(ok_cnt/max(1,len(items)), 3)
    }
    with open(os.path.join(out_dir,'report.json'),'w',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    # CSV
    with open(os.path.join(out_dir,'detail.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['page','para_index','ratio','has_bbox','char_start','char_end'])
        w.writeheader(); w.writerows(summary_rows)

    print(json.dumps({'ok':True, **report, 'out_dir': out_dir}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--aligned', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    draw_preview(args.pdf, args.aligned, args.out)


if __name__ == '__main__':
    main()
