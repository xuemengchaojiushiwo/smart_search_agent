#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将包含 <sub>pos: page=..., bbox=x0,y0,x1,y1</sub> 的 Markdown 转为 aligned_positions.json。
规则：
- 逐行扫描：遇到非空文本行，若下一行是 <sub>pos: ...</sub>，则作为一条 item
- 输出字段：text, page_num, bbox_union, bboxes([bbox]), char_start=-1, char_end=-1, para_index
"""
import re
import os
import json
import argparse
from typing import List, Dict, Any


POS_RE = re.compile(r"<sub>\s*pos:\s*page\s*=\s*(\d+)\s*,\s*bbox\s*=\s*([\d\.,-]+)\s*</sub>", re.IGNORECASE)


def parse_md_with_pos(md_input: str) -> List[Dict[str, Any]]:
    """
    解析包含 <sub>pos: page=..., bbox=x0,y0,x1,y1</sub> 的 Markdown
    
    Args:
        md_input: 可以是文件路径或markdown文本内容
    """
    # 判断输入是文件路径还是文本内容
    if os.path.exists(md_input):
        # 输入是文件路径
        with open(md_input, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip("\n") for ln in f]
    else:
        # 输入是文本内容，按行分割
        lines = md_input.split('\n')
        lines = [ln.rstrip("\n") for ln in lines]

    items: List[Dict[str, Any]] = []
    i = 0
    para_index = 0
    
    # 先找到所有的位置标签
    pos_tags = []
    for line_idx, line in enumerate(lines):
        if line.strip().startswith('<sub>pos:'):
            m = POS_RE.search(line.strip())
            if m:
                page_num = int(m.group(1))
                bbox_str = m.group(2)
                bbox_vals = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]
                bbox = bbox_vals[:4] if len(bbox_vals) >= 4 else []
                pos_tags.append({
                    'line_idx': line_idx,
                    'page_num': page_num,
                    'bbox': bbox
                })
    
    # 为每个位置标签找到对应的文本
    for pos_tag in pos_tags:
        line_idx = pos_tag['line_idx']
        page_num = pos_tag['page_num']
        bbox = pos_tag['bbox']
        
        # 向前查找最近的文本行（最多前看3行）
        text_line = ""
        for i in range(max(0, line_idx - 3), line_idx):
            if lines[i].strip() and not lines[i].strip().startswith('<sub>pos:'):
                text_line = lines[i].strip()
                break
        
        if text_line:
            items.append({
                'para_index': para_index,
                'text': text_line,
                'page_num': page_num,
                'bbox_union': bbox,
                'bboxes': [bbox] if bbox else [],
                'char_start': -1,
                'char_end': -1,
            })
            para_index += 1
    
    return items


def save_aligned(items: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'items': items}, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', required=True, help='带 <sub>pos: ...</sub> 的 Markdown 路径')
    ap.add_argument('--out', required=True, help='输出 aligned_positions.json 路径')
    args = ap.parse_args()
    items = parse_md_with_pos(args.md)
    save_aligned(items, args.out)
    print(json.dumps({'ok': True, 'count': len(items), 'out': args.out}, ensure_ascii=False))


if __name__ == '__main__':
    main()


