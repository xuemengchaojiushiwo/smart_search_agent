#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES 向量数据巡检与相似度查询脚本

功能：
- 统计索引中文档总数/含embedding的文档数
- 抽样打印若干含embedding的文档关键信息
- 对查询词进行embedding，执行向量相似度Top3检索并打印命中文档内容与溯源信息

用法：
  python scripts/es_inspect_embeddings.py
"""

import sys
import json
import requests
from elasticsearch import Elasticsearch

# 复用Python服务配置（以相对路径导入）
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg_path = ROOT / "python_service" / "config.py"
spec = importlib.util.spec_from_file_location("ps_config", str(cfg_path))
ps_config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ps_config)

ES_CONFIG = getattr(ps_config, "ES_CONFIG")
GEEKAI_EMBEDDING_URL = getattr(ps_config, "GEEKAI_EMBEDDING_URL")
DEFAULT_EMBEDDING_MODEL = getattr(ps_config, "DEFAULT_EMBEDDING_MODEL")
GEEKAI_API_KEY = getattr(ps_config, "GEEKAI_API_KEY", "")


def build_es_client() -> Elasticsearch:
    es = Elasticsearch(
        [f"http://{ES_CONFIG['host']}:{ES_CONFIG['port']}"] ,
        basic_auth=(ES_CONFIG["username"], ES_CONFIG["password"]) if ES_CONFIG.get("username") else None,
        verify_certs=ES_CONFIG.get("verify_certs", False),
    )
    return es


def get_embedding(text: str):
    headers = {
        "Authorization": f"Bearer {GEEKAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": DEFAULT_EMBEDDING_MODEL,
        "input": [text],
        "intent": "search_document",
    }
    resp = requests.post(GEEKAI_EMBEDDING_URL, headers=headers, json=data, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"获取embedding失败: {resp.status_code} - {resp.text}")
    emb = resp.json().get("data", [{}])[0].get("embedding")
    if not emb:
        raise RuntimeError("embedding返回为空")
    return emb


def pretty_trunc(s: str, max_len: int = 220) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ")
    return s[:max_len] + ("..." if len(s) > max_len else "")


def main():
    es = build_es_client()
    index = ES_CONFIG["index"]

    # 统计
    total = es.count(index=index).get("count", 0)
    with_emb = es.count(index=index, query={"exists": {"field": "embedding"}}).get("count", 0)
    without_emb = total - with_emb
    print(f"索引: {index}")
    print(f"总文档数: {total}")
    print(f"含embedding文档数: {with_emb}")
    print(f"不含embedding文档数: {without_emb}")

    # 抽样查看含embedding的文档
    print("\n=== 抽样查看含embedding的文档(最多5条) ===")
    sample = es.search(
        index=index,
        size=5,
        query={"exists": {"field": "embedding"}},
        _source=[
            "knowledge_id","knowledge_name","source_file","page_num","chunk_index","chunk_type",
            "char_start","char_end","bbox_union","content","embedding"
        ],
    )
    for i, hit in enumerate(sample.get("hits", {}).get("hits", []), start=1):
        src = hit.get("_source", {})
        emb = src.get("embedding") or []
        print(f"#{i} _id={hit.get('_id')} score={hit.get('_score')}")
        print(f"  文件={src.get('source_file')} 页={src.get('page_num')} 块序={src.get('chunk_index')} 类型={src.get('chunk_type')}")
        print(f"  内容={pretty_trunc(src.get('content'))}")
        print(f"  embedding维度={len(emb)} 范围示例={emb[:3] if emb else []}")

    # 语义查询(向量Top3)
    query_text = "安联基金总值"
    print(f"\n=== 向量相似度查询 Top3: '{query_text}' ===")
    try:
        qvec = get_embedding(query_text)
    except Exception as e:
        print(f"获取查询embedding失败: {e}")
        sys.exit(2)

    body = {
        "size": 3,
        "query": {
            "script_score": {
                "query": {"exists": {"field": "embedding"}},
                "script": {
                    "source": "cosineSimilarity(params.qvec, 'embedding')",
                    "params": {"qvec": qvec},
                },
            }
        },
        "_source": [
            "knowledge_name","source_file","page_num","chunk_index","content","bbox_union","char_start","char_end"
        ],
    }
    res = es.search(index=index, body=body)
    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        print("未命中任何文档。")
    for i, hit in enumerate(hits, start=1):
        src = hit.get("_source", {})
        print(f"Top{i} score={hit.get('_score')}")
        print(f"  文件={src.get('source_file')} 页={src.get('page_num')} 块序={src.get('chunk_index')}")
        print(f"  内容={pretty_trunc(src.get('content'), 400)}")
        print(f"  位置: bbox={src.get('bbox_union')} chars=({src.get('char_start')}, {src.get('char_end')})")


if __name__ == "__main__":
    main()


