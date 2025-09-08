#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

# ES配置
ES_HOST = "localhost"
ES_PORT = 9200
INDEX_NAME = "knowledge_base_new"

def test_es_search_directly():
    """直接测试ES搜索，绕过Java端逻辑"""
    try:
        print("=== 直接测试ES搜索 ===")
        
        # 1. 测试搜索"测试1"
        print("\n1. 搜索'测试1'")
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "测试1",
                                "fields": ["title^3", "content^1.5", "tags^2", "attachment_names^1.8", "author^1"]
                            }
                        }
                    ],
                    "filter": [
                        {"exists": {"field": "id"}},
                        {"exists": {"field": "title"}}
                    ]
                }
            },
            "size": 10,
            "_source": ["id", "title", "content", "workspaces"]
        }
        
        url = f"http://{ES_HOST}:{ES_PORT}/{INDEX_NAME}/_search"
        response = requests.post(url, json=search_query, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            total = result['hits']['total']['value']
            hits = result['hits']['hits']
            print(f"  ES直接搜索找到 {total} 个文档")
            
            if hits:
                print("  搜索结果:")
                for i, hit in enumerate(hits[:3]):
                    source = hit['_source']
                    score = hit['_score']
                    print(f"    结果 {i+1} (分数: {score}):")
                    print(f"      ID: {source.get('id')}")
                    print(f"      Title: {source.get('title')}")
                    print(f"      Workspaces: {source.get('workspaces', [])}")
            else:
                print("  ❌ ES直接搜索也没有结果")
        else:
            print(f"  ❌ ES搜索失败: {response.status_code}")
        
        # 2. 测试搜索"测试"（不带数字）
        print("\n2. 搜索'测试'")
        search_query["query"]["bool"]["must"][0]["multi_match"]["query"] = "测试"
        
        response = requests.post(url, json=search_query, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            total = result['hits']['total']['value']
            hits = result['hits']['hits']
            print(f"  ES直接搜索'测试'找到 {total} 个文档")
            
            if hits:
                print("  搜索结果:")
                for i, hit in enumerate(hits[:3]):
                    source = hit['_source']
                    score = hit['_score']
                    print(f"    结果 {i+1} (分数: {score}):")
                    print(f"      ID: {source.get('id')}")
                    print(f"      Title: {source.get('title')}")
                    print(f"      Workspaces: {source.get('workspaces', [])}")
            else:
                print("  ❌ ES直接搜索'测试'也没有结果")
        else:
            print(f"  ❌ ES搜索'测试'失败: {response.status_code}")
        
        # 3. 测试最简单的搜索
        print("\n3. 最简单的搜索（只过滤字段）")
        simple_query = {
            "query": {
                "bool": {
                    "filter": [
                        {"exists": {"field": "id"}},
                        {"exists": {"field": "title"}}
                    ]
                }
            },
            "size": 5,
            "_source": ["id", "title", "workspaces"]
        }
        
        response = requests.post(url, json=simple_query, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            total = result['hits']['total']['value']
            hits = result['hits']['hits']
            print(f"  简单过滤查询找到 {total} 个文档")
            
            if hits:
                print("  前3个结果:")
                for i, hit in enumerate(hits[:3]):
                    source = hit['_source']
                    print(f"    结果 {i+1}: ID={source.get('id')}, Title={source.get('title')}")
            else:
                print("  ❌ 简单过滤查询也没有结果")
        else:
            print(f"  ❌ 简单过滤查询失败: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ 直接测试ES搜索异常: {e}")
        return False

def check_es_index_status():
    """检查ES索引状态"""
    try:
        print("\n=== 检查ES索引状态 ===")
        
        # 检查索引基本信息
        url = f"http://{ES_HOST}:{ES_PORT}/{INDEX_NAME}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            info = response.json()
            print(f"索引名称: {INDEX_NAME}")
            print(f"文档总数: {info.get('_count', 'N/A')}")
            print(f"索引状态: {info.get('_status', 'N/A')}")
        else:
            print(f"❌ 获取索引信息失败: {response.status_code}")
        
        # 检查索引统计
        stats_url = f"http://{ES_HOST}:{ES_PORT}/{INDEX_NAME}/_stats"
        response = requests.get(stats_url, timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            indices = stats.get('indices', {})
            if INDEX_NAME in indices:
                index_stats = indices[INDEX_NAME]
                total = index_stats.get('total', {})
                docs = total.get('docs', {})
                print(f"索引统计 - 文档数: {docs.get('count', 'N/A')}")
            else:
                print("❌ 索引统计中找不到目标索引")
        else:
            print(f"❌ 获取索引统计失败: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ 检查ES索引状态异常: {e}")
        return False

def main():
    print("=== ES搜索调试 ===")
    
    # 1. 检查ES索引状态
    if not check_es_index_status():
        return
    
    # 2. 直接测试ES搜索
    if not test_es_search_directly():
        return
    
    print("\n🎉 调试完成！")

if __name__ == "__main__":
    main()
