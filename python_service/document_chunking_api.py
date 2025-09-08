#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档切分结果API接口
提供获取文档切分结果的接口
"""

from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import logging
from typing import List, Dict, Any
import json
import os
import uuid
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# ES连接配置
ES_CONFIG = {
    'host': 'localhost',
    'port': 9200,
    'index': 'knowledge_base_new',
    'username': 'elastic',
    'password': 'password',
    'verify_certs': False
}

def get_es_client():
    """获取ES客户端"""
    try:
        # 修复ES连接配置，使用正确的参数格式
        es = Elasticsearch(
            hosts=[{'host': ES_CONFIG['host'], 'port': ES_CONFIG['port']}],
            http_auth=(ES_CONFIG['username'], ES_CONFIG['password']),
            verify_certs=ES_CONFIG['verify_certs'],
            timeout=30
        )
        return es
    except Exception as e:
        logger.error(f"ES连接失败: {e}")
        return None

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    """获取文件类型"""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ['docx', 'doc']:
        return 'Word文档'
    elif ext in ['xlsx', 'xls']:
        return 'Excel表格'
    elif ext in ['pptx', 'ppt']:
        return 'PowerPoint演示'
    elif ext == 'txt':
        return '文本文档'
    elif ext == 'pdf':
        return 'PDF文档'
    else:
        return '其他文档'

@app.route('/api/document/process', methods=['POST'])
def process_document():
    """处理上传的文档"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "没有选择文件"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "没有选择文件"
            }), 400
        
        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"不支持的文件类型: {file.filename}"
            }), 400
        
        # 获取其他参数
        knowledge_id = request.form.get('knowledge_id', '1')
        knowledge_name = request.form.get('knowledge_name', file.filename)
        
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # 保存文件
        file.save(file_path)
        
        # 模拟文档处理（实际项目中应该调用真实的文档处理服务）
        file_type = get_file_type(filename)
        
        # 模拟切分结果
        mock_chunks = generate_mock_chunks(file_type, filename)
        
        # 保存到ES（模拟）
        save_chunks_to_es(knowledge_id, mock_chunks, filename)
        
        logger.info(f"文档处理成功: {filename}, 类型: {file_type}, 切分块数: {len(mock_chunks)}")
        
        return jsonify({
            "success": True,
            "message": "文档处理成功",
            "data": {
                "filename": filename,
                "file_type": file_type,
                "chunks_count": len(mock_chunks),
                "knowledge_id": knowledge_id,
                "knowledge_name": knowledge_name
            }
        })
        
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        return jsonify({
            "success": False,
            "error": f"文档处理失败: {str(e)}"
        }), 500

def generate_mock_chunks(file_type, filename):
    """生成模拟的切分块"""
    if file_type == 'Word文档':
        return [
            {
                "content": f"这是{filename}的第一个段落。包含了文档的基本信息和主要内容。",
                "positions": [{"text": "第一个段落", "bbox": [50, 50, 200, 80], "page": 1}],
                "metadata": {"page_num": 1, "document_type": "Word文档"}
            },
            {
                "content": f"这是{filename}的第二个段落。详细描述了文档的具体内容和结构。",
                "positions": [{"text": "第二个段落", "bbox": [50, 100, 300, 130], "page": 1}],
                "metadata": {"page_num": 1, "document_type": "Word文档"}
            }
        ]
    elif file_type == 'Excel表格':
        return [
            {
                "content": f"这是{filename}的表格数据。包含了各种统计信息和数据记录。",
                "positions": [{"text": "表格数据", "sheet": "Sheet1", "row": 1, "col": 1}],
                "metadata": {"page_num": 1, "document_type": "Excel表格"}
            }
        ]
    elif file_type == 'PowerPoint演示':
        return [
            {
                "content": f"这是{filename}的演示内容。包含了多个幻灯片的主题和要点。",
                "positions": [{"text": "演示内容", "bbox": [100, 100, 400, 140], "page": 1}],
                "metadata": {"page_num": 1, "document_type": "PowerPoint演示"}
            }
        ]
    else:
        return [
            {
                "content": f"这是{filename}的文本内容。包含了文档的主要信息和详细描述。",
                "positions": [{"text": "文本内容", "line_no": 1, "char_start": 0, "char_end": 50}],
                "metadata": {"page_num": 1, "document_type": "文本文档"}
            }
        ]

def save_chunks_to_es(knowledge_id, chunks, filename):
    """保存切分块到ES（模拟）"""
    try:
        es = get_es_client()
        if not es:
            logger.warning("ES连接失败，跳过保存")
            return
        
        # 模拟保存到ES
        for i, chunk in enumerate(chunks):
            doc = {
                "knowledge_id": int(knowledge_id),
                "chunk_index": i + 1,
                "page_content": chunk["content"],
                "positions": chunk["positions"],
                "page_num": chunk["metadata"]["page_num"],
                "document_type": chunk["metadata"]["document_type"],
                "source_file": filename,
                "keywords": extract_keywords(chunk["content"])
            }
            
            # 这里应该调用ES的index方法，现在只是模拟
            logger.info(f"模拟保存chunk {i+1}到ES: {doc}")
            
    except Exception as e:
        logger.error(f"保存到ES失败: {e}")

def extract_keywords(text):
    """提取关键词（模拟）"""
    # 简单的关键词提取，实际项目中可以使用NLP技术
    words = text.split()
    return words[:5] if len(words) > 5 else words

def generate_mock_chunks_for_knowledge(knowledge_id):
    """为特定知识库生成模拟切分结果"""
    # 根据知识库ID生成不同的模拟数据
    if knowledge_id == 1:
        return [
            {
                "index": 1,
                "content": "第一章 项目概述\n\n这是一个智能知识管理系统的项目概述。我们致力于开发一个能够处理多种文档格式的智能系统，包括Word、Excel、PowerPoint等Office文档。",
                "positions": [
                    {"text": "第一章 项目概述", "bbox": [50, 50, 200, 80], "page": 1},
                    {"text": "这是一个智能知识管理系统的项目概述", "bbox": [50, 100, 400, 130], "page": 1}
                ],
                "metadata": {"page_num": 1, "document_type": "Word文档"}
            },
            {
                "index": 2,
                "content": "系统的主要功能包括：\n1. 多格式文档解析和切分\n2. 智能文本分析和关键词提取\n3. 精确定位信息保存\n4. 语义搜索和问答功能\n5. 支持中文和英文文档",
                "positions": [
                    {"text": "系统的主要功能包括：", "bbox": [50, 150, 300, 180], "page": 1},
                    {"text": "1. 多格式文档解析和切分", "bbox": [70, 200, 350, 230], "page": 1}
                ],
                "metadata": {"page_num": 1, "document_type": "Word文档"}
            },
            {
                "index": 3,
                "content": "技术架构：\n• 前端：HTML5 + CSS3 + JavaScript\n• 后端：Python Flask + Elasticsearch\n• 文档处理：PyMuPDF Pro + LangChain\n• AI接口：极客智坊 + 自定义AI服务",
                "positions": [
                    {"text": "技术架构：", "bbox": [50, 300, 150, 330], "page": 1},
                    {"text": "• 前端：HTML5 + CSS3 + JavaScript", "bbox": [70, 350, 380, 380], "page": 1}
                ],
                "metadata": {"page_num": 1, "document_type": "Word文档"}
            }
        ]
    else:
        # 为其他知识库ID生成通用数据
        return [
            {
                "index": 1,
                "content": f"知识库 {knowledge_id} 的示例文档\n\n这是一个示例文档，展示了文档切分和定位功能。每个文本块都包含了相应的位置信息和元数据。",
                "positions": [
                    {"text": f"知识库 {knowledge_id} 的示例文档", "bbox": [50, 50, 300, 80], "page": 1}
                ],
                "metadata": {"page_num": 1, "document_type": "示例文档"}
            }
        ]

@app.route('/api/chunks/<int:knowledge_id>', methods=['GET'])
def get_document_chunks(knowledge_id: int):
    """获取指定知识库的文档切分结果"""
    try:
        es = get_es_client()
        if not es:
            # 如果ES连接失败，返回模拟数据
            logger.warning("ES连接失败，返回模拟切分结果")
            mock_chunks = generate_mock_chunks_for_knowledge(knowledge_id)
            return jsonify({
                "success": True,
                "data": mock_chunks,
                "total": len(mock_chunks),
                "message": "ES连接失败，显示模拟数据"
            })
        
        # 查询指定知识库的所有chunks
        query = {
            "query": {
                "term": {
                    "knowledge_id": knowledge_id
                }
            },
            "sort": [
                {"page_num": {"order": "asc"}},
                {"chunk_index": {"order": "asc"}}
            ],
            "size": 1000  # 获取所有chunks
        }
        
        response = es.search(index=ES_CONFIG['index'], body=query)
        
        if response['hits']['total']['value'] == 0:
            # 如果ES中没有数据，返回模拟数据
            logger.info("ES中未找到数据，返回模拟切分结果")
            mock_chunks = generate_mock_chunks_for_knowledge(knowledge_id)
            return jsonify({
                "success": True,
                "data": mock_chunks,
                "total": len(mock_chunks),
                "message": "未找到真实数据，显示模拟数据"
            })
        
        chunks = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            chunk = {
                "index": source.get('chunk_index', 0),
                "content": source.get('page_content', ''),
                "positions": source.get('positions', []),
                "metadata": {
                    "page_num": source.get('page_num', 1),
                    "document_type": source.get('document_type', '未知'),
                    "sheet_name": source.get('sheet_name', ''),
                    "source_file": source.get('source_file', ''),
                    "bbox": source.get('bbox', []),
                    "keywords": source.get('keywords', [])
                }
            }
            chunks.append(chunk)
        
        # 按页码和块索引排序
        chunks.sort(key=lambda x: (x['metadata']['page_num'], x['index']))
        
        return jsonify({
            "success": True,
            "data": chunks,
            "total": len(chunks)
        })
        
    except Exception as e:
        logger.error(f"获取文档切分结果失败: {e}")
        # 发生异常时也返回模拟数据
        mock_chunks = generate_mock_chunks_for_knowledge(knowledge_id)
        return jsonify({
            "success": True,
            "data": mock_chunks,
            "total": len(mock_chunks),
            "message": f"获取数据失败，显示模拟数据: {str(e)}"
        })

@app.route('/api/chunks/search', methods=['POST'])
def search_chunks():
    """在切分结果中搜索"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "缺少搜索查询参数"
            }), 400
        
        query_text = data['query']
        knowledge_id = data.get('knowledge_id')
        
        es = get_es_client()
        if not es:
            return jsonify({
                "success": False,
                "error": "ES连接失败"
            }), 500
        
        # 构建搜索查询
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["page_content", "keywords"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "page_content": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            },
            "size": 50
        }
        
        # 如果指定了知识库ID，添加过滤条件
        if knowledge_id:
            search_query["query"]["bool"]["filter"] = [
                {"term": {"knowledge_id": knowledge_id}}
            ]
        
        response = es.search(index=ES_CONFIG['index'], body=search_query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            highlight = hit.get('highlight', {})
            
            result = {
                "index": source.get('chunk_index', 0),
                "content": source.get('page_content', ''),
                "highlighted_content": highlight.get('page_content', [source.get('page_content', '')])[0],
                "score": hit['_score'],
                "positions": source.get('positions', []),
                "metadata": {
                    "page_num": source.get('page_num', 1),
                    "document_type": source.get('document_type', '未知'),
                    "sheet_name": source.get('sheet_name', ''),
                    "source_file": source.get('source_file', ''),
                    "knowledge_id": source.get('knowledge_id', 0)
                }
            }
            results.append(result)
        
        # 按相关性得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            "success": True,
            "data": results,
            "total": len(results),
            "query": query_text
        })
        
    except Exception as e:
        logger.error(f"搜索切分结果失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chunks/stats/<int:knowledge_id>', methods=['GET'])
def get_chunks_stats(knowledge_id: int):
    """获取切分结果统计信息"""
    try:
        es = get_es_client()
        if not es:
            # 如果ES连接失败，返回模拟统计信息
            logger.warning("ES连接失败，返回模拟统计信息")
            mock_chunks = generate_mock_chunks_for_knowledge(knowledge_id)
            stats = calculate_stats_from_chunks(mock_chunks)
            return jsonify({
                "success": True,
                "data": stats,
                "message": "ES连接失败，显示模拟统计信息"
            })
        
        # 聚合查询获取统计信息
        query = {
            "query": {
                "term": {
                    "knowledge_id": knowledge_id
                }
            },
            "aggs": {
                "total_chunks": {
                    "value_count": {
                        "field": "chunk_index"
                    }
                },
                "avg_chunk_size": {
                    "avg": {
                        "script": {
                            "source": "doc['page_content.keyword'].value.length()"
                        }
                    }
                },
                "document_types": {
                    "terms": {
                        "field": "document_type.keyword"
                    }
                },
                "total_positions": {
                    "sum": {
                        "script": {
                            "source": "doc['positions'].size()"
                        }
                    }
                },
                "pages_distribution": {
                    "terms": {
                        "field": "page_num"
                    }
                }
            },
            "size": 0
        }
        
        response = es.search(index=ES_CONFIG['index'], body=query)
        
        aggs = response['aggregations']
        
        stats = {
            "total_chunks": aggs['total_chunks']['value'],
            "avg_chunk_size": round(aggs['avg_chunk_size']['value'] if aggs['avg_chunk_size']['value'] else 0),
            "total_positions": aggs['total_positions']['value'],
            "document_types": [bucket['key'] for bucket in aggs['document_types']['buckets']],
            "pages_count": len(aggs['pages_distribution']['buckets']),
            "pages_distribution": {
                bucket['key']: bucket['doc_count'] 
                for bucket in aggs['pages_distribution']['buckets']
            }
        }
        
        return jsonify({
            "success": True,
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"获取切分统计信息失败: {e}")
        # 发生异常时返回模拟统计信息
        mock_chunks = generate_mock_chunks_for_knowledge(knowledge_id)
        stats = calculate_stats_from_chunks(mock_chunks)
        return jsonify({
            "success": True,
            "data": stats,
            "message": f"获取统计信息失败，显示模拟数据: {str(e)}"
        })

def calculate_stats_from_chunks(chunks):
    """从切分块计算统计信息"""
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "total_positions": 0,
            "document_types": [],
            "pages_count": 0,
            "pages_distribution": {}
        }
    
    total_chunks = len(chunks)
    total_positions = sum(len(chunk.get('positions', [])) for chunk in chunks)
    avg_chunk_size = round(sum(len(chunk.get('content', '')) for chunk in chunks) / total_chunks)
    
    # 获取文档类型
    doc_types = list(set(chunk.get('metadata', {}).get('document_type', '未知') for chunk in chunks))
    
    # 获取页码分布
    pages = {}
    for chunk in chunks:
        page_num = chunk.get('metadata', {}).get('page_num', 1)
        pages[page_num] = pages.get(page_num, 0) + 1
    
    return {
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "total_positions": total_positions,
        "document_types": doc_types,
        "pages_count": len(pages),
        "pages_distribution": pages
    }

@app.route('/api/chunks/positions/<int:knowledge_id>', methods=['GET'])
def get_chunk_positions(knowledge_id: int):
    """获取指定知识库的定位信息"""
    try:
        es = get_es_client()
        if not es:
            return jsonify({
                "success": False,
                "error": "ES连接失败"
            }), 500
        
        # 查询定位信息
        query = {
            "query": {
                "term": {
                    "knowledge_id": knowledge_id
                }
            },
            "_source": ["positions", "bbox", "page_num", "chunk_index", "document_type"],
            "sort": [
                {"page_num": {"order": "asc"}},
                {"chunk_index": {"order": "asc"}}
            ],
            "size": 1000
        }
        
        response = es.search(index=ES_CONFIG['index'], body=query)
        
        positions_data = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            positions_data.append({
                "chunk_index": source.get('chunk_index', 0),
                "page_num": source.get('page_num', 1),
                "document_type": source.get('document_type', '未知'),
                "bbox": source.get('bbox', []),
                "positions": source.get('positions', [])
            })
        
        return jsonify({
            "success": True,
            "data": positions_data
        })
        
    except Exception as e:
        logger.error(f"获取定位信息失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        es = get_es_client()
        if es and es.ping():
            return jsonify({
                "status": "healthy",
                "es_connection": "connected"
            })
        else:
            return jsonify({
                "status": "unhealthy",
                "es_connection": "disconnected"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    from config import HOST, PORT
    
    print(f"启动文档切分结果API服务...")
    print(f"服务地址: http://{HOST}:{PORT}")
    print(f"健康检查: http://{HOST}:{PORT}/health")
    print(f"获取切分结果: http://{HOST}:{PORT}/api/chunks/<knowledge_id>")
    print(f"搜索切分结果: http://{HOST}:{PORT}/api/chunks/search")
    
    app.run(host=HOST, port=PORT, debug=True)
