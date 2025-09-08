#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API管理接口，提供动态切换API的功能
"""

from flask import Flask, request, jsonify
from ai_client_manager import get_ai_manager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/status', methods=['GET'])
def get_api_status():
    """获取当前API状态"""
    try:
        manager = get_ai_manager()
        api_info = manager.get_api_info()
        
        return jsonify({
            "success": True,
            "data": api_info
        })
    except Exception as e:
        logger.error(f"获取API状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/switch', methods=['POST'])
def switch_api():
    """切换API"""
    try:
        data = request.get_json()
        if not data or 'api_name' not in data:
            return jsonify({
                "success": False,
                "error": "缺少api_name参数"
            }), 400
        
        api_name = data['api_name']
        manager = get_ai_manager()
        
        # 检查API是否可用
        if not manager.is_api_available(api_name):
            return jsonify({
                "success": False,
                "error": f"API {api_name} 不可用"
            }), 400
        
        # 切换API
        if manager.switch_api(api_name):
            return jsonify({
                "success": True,
                "message": f"已成功切换到 {api_name} API",
                "current_api": manager.get_current_api()
            })
        else:
            return jsonify({
                "success": False,
                "error": f"切换到 {api_name} API 失败"
            }), 500
            
    except Exception as e:
        logger.error(f"切换API失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/test', methods=['POST'])
def test_api():
    """测试指定API"""
    try:
        data = request.get_json()
        if not data or 'api_name' not in data:
            return jsonify({
                "success": False,
                "error": "缺少api_name参数"
            }), 400
        
        api_name = data['api_name']
        manager = get_ai_manager()
        
        # 检查API是否可用
        if not manager.is_api_available(api_name):
            return jsonify({
                "success": False,
                "error": f"API {api_name} 不可用"
            }), 400
        
        # 临时切换到指定API进行测试
        original_api = manager.get_current_api()
        if not manager.switch_api(api_name):
            return jsonify({
                "success": False,
                "error": f"切换到 {api_name} API 失败"
            }), 500
        
        try:
            # 测试简单聊天
            test_message = "你好，这是一个API测试"
            response = manager.simple_chat(test_message)
            
            # 测试向量化
            test_texts = ["测试文本"]
            embeddings_result = manager.get_embeddings(test_texts)
            
            # 切换回原API
            manager.switch_api(original_api)
            
            return jsonify({
                "success": True,
                "message": f"API {api_name} 测试成功",
                "chat_test": {
                    "message": test_message,
                    "response": response[:200] + "..." if len(response) > 200 else response
                },
                "embedding_test": {
                    "success": "error" not in embeddings_result,
                    "dimension": len(embeddings_result.get("data", [{}])[0].get("embedding", [])) if "data" in embeddings_result else 0
                }
            })
            
        except Exception as e:
            # 确保切换回原API
            manager.switch_api(original_api)
            raise e
            
    except Exception as e:
        logger.error(f"测试API失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天接口"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "缺少message参数"
            }), 400
        
        message = data['message']
        model = data.get('model')
        manager = get_ai_manager()
        
        response = manager.simple_chat(message, model)
        
        return jsonify({
            "success": True,
            "data": {
                "message": message,
                "response": response,
                "api": manager.get_current_api()
            }
        })
        
    except Exception as e:
        logger.error(f"聊天失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    """向量化接口"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                "success": False,
                "error": "缺少texts参数"
            }), 400
        
        texts = data['texts']
        model = data.get('model')
        manager = get_ai_manager()
        
        result = manager.get_embeddings(texts, model)
        
        if "error" in result:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
        
        return jsonify({
            "success": True,
            "data": result,
            "api": manager.get_current_api()
        })
        
    except Exception as e:
        logger.error(f"向量化失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/rag', methods=['POST'])
def rag_chat():
    """RAG对话接口"""
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'context' not in data:
            return jsonify({
                "success": False,
                "error": "缺少question或context参数"
            }), 400
        
        question = data['question']
        context = data['context']
        model = data.get('model')
        manager = get_ai_manager()
        
        result = manager.rag_chat(question, context, model)
        
        return jsonify({
            "success": True,
            "data": result,
            "api": manager.get_current_api()
        })
        
    except Exception as e:
        logger.error(f"RAG对话失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        manager = get_ai_manager()
        api_info = manager.get_api_info()
        
        return jsonify({
            "status": "healthy",
            "current_api": api_info["current_api"],
            "available_apis": api_info["available_apis"]
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    from config import HOST, PORT
    
    print(f"启动API管理服务...")
    print(f"服务地址: http://{HOST}:{PORT}")
    print(f"健康检查: http://{HOST}:{PORT}/health")
    print(f"API状态: http://{HOST}:{PORT}/api/status")
    
    app.run(host=HOST, port=PORT, debug=True)
