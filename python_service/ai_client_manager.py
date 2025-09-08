import logging
from typing import List, Dict, Any, Optional
from config import (
    AI_API_SWITCH, 
    GEEKAI_API_KEY, 
    GEEKAI_API_BASE,
    CUSTOM_AI_API_BASE,
    CUSTOM_AI_API_KEY,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    CUSTOM_AI_CHAT_MODEL,
    CUSTOM_AI_EMBEDDING_MODEL
)

# 导入客户端
try:
    from geekai_client import GeekAIClient, init_geekai_client, get_geekai_client
except ImportError:
    GeekAIClient = None
    init_geekai_client = None
    get_geekai_client = None

try:
    from custom_ai_client import CustomAIClient, init_custom_ai_client, get_custom_ai_client
except ImportError:
    CustomAIClient = None
    init_custom_ai_client = None
    get_custom_ai_client = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIClientManager:
    """AI客户端管理器，用于切换不同的API"""
    
    def __init__(self):
        self.current_api = AI_API_SWITCH
        self.geekai_client = None
        self.custom_ai_client = None
        self._init_clients()
    
    def _init_clients(self):
        """初始化所有客户端"""
        try:
            # 初始化极客智坊客户端
            if GeekAIClient and init_geekai_client:
                self.geekai_client = init_geekai_client(GEEKAI_API_KEY, GEEKAI_API_BASE)
                logger.info("极客智坊客户端初始化成功")
            else:
                logger.warning("极客智坊客户端模块未找到")
        except Exception as e:
            logger.error(f"极客智坊客户端初始化失败: {e}")
        
        try:
            # 初始化自定义AI客户端
            if CustomAIClient and init_custom_ai_client:
                self.custom_ai_client = init_custom_ai_client(CUSTOM_AI_API_BASE, CUSTOM_AI_API_KEY)
                logger.info("自定义AI客户端初始化成功")
            else:
                logger.warning("自定义AI客户端模块未找到")
        except Exception as e:
            logger.error(f"自定义AI客户端初始化失败: {e}")
    
    def switch_api(self, api_name: str):
        """切换API
        
        Args:
            api_name: API名称，支持 "geekai" 或 "custom"
        """
        if api_name not in ["geekai", "custom"]:
            logger.error(f"不支持的API名称: {api_name}")
            return False
        
        if api_name == "geekai" and not self.geekai_client:
            logger.error("极客智坊客户端未初始化")
            return False
        
        if api_name == "custom" and not self.custom_ai_client:
            logger.error("自定义AI客户端未初始化")
            return False
        
        self.current_api = api_name
        logger.info(f"已切换到 {api_name} API")
        return True
    
    def get_current_api(self) -> str:
        """获取当前使用的API名称"""
        return self.current_api
    
    def get_current_client(self):
        """获取当前使用的客户端"""
        if self.current_api == "geekai":
            return self.geekai_client
        elif self.current_api == "custom":
            return self.custom_ai_client
        else:
            return None
    
    def is_api_available(self, api_name: str) -> bool:
        """检查指定API是否可用
        
        Args:
            api_name: API名称
            
        Returns:
            是否可用
        """
        if api_name == "geekai":
            return self.geekai_client is not None
        elif api_name == "custom":
            return self.custom_ai_client is not None
        else:
            return False
    
    def get_available_apis(self) -> List[str]:
        """获取可用的API列表"""
        available = []
        if self.geekai_client:
            available.append("geekai")
        if self.custom_ai_client:
            available.append("custom")
        return available
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False) -> Dict[str, Any]:
        """
        调用聊天API
        
        Args:
            messages: 消息列表
            model: 模型名称，如果为None则使用默认模型
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式响应
            
        Returns:
            API响应结果
        """
        client = self.get_current_client()
        if not client:
            return {"error": f"当前API {self.current_api} 客户端不可用"}
        
        # 根据当前API选择默认模型
        if model is None:
            if self.current_api == "geekai":
                model = DEFAULT_CHAT_MODEL
            elif self.current_api == "custom":
                model = CUSTOM_AI_CHAT_MODEL
        
        try:
            if hasattr(client, 'chat_completion'):
                return client.chat_completion(messages, model, temperature, max_tokens, stream)
            else:
                return {"error": f"客户端 {self.current_api} 不支持聊天功能"}
        except Exception as e:
            logger.error(f"聊天API调用异常: {e}")
            return {"error": f"聊天API调用异常: {str(e)}"}
    
    def chat_completion_stream(self, 
                             messages: List[Dict[str, str]], 
                             model: str = None,
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None):
        """
        流式调用聊天API
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Yields:
            流式响应数据
        """
        client = self.get_current_client()
        if not client:
            yield {"error": f"当前API {self.current_api} 客户端不可用"}
            return
        
        # 根据当前API选择默认模型
        if model is None:
            if self.current_api == "geekai":
                model = DEFAULT_CHAT_MODEL
            elif self.current_api == "custom":
                model = CUSTOM_AI_CHAT_MODEL
        
        try:
            if hasattr(client, 'chat_completion_stream'):
                for chunk in client.chat_completion_stream(messages, model, temperature, max_tokens):
                    yield chunk
            else:
                yield {"error": f"客户端 {self.current_api} 不支持流式聊天功能"}
        except Exception as e:
            logger.error(f"流式聊天API调用异常: {e}")
            yield {"error": f"流式聊天API调用异常: {str(e)}"}
    
    def get_embeddings(self, texts: List[str], model: str = None) -> Dict[str, Any]:
        """
        获取文本向量化结果
        
        Args:
            texts: 文本列表
            model: 向量化模型名称
            
        Returns:
            向量化结果
        """
        client = self.get_current_client()
        if not client:
            return {"error": f"当前API {self.current_api} 客户端不可用"}
        
        # 根据当前API选择默认模型
        if model is None:
            if self.current_api == "geekai":
                model = DEFAULT_EMBEDDING_MODEL
            elif self.current_api == "custom":
                model = CUSTOM_AI_EMBEDDING_MODEL
        
        try:
            if hasattr(client, 'get_embeddings'):
                return client.get_embeddings(texts, model)
            else:
                return {"error": f"客户端 {self.current_api} 不支持向量化功能"}
        except Exception as e:
            logger.error(f"向量化API调用异常: {e}")
            return {"error": f"向量化API调用异常: {str(e)}"}
    
    def simple_chat(self, message: str, model: str = None) -> str:
        """
        简单聊天接口
        
        Args:
            message: 用户消息
            model: 模型名称
            
        Returns:
            AI回复内容
        """
        client = self.get_current_client()
        if not client:
            return f"当前API {self.current_api} 客户端不可用"
        
        # 根据当前API选择默认模型
        if model is None:
            if self.current_api == "geekai":
                model = DEFAULT_CHAT_MODEL
            elif self.current_api == "custom":
                model = CUSTOM_AI_CHAT_MODEL
        
        try:
            if hasattr(client, 'simple_chat'):
                return client.simple_chat(message, model)
            else:
                return f"客户端 {self.current_api} 不支持简单聊天功能"
        except Exception as e:
            logger.error(f"简单聊天API调用异常: {e}")
            return f"简单聊天API调用异常: {str(e)}"
    
    def rag_chat(self, 
                 question: str, 
                 context: List[str], 
                 model: str = None) -> Dict[str, Any]:
        """
        RAG对话接口
        
        Args:
            question: 用户问题
            context: 相关文档内容列表
            model: 模型名称
            
        Returns:
            RAG对话结果
        """
        client = self.get_current_client()
        if not client:
            return {
                "answer": f"当前API {self.current_api} 客户端不可用",
                "references": [],
                "success": False
            }
        
        # 根据当前API选择默认模型
        if model is None:
            if self.current_api == "geekai":
                model = DEFAULT_CHAT_MODEL
            elif self.current_api == "custom":
                model = CUSTOM_AI_CHAT_MODEL
        
        try:
            if hasattr(client, 'rag_chat'):
                return client.rag_chat(question, context, model)
            else:
                return {
                    "answer": f"客户端 {self.current_api} 不支持RAG对话功能",
                    "references": [],
                    "success": False
                }
        except Exception as e:
            logger.error(f"RAG对话API调用异常: {e}")
            return {
                "answer": f"RAG对话API调用异常: {str(e)}",
                "references": [],
                "success": False
            }
    
    def rag_chat_stream(self, 
                        question: str, 
                        context: List[str], 
                        model: str = None):
        """
        流式RAG对话接口
        
        Args:
            question: 用户问题
            context: 相关文档内容列表
            model: 模型名称
            
        Yields:
            流式RAG对话结果
        """
        client = self.get_current_client()
        if not client:
            yield {
                "type": "error",
                "error": f"当前API {self.current_api} 客户端不可用"
            }
            return
        
        # 根据当前API选择默认模型
        if model is None:
            if self.current_api == "geekai":
                model = DEFAULT_CHAT_MODEL
            elif self.current_api == "custom":
                model = CUSTOM_AI_CHAT_MODEL
        
        try:
            if hasattr(client, 'rag_chat_stream'):
                for chunk in client.rag_chat_stream(question, context, model):
                    yield chunk
            else:
                yield {
                    "type": "error",
                    "error": f"客户端 {self.current_api} 不支持流式RAG对话功能"
                }
        except Exception as e:
            logger.error(f"流式RAG对话API调用异常: {e}")
            yield {
                "type": "error",
                "error": f"流式RAG对话API调用异常: {str(e)}"
            }
    
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息"""
        return {
            "current_api": self.current_api,
            "available_apis": self.get_available_apis(),
            "geekai_available": self.is_api_available("geekai"),
            "custom_available": self.is_api_available("custom"),
            "geekai_configured": bool(GEEKAI_API_KEY and GEEKAI_API_BASE),
            "custom_configured": bool(CUSTOM_AI_API_BASE),
            "geekai_models": {
                "chat": DEFAULT_CHAT_MODEL,
                "embedding": DEFAULT_EMBEDDING_MODEL
            },
            "custom_models": {
                "chat": CUSTOM_AI_CHAT_MODEL,
                "embedding": CUSTOM_AI_EMBEDDING_MODEL
            }
        }


# 全局管理器实例
ai_manager = None

def init_ai_manager():
    """初始化AI管理器"""
    global ai_manager
    ai_manager = AIClientManager()
    logger.info("AI管理器初始化成功")
    return ai_manager

def get_ai_manager() -> AIClientManager:
    """获取AI管理器实例"""
    global ai_manager
    if ai_manager is None:
        ai_manager = init_ai_manager()
    return ai_manager
