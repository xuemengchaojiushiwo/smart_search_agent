import requests
import json
import logging
from typing import List, Dict, Any, Optional
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomAIClient:
    """自定义AI API客户端"""
    
    def __init__(self, api_base: str = "", api_key: str = ""):
        self.api_base = api_base
        self.api_key = api_key
        self.chat_url = f"{api_base}/chat/completions" if api_base else ""
        self.embedding_url = f"{api_base}/embeddings" if api_base else ""
        
        # 默认请求头
        self.headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def is_configured(self) -> bool:
        """检查API是否已配置"""
        return bool(self.api_base and self.chat_url and self.embedding_url)
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "default",
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False) -> Dict[str, Any]:
        """
        调用自定义AI聊天API
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "你好"}]
            model: 模型名称
            temperature: 温度参数，控制随机性
            max_tokens: 最大token数
            stream: 是否流式响应
            
        Returns:
            API响应结果
        """
        if not self.is_configured():
            return {"error": "自定义AI API未配置，请先设置API地址"}
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
            if stream:
                payload["stream"] = True
            
            logger.info(f"调用自定义AI API: model={model}, messages_count={len(messages)}")
            
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"自定义AI API调用成功: model={model}")
                return result
            else:
                logger.error(f"自定义AI API调用失败: status_code={response.status_code}, response={response.text}")
                return {"error": f"API调用失败: {response.status_code}", "details": response.text}
                
        except Exception as e:
            logger.error(f"自定义AI API调用异常: {str(e)}")
            return {"error": f"API调用异常: {str(e)}"}
    
    def chat_completion_stream(self, 
                             messages: List[Dict[str, str]], 
                             model: str = "default",
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None):
        """
        流式调用自定义AI聊天API
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Yields:
            流式响应数据
        """
        if not self.is_configured():
            yield {"error": "自定义AI API未配置，请先设置API地址"}
            return
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            logger.info(f"流式调用自定义AI API: model={model}")
            
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # 去掉 'data: ' 前缀
                            if data == '[DONE]':
                                break
                            try:
                                json_data = json.loads(data)
                                yield json_data
                            except json.JSONDecodeError:
                                logger.warning(f"解析流式数据失败: {data}")
                                continue
            else:
                logger.error(f"流式API调用失败: status_code={response.status_code}")
                yield {"error": f"流式API调用失败: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"流式API调用异常: {str(e)}")
            yield {"error": f"流式API调用异常: {str(e)}"}
    
    def get_embeddings(self, texts: List[str], model: str = "multilingual-e5-large-instruct") -> Dict[str, Any]:
        """
        获取文本向量化结果（1024维度）
        
        Args:
            texts: 文本列表
            model: 向量化模型名称
            
        Returns:
            向量化结果
        """
        if not self.is_configured():
            return {"error": "自定义AI API未配置，请先设置API地址"}
        
        try:
            payload = {
                "model": model,
                "input": texts
            }
            
            logger.info(f"调用自定义AI向量化API: model={model}, texts_count={len(texts)}")
            
            response = requests.post(
                self.embedding_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"向量化API调用成功: model={model}")
                
                # 验证embedding维度是否为1024
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0].get("embedding", [])
                    if len(embedding) != 1024:
                        logger.warning(f"自定义AI embedding维度为{len(embedding)}，期望1024")
                
                return result
            else:
                logger.error(f"向量化API调用失败: status_code={response.status_code}, response={response.text}")
                return {"error": f"向量化API调用失败: {response.status_code}", "details": response.text}
                
        except Exception as e:
            logger.error(f"向量化API调用异常: {str(e)}")
            return {"error": f"向量化API调用异常: {str(e)}"}
    
    def simple_chat(self, message: str, model: str = "default") -> str:
        """
        简单聊天接口
        
        Args:
            message: 用户消息
            model: 模型名称
            
        Returns:
            AI回复内容
        """
        messages = [{"role": "user", "content": message}]
        result = self.chat_completion(messages, model)
        
        if "error" in result:
            return f"调用失败: {result['error']}"
        
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"解析响应失败: {e}")
            return f"解析响应失败: {str(e)}"
    
    def rag_chat(self, 
                 question: str, 
                 context: List[str], 
                 model: str = "default") -> Dict[str, Any]:
        """
        RAG对话接口
        
        Args:
            question: 用户问题
            context: 相关文档内容列表
            model: 模型名称
            
        Returns:
            RAG对话结果
        """
        try:
            # 构建RAG提示词
            context_text = "\n\n".join(context)
            prompt = f"""基于以下文档内容回答问题：

文档内容：
{context_text}

问题：{question}

请基于上述文档内容回答用户的问题。如果文档中没有相关信息，请明确说明。回答要准确、简洁、有用。"""

            messages = [{"role": "user", "content": prompt}]
            result = self.chat_completion(messages, model, temperature=0.3)
            
            if "error" in result:
                return {
                    "answer": f"调用失败: {result['error']}",
                    "references": [],
                    "success": False
                }
            
            answer = result["choices"][0]["message"]["content"]
            
            return {
                "answer": answer,
                "references": context,
                "success": True,
                "model": model
            }
            
        except Exception as e:
            logger.error(f"RAG对话异常: {str(e)}")
            return {
                "answer": f"RAG对话异常: {str(e)}",
                "references": [],
                "success": False
            }
    
    def rag_chat_stream(self, 
                        question: str, 
                        context: List[str], 
                        model: str = "default"):
        """
        流式RAG对话接口
        
        Args:
            question: 用户问题
            context: 相关文档内容列表
            model: 模型名称
            
        Yields:
            流式RAG对话结果
        """
        try:
            # 构建RAG提示词
            context_text = "\n\n".join(context)
            prompt = f"""基于以下文档内容回答问题：

文档内容：
{context_text}

问题：{question}

请基于上述文档内容回答用户的问题。如果文档中没有相关信息，请明确说明。回答要准确、简洁、有用。"""

            messages = [{"role": "user", "content": prompt}]
            
            # 发送开始标记
            yield {
                "type": "start",
                "question": question,
                "references": context,
                "model": model
            }
            
            # 流式获取回答
            content_buffer = ""
            for chunk in self.chat_completion_stream(messages, model, temperature=0.3):
                if "error" in chunk:
                    yield {
                        "type": "error",
                        "error": chunk["error"]
                    }
                    return
                
                try:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            content_buffer += content
                            yield {
                                "type": "content",
                                "content": content,
                                "full_content": content_buffer
                            }
                except Exception as e:
                    logger.error(f"解析流式数据失败: {e}")
                    continue
            
            # 发送结束标记
            yield {
                "type": "end",
                "full_answer": content_buffer,
                "references": context
            }
            
        except Exception as e:
            logger.error(f"流式RAG对话异常: {str(e)}")
            yield {
                "type": "error",
                "error": f"流式RAG对话异常: {str(e)}"
            }


# 全局客户端实例
custom_ai_client = None

def init_custom_ai_client(api_base: str = "", api_key: str = ""):
    """初始化自定义AI客户端"""
    global custom_ai_client
    custom_ai_client = CustomAIClient(api_base, api_key)
    if custom_ai_client.is_configured():
        logger.info("自定义AI客户端初始化成功")
    else:
        logger.warning("自定义AI客户端初始化完成，但API地址未配置")
    return custom_ai_client

def get_custom_ai_client() -> CustomAIClient:
    """获取自定义AI客户端实例"""
    global custom_ai_client
    if custom_ai_client is None:
        custom_ai_client = CustomAIClient()
    return custom_ai_client
