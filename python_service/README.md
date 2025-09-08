# 智能知识库系统

## 系统概述

本系统是一个基于RAG（检索增强生成）的智能知识库管理系统，集成了PyMuPDF Pro、PyMuPDF4LLM、LangChain和极客智坊API，实现文档智能处理和智能问答功能。

## 核心架构

```
用户上传文档 → PyMuPDF Pro + PyMuPDF4LLM → LangChain分块 → Embedding → ES存储
                                                                    ↓
用户提问 → RAG检索 → 极客智坊API → 智能对话
```

## 技术栈

### 文档处理层
- **PyMuPDF Pro**: 统一文档处理引擎，支持PDF、Word、Excel、PowerPoint、TXT等
- **PyMuPDF4LLM**: 基于LlamaIndex的文档结构化处理，保持语义结构

### 分块处理层
- **LangChain**: 文本分块和向量化
- **MarkdownHeaderTextSplitter**: 基于Markdown标题的结构化分块
- **RecursiveCharacterTextSplitter**: 传统分块作为补充

### 向量化存储层
- **Embedding模型**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Elasticsearch**: 向量相似度检索和元数据管理

### LLM对话层
- **极客智坊API**: GPT-4o-mini模型，基于检索结果的智能问答

## 环境要求

- Python 3.8+
- Elasticsearch 8.x
- 至少4GB内存（用于运行embedding模型）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

### 主要配置文件

#### `config.py`
```python
# 文档处理配置
DOCUMENT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 300,
    "dynamic_overlap": {...}
}

# PyMuPDF Pro配置
PYMUPDF_PRO_CONFIG = {
    "enabled": True,
    "trial_key": "...",
    "supported_formats": {...}
}

# 极客智坊API配置
GEEKAI_API_KEY = "sk-..."
GEEKAI_CHAT_URL = "https://geekai.co/api/v1/chat/completions"
```

## 启动服务

```bash
# 使用启动脚本
start_python_service.bat

# 或直接运行
cd python_service
python app_main.py
```

服务将在 http://localhost:8000 启动

## API接口

### 1. 健康检查
```
GET /api/health
```

### 2. LDAP用户验证
```
POST /api/ldap/validate
{
  "username": "admin",
  "password": "password"
}
```

### 3. 文档处理
```
POST /api/document/process
Content-Type: multipart/form-data

参数:
- file: 上传的文档文件
- knowledge_id: 知识ID
- knowledge_name: 知识名称
- description: 知识描述
- tags: 标签（逗号分隔）
- effective_time: 生效时间
```

### 4. RAG智能问答
```
POST /api/rag/chat
{
  "question": "用户问题",
  "user_id": "用户ID"
}
```

## 支持的文件类型

### PDF文件 (.pdf)
- 使用 PyMuPDF Pro 解析
- PyMuPDF4LLM 结构化处理
- 提取文本内容和页面信息

### Word文档 (.docx)
- 使用 PyMuPDF Pro 解析
- 提取段落文本和表格内容
- 保持文档结构

### Excel表格 (.xlsx)
- 使用 PyMuPDF Pro 解析
- 提取所有工作表数据
- 按行组织数据

### PowerPoint演示文稿 (.pptx)
- 使用 PyMuPDF Pro 解析
- 提取幻灯片文本内容
- 按幻灯片组织内容

### 文本文件 (.txt)
- 直接读取文本内容
- 支持UTF-8编码

## 核心功能

### 1. 智能文档处理
- **多格式支持**: PDF、Word、Excel、PowerPoint、TXT等
- **结构化分块**: 基于文档标题层级的智能分块
- **语义保持**: 保持文档的语义完整性和上下文连贯性

### 2. 向量化存储
- **高效向量化**: 使用多语言Embedding模型
- **相似度检索**: 基于余弦相似度的文档检索
- **元数据管理**: 完整的知识库元数据管理

### 3. 智能问答
- **RAG检索**: 基于向量相似度的相关文档检索
- **上下文构建**: 动态组合检索到的文档内容
- **智能生成**: 使用极客智坊API生成高质量回答

## 性能优化

### 1. 文档处理优化
- **并行处理**: 支持多文档并行处理
- **内存管理**: 使用临时文件处理大文档
- **缓存机制**: 缓存Embedding模型

### 2. 检索优化
- **索引优化**: ES索引配置优化
- **查询优化**: 相似度检索算法优化
- **结果缓存**: 缓存常用查询结果

### 3. API调用优化
- **超时设置**: 30秒超时保护
- **重试机制**: API调用失败自动重试
- **回退策略**: API失败时使用模拟回答

## 监控和日志

### 健康检查
- **ES连接检查**: 验证Elasticsearch连接
- **PyMuPDF Pro检查**: 验证文档处理能力
- **极客智坊API检查**: 验证API连接状态

### 日志记录
- **处理日志**: 记录文档处理过程
- **API日志**: 记录极客智坊API调用
- **错误日志**: 记录异常和错误信息

## 故障排除

### 常见问题

**文档解析失败**
- 检查文件格式是否正确
- 确认文件未损坏
- 查看日志获取详细错误信息

**ES连接失败**
- 检查ES服务是否启动
- 确认连接配置正确
- 检查网络连接

**极客智坊API调用失败**
- 检查API密钥是否正确
- 确认网络连接正常
- 查看API调用日志

**内存不足**
- 增加系统内存
- 减少chunk_size配置
- 分批处理大文档

## 扩展性

### 1. 新文档格式支持
- 在PyMuPDF Pro配置中添加新格式
- 在文档处理函数中添加解析逻辑

### 2. 新LLM集成
- 在配置中添加新的API配置
- 在对话函数中添加新的API调用

### 3. 新检索策略
- 在检索函数中添加新的检索算法
- 支持混合检索策略

## 总结

本系统实现了完整的RAG架构：
- **文档处理**: PyMuPDF Pro + PyMuPDF4LLM
- **分块策略**: LangChain结构化分块
- **向量存储**: Elasticsearch
- **智能问答**: 极客智坊API + GPT-4o-mini

通过这种架构，系统能够：
1. 智能处理多种文档格式
2. 保持文档的语义结构
3. 提供准确的相似度检索
4. 生成高质量的智能回答 