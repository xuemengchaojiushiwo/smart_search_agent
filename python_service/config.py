# Python服务配置文件
import os
import json
from pathlib import Path
import os

# Elasticsearch配置
EMBEDDING_DIMS = 1536  # 默认1536维，切换自定义API时改为1024
ES_INDEX = f'knowledge_base_new_{EMBEDDING_DIMS}'

ES_CONFIG = {
    'host': 'localhost',
    'port': 9200,
    'index': ES_INDEX,
    'username': 'elastic',
    'password': 'password',
    'verify_certs': False
}

# 文档处理配置
DOCUMENT_CONFIG = {
    "chunk_size": 4000,  # 从1000增加到4000，减少过度分块
    "chunk_overlap": 800,  # 从300增加到800，提高重叠比例
    "allowed_extensions": {".pdf", ".docx", ".xlsx", ".txt", ".pptx"},
    # 动态重叠策略
    "dynamic_overlap": {
        "pdf": 0.25,      # PDF文档25%重叠（从35%减少）
        "docx": 0.20,     # Word文档20%重叠
        "pptx": 0.20,     # PowerPoint文档20%重叠
        "xlsx": 0.25,     # Excel文档25%重叠
        "txt": 0.15,      # 文本文档15%重叠
        "default": 0.20   # 默认20%重叠
    },
    # 分块器配置
    "splitter_config": {
        "separators": [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            "。",    # 中文句号
            "！",    # 中文感叹号
            "？",    # 中文问号
            ".",     # 英文句号
            "!",     # 英文感叹号
            "?",     # 英文问号
            ";",     # 分号
            ":",     # 冒号
            "，",    # 中文逗号
            ",",     # 英文逗号
            " ",     # 空格
        ],
        "keep_separator": True,  # 保留分隔符
        "is_separator_regex": False  # 不使用正则表达式
    }
}

# Embedding模型配置
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "cpu",
}

# RAG配置
RAG_CONFIG = {
    "top_k": 3,  # 检索最相近的文档数量（从5改为3）
    "context_limit": 500  # 上下文长度限制
}

# PyMuPDF Pro 配置
PYMUPDF_PRO_CONFIG = {
    "enabled": True,
    "trial_key": "HZ1A5z94wQ9+85/85z+jkMX3",  # 试用密钥
    "supported_formats": {
        "word": [".doc", ".docx"],
        "excel": [".xls", ".xlsx"],
        "powerpoint": [".ppt", ".pptx"],
        "pdf": [".pdf"],
        "text": [".txt"],
        "hangul": [".hwp", ".hwpx"]
    },
    "processing_methods": {
        "primary": "pymupdf_pro",
        "fallback": "traditional",
        "chunking": "mypymupdf4llm"
    }
}

# 分块策略配置
CHUNKING_CONFIG = {
    "use_pymupdf4llm_for_pdf": True,
    "fallback_to_current": True,
    "markdown_headers": [
        ("#", "标题1"),
        ("##", "标题2"), 
        ("###", "标题3"),
        ("####", "标题4"),
        ("#####", "标题5"),
        ("######", "标题6"),
    ],
    "traditional_splitter": {
        "chunk_size": 4000,  # 从1000增加到4000
        "chunk_overlap": 800,  # 从300增加到800
        "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
    },
    # PyMuPDF4LLM 优化配置
    "pymupdf4llm_config": {
        "min_chunks": 2,  # 最少chunks数量（从3减少到2）
        "max_chunk_size": 6000,  # 最大chunk大小（从1500增加到6000）
        "min_chunk_size": 200,   # 最小chunk大小
        "use_semantic_overlap": True,  # 使用语义重叠
        "preserve_structure": True,     # 保持文档结构
        "enhance_with_traditional": True  # 使用传统分块增强
    }
}

# AI API 配置开关
AI_API_SWITCH = "geekai"  # 可选值: "geekai", "custom"

# 极客智坊 API 配置（优先从本地 secrets.json 读取，其次环境变量）
def _load_secret_key_from_file() -> str:
    try:
        secrets_path = Path(__file__).parent / "secrets.json"
        if secrets_path.exists():
            with open(secrets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                key = data.get("GEEKAI_API_KEY", "")
                if isinstance(key, str):
                    return key
    except Exception:
        # 静默失败，回退到环境变量
        pass
    return ""

GEEKAI_API_KEY = _load_secret_key_from_file() or os.getenv("GEEKAI_API_KEY", "")
GEEKAI_API_BASE = "https://geekai.co/api/v1"
GEEKAI_EMBEDDING_URL = f"{GEEKAI_API_BASE}/embeddings"
GEEKAI_CHAT_URL = f"{GEEKAI_API_BASE}/chat/completions"

# 自定义AI API 配置
CUSTOM_AI_API_BASE = ""  # 自定义AI API地址，暂时为空
CUSTOM_AI_API_KEY = ""   # 自定义AI API密钥，暂时为空
CUSTOM_AI_CHAT_URL = f"{CUSTOM_AI_API_BASE}/chat/completions" if CUSTOM_AI_API_BASE else ""
CUSTOM_AI_EMBEDDING_URL = f"{CUSTOM_AI_API_BASE}/embeddings" if CUSTOM_AI_API_BASE else ""

# 默认模型配置
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# 自定义AI模型配置
CUSTOM_AI_CHAT_MODEL = "default"
CUSTOM_AI_EMBEDDING_MODEL = "multilingual-e5-large-instruct"

# 服务配置
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 