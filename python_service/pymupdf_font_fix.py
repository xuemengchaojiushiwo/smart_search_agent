#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMuPDF Pro 字体路径修复工具
解决Windows中文用户名导致的字体路径问题
"""

import os
import tempfile
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def find_system_fonts():
    """查找系统字体目录"""
    font_dirs = []
    
    if os.name == 'nt':  # Windows
        # 常见的Windows字体目录
        possible_dirs = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts'),
            'C:\\Windows\\Fonts',
            'C:\\Fonts'
        ]
        
        for font_dir in possible_dirs:
            if os.path.exists(font_dir):
                # 检查是否包含中文字符
                if not any('\u4e00' <= char <= '\u9fff' for char in font_dir):
                    font_dirs.append(font_dir)
                    logger.info(f"找到有效字体目录: {font_dir}")
                else:
                    logger.warning(f"跳过包含中文字符的字体目录: {font_dir}")
    
    return font_dirs

def create_temp_font_dir():
    """创建临时字体目录"""
    temp_dir = os.path.join(tempfile.gettempdir(), 'pymupdf_fonts')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 复制一些基本字体到临时目录
    system_fonts = find_system_fonts()
    if system_fonts:
        source_dir = system_fonts[0]
        basic_fonts = ['arial.ttf', 'times.ttf', 'calibri.ttf', 'verdana.ttf']
        
        for font in basic_fonts:
            source_path = os.path.join(source_dir, font)
            target_path = os.path.join(temp_dir, font)
            if os.path.exists(source_path) and not os.path.exists(target_path):
                try:
                    shutil.copy2(source_path, target_path)
                    logger.info(f"复制字体: {font}")
                except Exception as e:
                    logger.warning(f"复制字体失败 {font}: {e}")
    
    return temp_dir

def setup_pymupdf_pro_environment():
    """设置 PyMuPDF Pro 环境"""
    try:
        # 查找系统字体目录
        system_font_dirs = find_system_fonts()
        
        if system_font_dirs:
            # 使用第一个有效的系统字体目录
            font_dir = system_font_dirs[0]
            os.environ['PYMUPDF_FONT_DIR'] = font_dir
            logger.info(f"设置字体目录: {font_dir}")
        else:
            # 创建临时字体目录
            temp_font_dir = create_temp_font_dir()
            os.environ['PYMUPDF_FONT_DIR'] = temp_font_dir
            logger.info(f"使用临时字体目录: {temp_font_dir}")
        
        # 设置其他环境变量
        os.environ['PYMUPDF_SKIP_FONT_INSTALL'] = '1'  # 跳过字体安装
        os.environ['PYMUPDF_USE_SYSTEM_FONTS'] = '1'   # 使用系统字体
        
        return True
        
    except Exception as e:
        logger.error(f"设置 PyMuPDF Pro 环境失败: {e}")
        return False

def test_pymupdf_pro_initialization():
    """测试 PyMuPDF Pro 初始化"""
    try:
        # 设置环境
        if not setup_pymupdf_pro_environment():
            return False
        
        # 尝试导入和初始化
        import pymupdf.pro
        
        # 尝试解锁（如果有密钥）
        try:
            # 这里可以添加试用密钥
            # pymupdf.pro.unlock("YOUR_TRIAL_KEY")
            pymupdf.pro.unlock()
            logger.info("PyMuPDF Pro 解锁成功")
        except Exception as e:
            logger.warning(f"PyMuPDF Pro 解锁失败，将使用免费版本: {e}")
        
        # 测试基本功能
        import pymupdf
        logger.info("PyMuPDF Pro 初始化成功")
        return True
        
    except Exception as e:
        logger.error(f"PyMuPDF Pro 初始化失败: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🔧 PyMuPDF Pro 字体路径修复工具")
    print("=" * 50)
    
    if test_pymupdf_pro_initialization():
        print("✅ PyMuPDF Pro 环境设置成功")
    else:
        print("❌ PyMuPDF Pro 环境设置失败") 