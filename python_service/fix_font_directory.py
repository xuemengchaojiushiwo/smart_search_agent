#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复PyMuPDF Pro字体目录问题
"""

import os
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

def create_temp_font_directory():
    """创建临时字体目录"""
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="pymupdf_fonts_")
        logger.info(f"创建临时字体目录: {temp_dir}")
        
        # 设置环境变量
        os.environ['FONTCONFIG_PATH'] = temp_dir
        os.environ['PYMUPDF_FONT_DIR'] = temp_dir
        
        # 创建字体配置文件
        font_config = os.path.join(temp_dir, "fonts.conf")
        with open(font_config, 'w', encoding='utf-8') as f:
            f.write("""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
    <dir>.</dir>
    <cachedir>.</cachedir>
</fontconfig>
""")
        
        logger.info("临时字体目录设置完成")
        return temp_dir
        
    except Exception as e:
        logger.error(f"创建临时字体目录失败: {e}")
        return None

def cleanup_temp_font_directory(temp_dir):
    """清理临时字体目录"""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"清理临时字体目录: {temp_dir}")
    except Exception as e:
        logger.error(f"清理临时字体目录失败: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    temp_dir = create_temp_font_directory()
    print(f"临时字体目录: {temp_dir}")
