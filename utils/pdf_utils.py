"""
PDF 处理工具函数
"""

import os
import base64
import tempfile
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image
import io
from loguru import logger


def encode_pdf_to_base64(pdf_path: str) -> Optional[str]:
    """
    将 PDF 文件编码为 Base64 字符串
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        Base64 编码的字符串，失败返回 None
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.error(f"读取 PDF 文件失败: {str(e)}")
        return None


def compress_pdf(pdf_path: str, output_path: Optional[str] = None, dpi: int = 72, quality: int = 50, logger_instance=None) -> Optional[str]:
    """
    压缩 PDF 文件以减少文件大小
    
    Args:
        pdf_path: 原始 PDF 文件路径
        output_path: 输出文件路径（如果为 None，则创建临时文件）
        dpi: 图片分辨率（默认 72 DPI）
        quality: JPEG 压缩质量（1-100，默认 50）
        logger_instance: 日志记录器实例（可选）
        
    Returns:
        压缩后的 PDF 文件路径，失败返回 None
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    
    log = logger_instance if logger_instance else logger
    
    try:
        # 如果没有指定输出路径，创建临时文件
        if output_path is None:
            temp_fd, output_path = tempfile.mkstemp(suffix='.pdf', prefix='compressed_')
            os.close(temp_fd)
        
        doc = fitz.open(pdf_path)
        
        # 压缩每页中的图片
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            
            for img_index, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 使用 PIL 处理图片
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    
                    # 转换为 RGB（如果是 RGBA 或其他格式）
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    
                    # 计算缩放比例（降低分辨率）
                    scale_factor = dpi / 150.0  # 假设原始 DPI 为 150
                    new_width = int(img_pil.width * scale_factor)
                    new_height = int(img_pil.height * scale_factor)
                    
                    # 调整图片大小
                    if scale_factor < 1.0:
                        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 保存为 JPEG（压缩质量较低）
                    img_byte_arr = io.BytesIO()
                    img_pil.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                    img_byte_arr.seek(0)
                    
                    # 使用 page.replace_image() 替换图片
                    page.replace_image(xref, stream=img_byte_arr)
                except Exception as e:
                    # 如果某张图片处理失败，继续处理其他图片
                    log.warning(f"压缩第 {page_num + 1} 页的图片 {img_index} 失败: {str(e)}")
                    continue
        
        # 保存压缩后的 PDF（使用垃圾回收和压缩选项）
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        
        original_size = os.path.getsize(pdf_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        log.info(f"PDF 压缩完成: {original_size / 1024 / 1024:.2f} MB -> {compressed_size / 1024 / 1024:.2f} MB (压缩率: {compression_ratio:.1f}%)")
        
        return output_path
        
    except Exception as e:
        log.error(f"压缩 PDF 失败: {str(e)}")
        return None


def pdf_page_to_image(doc: fitz.Document, page_num: int, zoom: float = 2.0, logger_instance=None) -> Optional[Image.Image]:
    """
    从已打开的PDF文档中提取指定页面并转换为PIL Image
    
    Args:
        doc: 已打开的PDF文档对象
        page_num: 页面编号（从0开始）
        zoom: 缩放因子，用于提高图片清晰度
        logger_instance: 日志记录器实例（可选）
        
    Returns:
        PIL Image对象，失败返回None
    """
    log = logger_instance if logger_instance else logger
    
    try:
        if page_num < 0 or page_num >= len(doc):
            log.error(f"页面编号 {page_num} 超出范围（总页数: {len(doc)}）")
            return None
        
        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为 PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        return image
        
    except Exception as e:
        log.error(f"转换PDF第 {page_num + 1} 页为图片时出错: {e}", exc_info=True)
        return None


def is_valid_pdf(pdf_path: str, min_size: int = 1024) -> bool:
    """
    简单校验 PDF 文件是否完整且非空
    
    Args:
        pdf_path: PDF 文件路径
        min_size: 最小文件大小（字节），默认 1024
        
    Returns:
        如果是有效的 PDF 文件，返回 True
    """
    if not os.path.exists(pdf_path):
        return False
    try:
        if os.path.getsize(pdf_path) < min_size:
            return False
        with open(pdf_path, "rb") as f:
            header = f.read(5)
        return header.startswith(b"%PDF-")
    except OSError:
        return False
