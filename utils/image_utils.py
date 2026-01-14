"""
图片处理工具函数
"""

import base64
import io
from typing import Dict
from PIL import Image


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    将PIL Image转换为base64编码的字符串
    
    Args:
        image: PIL Image对象
        format: 图片格式（默认 PNG）
        
    Returns:
        base64编码的字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


def crop_image(image: Image.Image, box: Dict[str, float]) -> Image.Image:
    """
    根据边界框裁剪图片
    
    Args:
        image: 原始PIL Image对象
        box: 包含x, y, width, height的字典
        
    Returns:
        裁剪后的PIL Image对象
    """
    x = int(box["x"])
    y = int(box["y"])
    width = int(box["width"])
    height = int(box["height"])
    
    # 确保坐标在图片范围内
    img_width, img_height = image.size
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    width = max(0.1, min(width, img_width - x))
    height = max(0.1, min(height, img_height - y))
    
    # 裁剪图片
    crop_box = (x, y, x + width, y + height)
    cropped = image.crop(crop_box)
    
    return cropped
