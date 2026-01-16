"""
PDF 图片提取模块
从 PDF 文件中提取论文的 Overview/Architecture 图
使用 GPT-4o-mini 识别第一页中的概览图位置并截取
"""

import os
import re
from datetime import datetime
from typing import Optional, Dict
import fitz  # PyMuPDF
from loguru import logger

from utils.pdf_utils import pdf_page_to_image
from utils.image_utils import image_to_base64, crop_image
from utils.file_utils import get_safe_model_name
from utils.llm_client import create_llm_client

import dotenv

dotenv.load_dotenv(override=True)

prompt = """
你是一个专业的视觉定位助手。请分析这张学术论文第一页的图片，识别出论文的概览图（Overview Figure）或架构图（Architecture Figure）。

概览图的特征：
1. 展示整个系统/方法架构的示意图
2. 包含多个组件和它们之间关系的图表
3. 可能包含"Figure 1"、"Overview"、"Architecture" 等 Caption

输出格式要求：
- 使用 <box>(x1,y1),(x2,y2)</box> 格式输出边界框坐标
- 坐标值必须是归一化到 [0, 1000] 范围的整数
- (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标
- 只输出 <box></box> 标签，不要输出任何其他文字
- 如果找不到概览图，输出 <box>null</box>

示例格式：<box>(100,200),(800,600)</box>
"""

class PDFImageExtractor:
    """PDF 图片提取器 - 使用 VLM识别概览图位置"""
    
    def __init__(self, output_dir: str, grounding_model: str):
        """
        初始化图片提取器
        
        Args:
            output_dir: 图片保存目录
            grounding_model: 用于识别概览图的视觉模型
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logger
        self.llm_client = create_llm_client(is_grounding=True)
        self.grounding_model = grounding_model
    
    def extract_image(
        self, 
        pdf_path: str, 
        arxiv_id: str, 
        max_pages: int
    ) -> Optional[str]:
        """
        从 PDF 中提取概览图，逐页搜索直到找到或达到限制
        
        Args:
            pdf_path: PDF 文件路径
            arxiv_id: arXiv ID，用于命名图片文件
            max_pages: 最大搜索页数
            
        Returns:
            图片文件路径，失败返回 None
        """
        if not pdf_path or not os.path.exists(pdf_path):
            self.logger.warning(f"PDF 文件不存在: {pdf_path}")
            return None
        
        # 构建文件名后缀
        suffix_parts = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if timestamp:
            suffix_parts.append(timestamp)
        # 使用grounding model的名称（转换为安全的文件名格式）
        safe_model_name = get_safe_model_name(self.grounding_model)
        suffix_parts.append(safe_model_name)
        suffix = "_".join(suffix_parts) if suffix_parts else ""
        
        try:
            # 打开PDF文件，获取总页数
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            if total_pages == 0:
                self.logger.error("PDF文件为空")
                doc.close()
                return None
            
            # 确定实际搜索的页数
            search_pages = min(max_pages, total_pages)
            self.logger.info(f"PDF共有 {total_pages} 页，将搜索前 {search_pages} 页")
            
            # 逐页搜索概览图
            for page_num in range(search_pages):
                self.logger.info(f"正在搜索第 {page_num + 1} 页...")
                
                # 1. 读取当前页并转换为图片
                page_image = pdf_page_to_image(doc, page_num, logger_instance=self.logger)
                if page_image is None:
                    self.logger.warning(f"无法将第 {page_num + 1} 页转换为图片，跳过")
                    continue
                
                # 2. 将图片转换为base64
                image_base64 = image_to_base64(page_image)
                
                # 获取图片尺寸
                img_width, img_height = page_image.size
                
                # 3. 调用 VLM 识别概览图的 bbox
                self.logger.info(f"正在识别第 {page_num + 1} 页的概览图位置...")
                box = self._identify_overview_bbox_with_gpt(image_base64, img_width, img_height)
                if box is None:
                    self.logger.info(f"第 {page_num + 1} 页未找到概览图，继续搜索下一页")
                    continue
                
                # 4. 根据Box裁剪图片
                self.logger.info(f"在第 {page_num + 1} 页识别到概览图位置: {box}")
                cropped_image = crop_image(page_image, box)
                
                # 5. 保存裁剪后的图片
                if suffix:
                    image_filename = f"{arxiv_id}_overview_{suffix}.png"
                else:
                    image_filename = f"{arxiv_id}_overview.png"
                image_path = os.path.join(self.output_dir, image_filename)
                cropped_image.save(image_path, "PNG")
                
                doc.close()
                self.logger.info(f"成功从第 {page_num + 1} 页提取图片: {image_path}")
                return image_path
            
            # 所有页面都搜索完毕，未找到概览图
            doc.close()
            self.logger.warning(f"已搜索前 {search_pages} 页，均未找到概览图")
            return None
            
        except Exception as e:
            self.logger.error(f"提取图片时出错: {e}", exc_info=True)
            return None
    
    def _identify_overview_bbox_with_gpt(self, image_base64: str, img_width: int, img_height: int) -> Optional[Dict[str, float]]:
        """
        识别概览图的边界框
        
        Args:
            image_base64: base64编码的图片
            img_width: 图片宽度（像素）
            img_height: 图片高度（像素）
            
        Returns:
            包含x, y, width, height的字典（像素坐标），失败返回None
        """

        try:
            # 使用统一的图片输入接口
            result = self.llm_client.chat_with_image(
                model=self.grounding_model,
                prompt=prompt,
                image_base64=image_base64,
                image_mime="image/png",
                max_tokens=8192,
                temperature=0.1
            )
            
            if "error" in result:
                self.logger.error(f"调用模型识别概览图时出错: {result['error']}")
                return None
            
            content = result.get("content", "").strip()
            self.logger.debug(f"模型响应: {content}")
            
            # 解析 <box>(x1, y1),(x2, y2)</box> 格式
            # 支持两种格式：<box>(x1,y1),(x2,y2)</box> 和 <box>(x1, y1),(x2, y2)</box>
            box_patterns = [
                r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>',  # 无空格格式
                r'<box>\((\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)</box>'  # 有空格格式
            ]
            
            match = None
            for pattern in box_patterns:
                match = re.search(pattern, content)
                if match:
                    break
            
            if match:
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, match.groups())
                
                # 模型输出的坐标通常是归一化到 [0, 1000] 范围的整数，需要转换为像素坐标
                x1_pixel = (x1_norm / 1000.0) * img_width
                y1_pixel = (y1_norm / 1000.0) * img_height
                x2_pixel = (x2_norm / 1000.0) * img_width
                y2_pixel = (y2_norm / 1000.0) * img_height
                
                # 确保坐标顺序正确
                x_min = min(x1_pixel, x2_pixel)
                x_max = max(x1_pixel, x2_pixel)
                y_min = min(y1_pixel, y2_pixel)
                y_max = max(y1_pixel, y2_pixel)
                
                # 计算原始宽度和高度
                width = x_max - x_min
                height = y_max - y_min
                
                # 向外扩展 bbox（按比例和固定像素扩展，确保信息完整）
                # 扩展比例：3%，最小扩展：3像素
                expand_ratio = 0.03
                min_expand_pixels = 3
                
                expand_x = max(width * expand_ratio, min_expand_pixels)
                expand_y = max(height * expand_ratio, min_expand_pixels)
                
                # 扩展边界
                x_min_expanded = max(0, x_min - expand_x)
                x_max_expanded = min(img_width, x_max + expand_x)
                y_min_expanded = max(0, y_min - expand_y)
                y_max_expanded = min(img_height, y_max + expand_y)
                
                # 转换为 x, y, width, height 格式（像素坐标）
                bbox = {
                    "x": float(x_min_expanded),
                    "y": float(y_min_expanded),
                    "width": float(x_max_expanded - x_min_expanded),
                    "height": float(y_max_expanded - y_min_expanded)
                }
                
                self.logger.info(f"归一化坐标: ({x1_norm}, {y1_norm}), ({x2_norm}, {y2_norm})")
                self.logger.info(f"处理后的像素坐标: 左上角({x_min_expanded:.1f}, {y_min_expanded:.1f}), 右下角({x_max_expanded:.1f}, {y_max_expanded:.1f})")
                return bbox
            
            # 检查是否有 null
            if "null" in content.lower() or "<box>null</box>" in content.lower():
                self.logger.warning("模型返回null，未找到概览图")
                return None
            
            self.logger.error(f"无法解析box坐标，响应内容: {content}")
            return None
            
        except Exception as e:
            self.logger.error(f"调用模型识别概览图时出错: {e}", exc_info=True)
            return None

def main():
    import sys
    
    output_dir = 'test_images'
    pdf_path = 'pdfs/2410.08164.pdf'
    arxiv_id = '2410.08164'
    
    extractor = PDFImageExtractor(output_dir=output_dir, grounding_model="Qwen/Qwen3-VL-235B-A22B-Instruct")
    
    print(f"开始提取图片...")
    print(f"PDF 文件: {pdf_path}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    # 提取图片
    image_path = extractor.extract_image(pdf_path, arxiv_id)
    
    if image_path:
        print(f"\n✓ 成功提取图片: {image_path}")
        print(f"  文件大小: {os.path.getsize(image_path) / 1024:.1f} KB")
    else:
        print("\n✗ 未能提取图片")
        sys.exit(1)


if __name__ == "__main__":
    main()
