"""
播客文章生成模块
调用支持文件输入的 LLM（如 Gemini）直接处理 PDF 文件生成播客文章
"""

import os
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from loguru import logger

from utils.pdf_utils import encode_pdf_to_base64, compress_pdf

import dotenv

dotenv.load_dotenv(override=True)

# 播客生成 Prompt 模板
PODCAST_PROMPT = """
总结给定 pdf 中的论文，写成一篇博客文章
写成三段
1、问题背景（包括当前该领域的方法，这种方法会造成什么问题，本文动机等内容）
2、核心贡献（如果有，介绍论文提出的新方法、框架、模型或数据集）、关键技术（如果有，介绍解决方案的核心创新点、架构设计或训练策略）
3、实验设置（简略）、实验结果（重点说结论，辅以小部分重要数据）、该工作的意义与影响

内容要求：
1、论文的任务，动机，挑战，设计理念和high-level的方法多介绍，不用太细节
2、语言要自然流畅规范（重要），作为每日论文播客的内容
3、三段字数分别为 200、400、300 左右，总字数严格控制在 900 字以内（非常重要）

输出格式要求：
1、加粗重要的句子或词语，不要加粗太多内容
2、直接输出三段正文内容，不要添加段落标题或序号
3、英文、数字以及带引号的词要和两边的中文空一格，比如 “这是 trival 的”，“像 “大脑” 一样”。
4、不需要加入对应到文章某一页之类的内容
"""

class PodcastGenerator:
    """播客文章生成器"""
    
    def __init__(self, model: str):
        """
        初始化生成器
        
        Args:
            model: 使用的 LLM 模型名称，需要支持文件输入
        """
        self.client = OpenAI()
        self.model = model
        self.logger = logger
    
    def _call_llm_api(self, pdf_path: str, prompt: str) -> Dict[str, Any]:
        """
        调用 LLM API 生成播客内容
        
        Args:
            pdf_path: PDF 文件路径
            prompt: 提示词
            
        Returns:
            包含生成结果的字典，失败时包含 error 字段
        """
        # 将 PDF 编码为 Base64
        base64_string = encode_pdf_to_base64(pdf_path)
        if not base64_string:
            return {"error": f"无法读取 PDF 文件: {pdf_path}"}
        
        # 获取文件名
        filename = os.path.basename(pdf_path)

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": f"data:application/pdf;base64,{base64_string}",
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=8196
            )
            
            podcast_content = response.choices[0].message.content
            podcast_content = re.sub(r'<think>.*?</think>', '', podcast_content, flags=re.DOTALL).strip()
            
            return {"podcast_content": podcast_content}
            
        except Exception as e:
            error_msg = str(e)
            return {"error": f"LLM 调用失败: {error_msg}", "error_msg": error_msg}
    
    def generate_podcast(
        self, 
        paper_info: Dict[str, Any],
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成播客文章
        
        Args:
            paper_info: 论文信息字典，必须包含 pdf_path 字段
            custom_prompt: 自定义 prompt（可选）
            
        Returns:
            包含生成结果的字典
        """
        if "error" in paper_info:
            return {"error": paper_info["error"]}
        
        pdf_path = paper_info.get("pdf_path")
        if not pdf_path:
            return {"error": "未找到 PDF 文件路径"}
        
        # 使用默认或自定义 prompt
        prompt = custom_prompt or PODCAST_PROMPT
        
        # 第一次尝试：使用原始 PDF
        result = self._call_llm_api(pdf_path, prompt)
        
        # 如果失败且是文件大小超限错误，尝试压缩后重试
        compressed_pdf_path = None
        if "error" in result:
            error_msg = result.get("error_msg", result.get("error", ""))
            if 'file_above_max_size' in error_msg:
                self.logger.warning(f"PDF 文件大小超限，尝试压缩后重试: {pdf_path}")
                
                # 压缩 PDF
                compressed_pdf_path = compress_pdf(pdf_path, logger_instance=self.logger)
                if compressed_pdf_path:
                    try:
                        # 使用压缩后的 PDF 重试
                        result = self._call_llm_api(compressed_pdf_path, prompt)
                    except Exception as e:
                        self.logger.error(f"使用压缩 PDF 重试时出错: {str(e)}")
                        result = {"error": f"使用压缩 PDF 重试时出错: {str(e)}"}
                else:
                    self.logger.error("PDF 压缩失败，无法重试")
        
        # 清理临时压缩文件
        if compressed_pdf_path and os.path.exists(compressed_pdf_path):
            try:
                os.remove(compressed_pdf_path)
            except Exception as e:
                self.logger.warning(f"删除临时压缩文件失败: {str(e)}")
        
        if "error" in result:
            return {"error": result["error"]}
        
        return {
            "title": paper_info.get('title', 'Unknown'),
            "arxiv_id": paper_info.get('arxiv_id', ''),
            "authors": paper_info.get('authors', ''),
            "podcast_content": result.get("podcast_content", ""),
            "model_used": self.model,
            "abs_url": paper_info.get('abs_url', '')
        }
    
    def format_output(self, result: Dict[str, Any], include_metadata: bool = True) -> str:
        """
        格式化输出结果
        
        Args:
            result: 生成结果字典
            include_metadata: 是否包含元数据
            
        Returns:
            格式化的字符串
        """
        if "error" in result:
            return f"错误: {result['error']}"
        
        output_parts = []
        
        if include_metadata:
            output_parts.append(f"# {result['title']}")
            output_parts.append(f"arXiv: {result['abs_url']}")
            output_parts.append(f"作者: {result['authors']}")
            output_parts.append("---")
            
            # 如果有概览图，添加图片
            if result.get('overview_image_relative'):
                image_path = result['overview_image_relative']
                output_parts.append(f"![概览图]({image_path})")
                output_parts.append("")
        
        output_parts.append(result['podcast_content'])
        
        return "\n\n".join(output_parts)


def main():
    """测试函数"""
    paper_info={
        "pdf_path": "./pdfs/2601.03252.pdf"
    }

    print("正在生成播客文章...")

    # 生成播客
    generator = PodcastGenerator(model="gpt-4.1-mini")
    result = generator.generate_podcast(paper_info)
    
    if "error" in result:
        print(f"生成失败: {result['error']}")
    else:
        output = generator.format_output(result)
        print("\n" + "="*50)
        print(output)


if __name__ == "__main__":
    main()
