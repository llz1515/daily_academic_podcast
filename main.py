#!/usr/bin/env python3
"""
arXiv 论文 Daily PodCast 自动化生成工具

将 arXiv 论文 URL 转换为播客风格的文章
直接将 PDF 文件传递给支持文件输入的 LLM（如 Gemini）
支持批量处理多篇论文
"""

import argparse
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from arxiv_fetcher import ArxivFetcher
from podcast_generator import PodcastGenerator
from paper_crawler import PaperCrawler
from pdf_image_extractor import PDFImageExtractor

import dotenv

dotenv.load_dotenv(override=True)

# 配置日志
def setup_logging(log_dir: str = "logs"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 移除默认的 handler
    logger.remove()
    
    # 添加控制台输出（带颜色，支持多线程，显示线程ID）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>T{thread.id}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level="INFO",
        colorize=True,
        enqueue=True,  # 线程安全
    )
    
    # 添加文件输出（无颜色，支持多线程，显示线程ID）
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | T{thread.id} | {name}:{function}:{line} | {message}",
        level="INFO",
        encoding="utf-8",
        enqueue=True,  # 线程安全
        rotation="100 MB",  # 日志文件大小限制
        retention="7 days",  # 保留7天的日志
        compression="zip",  # 压缩旧日志
    )
    
    return logger

class ArxivPodcastPipeline:
    """arXiv 论文播客生成流水线"""
    
    def __init__(
        self,
        output_dir: str,
        pdf_dir: str,
        model: str,
        custom_prompt: Optional[str],
        grounding_model: str,
        max_pages: int,
        max_workers: int
    ):
        """
        初始化流水线
        
        Args:
            output_dir: 输出目录
            pdf_dir: PDF 文件保存目录
            model: LLM 模型名称（需支持 PDF 文件输入）
            custom_prompt: 自定义 prompt
            grounding_model: 用于识别概览图的视觉模型
            max_pages: 搜索概览图的最大页数
            max_workers: 最大并行处理数量
        """
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.pdf_dir = pdf_dir
        self.custom_prompt = custom_prompt
        self.model = model
        
        self.fetcher = ArxivFetcher(pdf_dir=pdf_dir)
        self.generator = PodcastGenerator(model=model)
        image_output_dir = os.path.join(output_dir, "images")
        self.image_extractor = PDFImageExtractor(
            output_dir=image_output_dir,
            grounding_model=grounding_model
        )
        self.logger = logger
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)
    
    def process_single(self, url: str) -> Dict[str, Any]:
        """
        处理单篇论文
        
        Args:
            url: arXiv 论文 URL
            
        Returns:
            处理结果字典
        """
        self.logger.info(f"开始处理: {url}")
        
        # 获取论文信息并下载 PDF
        self.logger.info("正在获取论文信息并下载 PDF...")
        paper_info = self.fetcher.fetch_paper(url)
        
        if "error" in paper_info:
            self.logger.error(f"获取论文失败: {paper_info['error']}")
            return {"url": url, "error": paper_info['error']}
        
        self.logger.info(f"论文标题: {paper_info['title']}")
        pdf_path = paper_info.get('pdf_path')
        self.logger.info(f"PDF 路径: {pdf_path if pdf_path else 'N/A'}")
        
        # 提取概览图
        overview_image_path = None
        arxiv_id = paper_info.get('arxiv_id', '').replace('/', '_')
        if pdf_path and arxiv_id:
            try:     
                self.logger.info("正在提取论文概览图...")
                overview_image_path = self.image_extractor.extract_image(
                    pdf_path, 
                    arxiv_id,
                    max_pages=self.max_pages
                )
                if overview_image_path:
                    self.logger.info(f"成功提取概览图: {overview_image_path}")
                else:
                    self.logger.warning("未能提取概览图")
            except Exception as e:
                self.logger.warning(f"提取概览图时出错: {e}，继续处理")
        
        # 生成播客文章
        self.logger.info("正在生成播客文章...")
        result = self.generator.generate_podcast(paper_info, self.custom_prompt)
        
        if "error" in result:
            self.logger.error(f"生成播客失败: {result['error']}")
            return {"url": url, "error": result['error']}
        
        result["url"] = url
        # 添加概览图路径到结果中
        if overview_image_path:
            result["overview_image_path"] = overview_image_path
            # 计算相对路径用于 Markdown 显示
            result["overview_image_relative"] = os.path.relpath(overview_image_path, self.output_dir)
        
        self.logger.info("播客文章生成成功")
        
        # 处理完成后删除 PDF 文件
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                self.logger.info(f"已删除 PDF 文件: {pdf_path}")
            except Exception as e:
                self.logger.warning(f"删除 PDF 文件失败: {pdf_path}, 错误: {str(e)}")
        
        return result
    
    def process_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理多篇论文（并行处理）
        
        Args:
            urls: arXiv 论文 URL 列表
            
        Returns:
            处理结果列表
        """
        total = len(urls)
        self.logger.info(f"开始批量处理 {total} 篇论文（最大并行数: {self.max_workers}）")
        
        # 使用字典存储结果，key 为索引，value 为结果
        results_dict = {}
        completed_count = 0
        
        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.process_single, url): i 
                for i, url in enumerate(urls)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_dict[index] = result
                    completed_count += 1
                    
                    self.logger.info(f"[{completed_count}/{total}] 处理完成: {result.get('url', 'N/A')}")
                    
                    # 保存单篇结果
                    if "error" not in result:
                        self._save_single_result(result)
                        
                except Exception as e:
                    # 如果处理过程中出现异常，记录错误
                    url = urls[index]
                    error_result = {"url": url, "error": str(e)}
                    results_dict[index] = error_result
                    completed_count += 1
                    self.logger.error(f"[{completed_count}/{total}] 处理失败: {url}, 错误: {e}")
        
        # 按照原始顺序重新排列结果
        results = [results_dict[i] for i in range(total)]
        
        # 保存汇总结果
        self._save_summary(results)
        
        success_count = sum(1 for r in results if "error" not in r)
        self.logger.info(f"批量处理完成: {success_count}/{total} 成功")
        
        return results
    
    def _get_safe_model_name(self) -> str:
        """获取安全的模型名称（用于文件名）"""
        # 替换文件名中不合法的字符
        model_name = self.model.replace('/', '_').replace('-', '_').replace(' ', '_')
        # 移除其他可能不合法的字符
        model_name = ''.join(c if c.isalnum() or c in ('_', '.') else '_' for c in model_name)
        return model_name
    
    def _save_single_result(self, result: Dict[str, Any]) -> str:
        """保存单篇播客文章"""
        arxiv_id = result.get('arxiv_id', 'unknown').replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self._get_safe_model_name()
        filename = f"podcast_{arxiv_id}_{timestamp}_{model_name}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        # 确保图片相对路径正确（相对于输出目录）
        if result.get('overview_image_path') and not result.get('overview_image_relative'):
            result['overview_image_relative'] = os.path.relpath(
                result['overview_image_path'], 
                self.output_dir
            )
        
        output = self.generator.format_output(result, include_metadata=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        
        self.logger.info(f"已保存: {filepath}")
        return filepath
    
    def _save_summary(self, results: List[Dict[str, Any]]) -> str:
        """保存批量处理汇总"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self._get_safe_model_name()
        
        # 保存 JSON 格式汇总
        json_path = os.path.join(self.output_dir, f"summary_{timestamp}_{model_name}.json")
        summary_data = {
            "timestamp": timestamp,
            "model": self.model,
            "total": len(results),
            "success": sum(1 for r in results if "error" not in r),
            "results": results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 保存 Markdown 格式汇总
        md_path = os.path.join(self.output_dir, f"daily_podcast_{timestamp}_{model_name}.md")
        md_content = self._generate_daily_digest(results)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"汇总已保存: {md_path}")
        return md_path
    
    def _generate_daily_digest(self, results: List[Dict[str, Any]]) -> str:
        """针对多篇文章生成每日播客汇总 markdown 文档"""
        date_str = datetime.now().strftime('%Y年%m月%d日')
        
        lines = [
            f"# 每日学术播客 - {date_str}",
            "",
            f"今日共收录 {len(results)} 篇论文",
            "",
            "---",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                lines.append(f"## {i}. 处理失败")
                lines.append(f"URL: {result.get('url', 'N/A')}")
                lines.append(f"错误: {result['error']}")
            else:
                lines.append(f"## {i}. {result.get('title', 'Unknown')}")
                lines.append(f"arXiv: {result.get('abs_url', 'N/A')}")
                lines.append(f"作者: {result.get('authors', 'N/A')}")
                lines.append("")
                
                # 如果有概览图，添加图片
                if result.get('overview_image_relative'):
                    image_path = result['overview_image_relative']
                    lines.append(f"![概览图]({image_path})")
                    lines.append("")
                
                lines.append(result.get('podcast_content', ''))
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)


def load_urls_from_file(filepath: str) -> List[str]:
    """从文件加载 URL 列表"""
    urls = []
    if not os.path.exists(filepath):
        return urls
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls


def get_huggingface_daily_papers() -> List[str]:
    """从 Huggingface 获取每日论文列表"""
    try:
        crawler = PaperCrawler()
        paper_list = crawler.get_Huggingface_Daily_Paper_list()
        return paper_list
    except Exception as e:
        print(f"获取 Huggingface 每日论文失败: {str(e)}")
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='arXiv 论文 Daily PodCast 自动化生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            输入方式（至少选择一种，可多选）:
            -f, --file:     从文件读取 URL 列表
            -d, --daily:    从 Huggingface 爬取每日论文
            -a, --arxiv:    直接输入 arXiv 论文链接

            示例用法:
            # 从 Huggingface 爬取每日论文
            python main.py -d
            
            # 从文件读取 URL 列表
            python main.py -f urls.txt
            
            # 直接输入 arXiv 链接
            python main.py -a https://arxiv.org/abs/2310.08560
            
            # 组合使用：爬取每日论文 + 文件中的额外论文
            python main.py -d -f urls.txt
            
            # 组合使用：爬取每日论文 + 直接输入的链接
            python main.py -d -a https://arxiv.org/abs/2310.08560 https://arxiv.org/abs/2312.00752
            
            # 三种方式组合使用
            python main.py -d -f urls.txt -a https://arxiv.org/abs/2310.08560
            
            # 限制每日论文数量（保留前 5 篇）
            python main.py -d --daily-limit 7

            # 调整最大并行处理数量
            python main.py -d --max-workers 10
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='从文件读取 URL 列表（每行一个 URL）'
    )
    
    parser.add_argument(
        '-d', '--daily',
        action='store_true',
        help='从 Huggingface 爬取每日论文'
    )
    
    parser.add_argument(
        '-a', '--arxiv',
        nargs='+',
        help='直接输入 arXiv 论文链接（可输入多个）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='输出目录（默认: output）'
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='pdfs',
        help='PDF 文件保存目录（默认: pdfs）'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4.1-mini',
        help='LLM 模型名称，需支持 PDF 文件输入（默认: gpt-4.1-mini）'
    )
    
    parser.add_argument(
        '-p', '--prompt-file',
        type=str,
        help='自定义 prompt 文件路径'
    )
    
    parser.add_argument(
        '--daily-limit',
        type=int,
        default=5,
        help='保留 Huggingface 每日论文的前 N 篇（默认: 5）'
    )
    
    parser.add_argument(
        '--grounding-model',
        type=str,
        default='gpt-4.1-mini',
        help='用于识别概览图的视觉模型（默认: gpt-4.1-mini）'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=5,
        help='搜索概览图的最大页数（默认: 5）'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='最大并行处理数量（默认: 5）'
    )
    
    args = parser.parse_args()
    
    # 先设置日志（以便后续操作可以记录日志）
    logger = setup_logging()
    
    # 检查是否至少选择了一种输入方式
    has_file_input = args.file is not None
    has_daily_input = args.daily
    has_arxiv_input = args.arxiv is not None and len(args.arxiv) > 0
    
    if not (has_file_input or has_daily_input or has_arxiv_input):
        parser.print_help()
        logger.error("未选择任何输入方式")
        print("\n错误: 请至少选择一种输入方式（-f, -d, 或 -a）")
        sys.exit(1)
    
    # 收集 URL
    urls = []
    
    # 方式1: 从文件读取
    if has_file_input:
        if os.path.exists(args.file):
            file_urls = load_urls_from_file(args.file)
            urls.extend(file_urls)
            if file_urls:
                logger.info(f"从文件 {args.file} 读取到 {len(file_urls)} 个 URL")
                print(f"从文件 {args.file} 读取到 {len(file_urls)} 个 URL")
            else:
                logger.warning(f"文件 {args.file} 为空或未包含有效 URL")
                print(f"警告: 文件 {args.file} 为空或未包含有效 URL")
        else:
            logger.error(f"文件不存在: {args.file}")
            print(f"错误: 文件不存在 - {args.file}")
            sys.exit(1)
    
    # 方式2: 从 Huggingface 爬取每日论文
    if has_daily_input:
        logger.info("正在从 Huggingface 获取每日论文...")
        print("正在从 Huggingface 获取每日论文...")
        hf_papers = get_huggingface_daily_papers()
        if hf_papers:
            original_count = len(hf_papers)
            # 限制保留前 N 篇
            daily_limit = min(args.daily_limit, original_count)
            hf_papers = hf_papers[:daily_limit]
            logger.info(f"从 {original_count} 篇论文中限制保留前 {daily_limit} 篇")
            print(f"从 {original_count} 篇论文中限制保留前 {daily_limit} 篇")
            urls.extend(hf_papers)
            logger.info(f"从 Huggingface 获取到 {len(hf_papers)} 篇论文")
            print(f"从 Huggingface 获取到 {len(hf_papers)} 篇论文")
        else:
            logger.warning("未能从 Huggingface 获取到论文，请检查网络连接")
            print("警告: 未能从 Huggingface 获取到论文")
    
    # 方式3: 直接输入 arXiv 链接
    if has_arxiv_input:
        arxiv_urls = list(args.arxiv)
        urls.extend(arxiv_urls)
        logger.info(f"从命令行输入了 {len(arxiv_urls)} 个 arXiv 链接")
        print(f"从命令行输入了 {len(arxiv_urls)} 个 arXiv 链接")
    
    # 去重（保持顺序）
    urls = list(dict.fromkeys(urls))
    
    if not urls:
        logger.error("未找到任何可处理的 URL")
        print("\n错误: 所有输入方式均未获取到有效的 URL")
        if has_daily_input and not has_file_input and not has_arxiv_input:
            print("提示: Huggingface 爬取失败，请检查网络连接或使用其他输入方式（-f 或 -a）")
        elif has_file_input and not has_daily_input and not has_arxiv_input:
            print("提示: 文件为空或未包含有效 URL，请检查文件内容")
        elif has_arxiv_input and not has_daily_input and not has_file_input:
            print("提示: 输入的 arXiv 链接无效，请检查链接格式")
        sys.exit(1)
    
    logger.info(f"去重后，总共将处理 {len(urls)} 篇论文")
    print(f"\n去重后，总共将处理 {len(urls)} 篇论文")
    
    # 加载自定义 prompt
    custom_prompt = None
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            custom_prompt = f.read()
    
    # 创建流水线并处理
    pipeline = ArxivPodcastPipeline(
        output_dir=args.output,
        pdf_dir=args.pdf_dir,
        model=args.model,
        custom_prompt=custom_prompt,
        grounding_model=args.grounding_model,
        max_pages=args.max_pages,
        max_workers=args.max_workers
    )
    
    if len(urls) == 1:
        result = pipeline.process_single(urls[0])
        if "error" not in result:
            pipeline._save_single_result(result)
            output = pipeline.generator.format_output(result)
            print("\n" + "="*60)
            print(output)
    else:
        pipeline.process_batch(urls)


if __name__ == "__main__":
    main()
