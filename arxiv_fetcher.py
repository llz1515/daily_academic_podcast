"""
arXiv 论文内容获取模块
从 arXiv URL 获取论文的标题、摘要和 PDF 文件
"""

import re
import requests
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
import os
import tempfile
import shutil

class ArxivFetcher:
    """arXiv 论文获取器"""
    
    def __init__(self, pdf_dir: str = None):
        """
        初始化获取器
        
        Args:
            pdf_dir: PDF 文件保存目录，默认为临时目录
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.pdf_dir = pdf_dir or tempfile.gettempdir()
        os.makedirs(self.pdf_dir, exist_ok=True)

    def _is_valid_pdf(self, pdf_path: str, min_size: int = 1024) -> bool:
        """简单校验 PDF 文件是否完整且非空"""
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
    
    def extract_arxiv_id(self, url: str) -> Optional[str]:
        """
        从 arXiv URL 中提取论文 ID
        支持格式：
        - https://arxiv.org/abs/2310.08560
        - https://arxiv.org/pdf/2310.08560
        - https://arxiv.org/abs/2310.08560v1
        """
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)',
            r'arxiv\.org/pdf/(\d+\.\d+(?:v\d+)?)',
            r'arxiv\.org/abs/([\w-]+/\d+(?:v\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def fetch_abstract_page(self, arxiv_id: str) -> Dict[str, Any]:
        """
        获取论文摘要页面信息
        返回：标题、作者、摘要、PDF链接等
        """
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        
        try:
            response = self.session.get(abs_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            return {"error": f"获取页面失败: {str(e)}"}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取标题
        title_elem = soup.find('h1', class_='title')
        if title_elem:
            title = title_elem.get_text().replace('Title:', '').strip()
        else:
            title_meta = soup.find('meta', {'name': 'citation_title'})
            title = title_meta['content'] if title_meta else "Unknown Title"
        
        # 提取作者
        authors_elem = soup.find('div', class_='authors')
        if authors_elem:
            authors = authors_elem.get_text().replace('Authors:', '').strip()
        else:
            author_metas = soup.find_all('meta', {'name': 'citation_author'})
            authors = ', '.join([m['content'] for m in author_metas])
        
        # 提取摘要
        abstract_elem = soup.find('blockquote', class_='abstract')
        if abstract_elem:
            abstract = abstract_elem.get_text().replace('Abstract:', '').strip()
        else:
            abstract_meta = soup.find('meta', {'name': 'citation_abstract'})
            abstract = abstract_meta['content'] if abstract_meta else ""
        
        # 提取分类
        subjects_elem = soup.find('span', class_='primary-subject')
        subjects = subjects_elem.get_text().strip() if subjects_elem else ""
        
        # PDF 链接
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "subjects": subjects,
            "pdf_url": pdf_url,
            "abs_url": abs_url
        }
    
    def download_pdf(self, arxiv_id: str) -> Optional[str]:
        """
        下载 PDF 文件到本地
        
        Args:
            arxiv_id: arXiv 论文 ID
            
        Returns:
            PDF 文件路径，失败返回 None
        """
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = os.path.join(self.pdf_dir, f"{arxiv_id.replace('/', '_')}.pdf")
        tmp_path = f"{pdf_path}.part"

        # 已有缓存文件先校验，坏文件先删除再重新下载
        if os.path.exists(pdf_path):
            if self._is_valid_pdf(pdf_path):
                return pdf_path
            try:
                os.remove(pdf_path)
            except OSError:
                pass

        try:
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            # 必须是 PDF 响应
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type:
                print(f"下载 PDF 失败: Content-Type 异常 {content_type}")
                return None

            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 下载完成后再校验，失败则删除
            if not self._is_valid_pdf(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                print("下载 PDF 失败: 文件校验不通过")
                return None

            shutil.move(tmp_path, pdf_path)
            return pdf_path
        except requests.RequestException as e:
            print(f"下载 PDF 失败: {str(e)}")
            return None
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    
    def fetch_paper(self, url: str) -> Dict[str, Any]:
        """
        获取论文完整信息，包括下载 PDF 文件
        
        Args:
            url: arXiv 论文 URL
            
        Returns:
            包含论文信息的字典，包括 pdf_path 字段
        """
        arxiv_id = self.extract_arxiv_id(url)
        if not arxiv_id:
            return {"error": f"无法解析 arXiv URL: {url}"}
        
        # 获取摘要页面信息
        paper_info = self.fetch_abstract_page(arxiv_id)
        if "error" in paper_info:
            return paper_info
        
        # 下载 PDF 文件
        pdf_path = self.download_pdf(arxiv_id)
        if pdf_path:
            paper_info["pdf_path"] = pdf_path
        else:
            paper_info["pdf_path"] = None
        
        return paper_info


def main():
    fetcher = ArxivFetcher(pdf_dir="./pdfs")
    
    test_url = "https://arxiv.org/abs/2510.26794"
    print(f"正在获取论文: {test_url}")
    
    paper = fetcher.fetch_paper(test_url)
    
    if "error" in paper:
        print(f"错误: {paper['error']}")
    else:
        print(f"标题: {paper['title']}")
        print(f"作者: {paper['authors']}")
        print(f"摘要: {paper['abstract'][:200]}...")
        print(f"PDF 路径: {paper.get('pdf_path', 'N/A')}")


if __name__ == "__main__":
    main()
