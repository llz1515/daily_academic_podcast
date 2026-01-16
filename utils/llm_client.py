"""
LLM API 客户端封装模块
支持 OpenAI 和 Gemini 两种 API，根据环境变量自动选择
"""

import os
import base64
from typing import Dict, Any, Optional
from enum import Enum
from loguru import logger

import dotenv

dotenv.load_dotenv(override=True)


class LLMProvider(Enum):
    """LLM 提供商枚举"""
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMClient:
    """统一的 LLM 客户端接口"""
    
    def __init__(self, provider: LLMProvider, api_key: str, base_url: Optional[str] = None):
        """
        初始化 LLM 客户端
        
        Args:
            provider: LLM 提供商
            api_key: API 密钥
            base_url: API 基础 URL（可选，用于兼容服务）
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        
        # 根据提供商初始化对应的客户端
        if provider == LLMProvider.OPENAI:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        elif provider == LLMProvider.GEMINI:
            from google import genai
            import google.genai.types as types
            if base_url:
                self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
                    base_url= base_url
                ))
            else:
                self.client = genai.Client(api_key=api_key)
    
    def chat_with_image(
        self,
        model: str,
        prompt: str,
        image_base64: str,
        image_mime: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带图片输入的对话接口
        
        Args:
            model: 模型名称
            prompt: 提示词
            image_base64: base64 编码的图片字符串（不含 data URI 前缀）
            image_mime: 图片 MIME 类型，如 "image/png" 或 "image/jpeg"
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Returns:
            包含响应内容的字典，失败时包含 error 字段
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._openai_chat_with_image(
                    model=model,
                    prompt=prompt,
                    image_base64=image_base64,
                    image_mime=image_mime,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            elif self.provider == LLMProvider.GEMINI:
                return self._gemini_chat_with_image(
                    model=model,
                    prompt=prompt,
                    image_base64=image_base64,
                    image_mime=image_mime,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM 图片 API 调用失败: {error_msg}")
            return {"error": f"LLM 图片 API 调用失败: {error_msg}", "error_msg": error_msg}
    
    def chat_with_file(
        self,
        model: str,
        prompt: str,
        file_path: str,
        file_mime: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带文件输入的对话接口（用于 PDF 等文件）
        
        Args:
            model: 模型名称
            prompt: 提示词
            file_path: 文件路径
            file_mime: 文件 MIME 类型，如 "application/pdf"
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Returns:
            包含响应内容的字典，失败时包含 error 字段
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._openai_chat_with_file(
                    model=model,
                    prompt=prompt,
                    file_path=file_path,
                    file_mime=file_mime,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            elif self.provider == LLMProvider.GEMINI:
                return self._gemini_chat_with_file(
                    model=model,
                    prompt=prompt,
                    file_path=file_path,
                    file_mime=file_mime,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM 文件 API 调用失败: {error_msg}")
            return {"error": f"LLM 文件 API 调用失败: {error_msg}", "error_msg": error_msg}
    
    def _openai_chat_with_image(
        self,
        model: str,
        prompt: str,
        image_base64: str,
        image_mime: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI 图片输入实现"""
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        content = response.choices[0].message.content
        return {"content": content}
    
    def _gemini_chat_with_image(
        self,
        model: str,
        prompt: str,
        image_base64: str,
        image_mime: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Gemini 图片输入实现"""
        from google.genai import types
        
        # 解码 base64 字符串为 bytes
        image_data = base64.b64decode(image_base64)
        
        # 使用 types.Part.from_bytes 创建图片部分
        image_part = types.Part.from_bytes(data=image_data, mime_type=image_mime)
        
        # 构建生成配置参数
        config_kwargs = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        
        all_kwargs = {**config_kwargs, **kwargs}
        
        # 调用 API
        response = self.client.models.generate_content(
            model=model,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                **all_kwargs
            )
        )
        
        content = response.text
        return {"content": content}
    
    def _openai_chat_with_file(
        self,
        model: str,
        prompt: str,
        file_path: str,
        file_mime: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI 文件输入实现"""
        from utils.pdf_utils import encode_pdf_to_base64
        import os
        
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}
        
        # 将文件编码为 Base64
        if file_mime == "application/pdf":
            base64_string = encode_pdf_to_base64(file_path)
        else:
            with open(file_path, "rb") as f:
                file_data = f.read()
            base64_string = base64.b64encode(file_data).decode('utf-8')
        
        if not base64_string:
            return {"error": f"无法读取文件: {file_path}"}
        
        filename = os.path.basename(file_path)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": f"data:{file_mime};base64,{base64_string}",
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        content = response.choices[0].message.content
        return {"content": content}
    
    def _gemini_chat_with_file(
        self,
        model: str,
        prompt: str,
        file_path: str,
        file_mime: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Gemini 文件输入实现"""
        from google.genai import types
        import os
        
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}
        
        # 读取文件 bytes
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # 使用 types.Part.from_bytes 创建文件部分
        file_part = types.Part.from_bytes(data=file_data, mime_type=file_mime)
        
        # 构建生成配置参数
        config_kwargs = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        
        all_kwargs = {**config_kwargs, **kwargs}
        
        # 调用 API
        response = self.client.models.generate_content(
            model=model,
            contents=[file_part, prompt],
            config=types.GenerateContentConfig(
                **all_kwargs
            )
        )
        
        content = response.text
        return {"content": content}

def create_llm_client(is_grounding: bool = False) -> LLMClient:
    """
    根据环境变量创建 LLM 客户端
    
    Args:
        is_grounding: 是否为 grounding 模型
                    False 表示使用 OPENAI_API_KEY 或 GEMINI_API_KEY
                    True 表示使用 GROUNDING_OPENAI_API_KEY 或 GROUNDING_GEMINI_API_KEY
    
    Returns:
        LLMClient 实例
        
    Raises:
        ValueError: 如果未配置任何 API key
    """
    if is_grounding:
        prefix = "GROUNDING_"
    else:
        prefix = ""

    # 构建环境变量名称
    openai_key_name = f"{prefix}OPENAI_API_KEY"
    gemini_key_name = f"{prefix}GEMINI_API_KEY"
    
    # 获取 API keys
    openai_key = os.getenv(openai_key_name)
    gemini_key = os.getenv(gemini_key_name)
    
    # 判断使用哪个 API
    if openai_key:
        logger.info(f"使用 OpenAI API (key: {openai_key_name})")
        base_url = os.getenv(f"{prefix}OPENAI_BASE_URL")
        return LLMClient(provider=LLMProvider.OPENAI, api_key=openai_key, base_url=base_url)
    elif gemini_key:
        logger.info(f"使用 Gemini API (key: {gemini_key_name})")
        base_url = os.getenv(f"{prefix}GEMINI_BASE_URL")
        return LLMClient(provider=LLMProvider.GEMINI, api_key=gemini_key, base_url=base_url)
    else:
        error_msg = f"未配置 API key。请设置 {openai_key_name} 或 {gemini_key_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)