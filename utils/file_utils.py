"""
文件处理工具函数
"""


def get_safe_model_name(model_name: str) -> str:
    """
    获取安全的模型名称（用于文件名）
    
    Args:
        model_name: 原始模型名称
        
    Returns:
        安全的文件名格式的字符串
    """
    # 替换文件名中不合法的字符
    safe_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_')
    # 移除其他可能不合法的字符
    safe_name = ''.join(c if c.isalnum() or c in ('_', '.') else '_' for c in safe_name)
    return safe_name
