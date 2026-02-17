# 推荐配置说明

经过实际测试验证，以下配置可以获得最佳效果：

## 推荐模型

```yaml
models:
  podcast_model: "gemini-3-pro-preview"
  grounding_model: "gemini-3-pro-preview"
```

## 推荐提示词

```yaml
prompt:
  custom_file: "prompts/gpt5_podcast_prompt.txt"
  grounding_file: "prompts/gpt5_grounding_prompt.txt"
```

## 完整配置示例

```yaml
# arXiv Daily Podcast 配置文件

input:
  daily:
    enabled: true
    limit: 5

output:
  dir: "output"
  pdf_dir: "pdfs"

models:
  podcast_model: "gemini-3-pro-preview"  # 最佳效果
  grounding_model: "gemini-3-pro-preview"  # 统一模型

prompt:
  custom_file: "prompts/gpt5_podcast_prompt.txt"
  grounding_file: "prompts/gpt5_grounding_prompt.txt"

performance:
  max_workers: 5
  max_pages: 5

logging:
  dir: "logs"
  level: "INFO"
```

## 测试结论

- **播客生成**: `gemini-3-pro-preview` 在内容质量、语言流畅度、格式规范等方面表现最佳
- **概览图识别**: `gemini-3-pro-preview` 在准确识别和精确定位方面表现稳定可靠
- **统一模型**: 使用相同模型处理两个任务可以获得更一致的效果

## 备用方案

如果需要专门的视觉识别能力，可以考虑：

```yaml
models:
  podcast_model: "gemini-3-pro-preview"
  grounding_model: "Qwen/Qwen3-VL-235B-A22B-Instruct"
```

## API 配置

确保在 `.env` 文件中配置了相应的 API 密钥：

```bash
# 用于播客生成和概览图识别
GEMINI_API_KEY=your-gemini-api-key-here

# 或者分别配置
GROUNDING_GEMINI_API_KEY=your-gemini-api-key-here
```
