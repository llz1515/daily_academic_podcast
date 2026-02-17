# Prompt 模板目录

本目录包含针对不同模型优化的提示词模板。

## 推荐配置

经过测试，推荐使用以下配置获得最佳效果：

```yaml
models:
  podcast_model: "gemini-3-pro-preview"
  grounding_model: "gemini-3-pro-preview"

prompt:
  custom_file: "prompts/gpt5_podcast_prompt.txt"
  grounding_file: "prompts/gpt5_grounding_prompt.txt"
```

**说明**：
- 虽然提示词文件名为 `gpt5_*`，但经测试这些提示词在 `gemini-3-pro-preview` 模型上效果最佳
- 提示词的设计原则适用于多种高级语言模型
- 建议使用相同模型处理播客生成和概览图识别，以获得一致的效果

## 文件说明

### GPT-5.2 优化提示词

#### 1. gpt5_podcast_prompt.txt
- **用途**：生成播客风格的学术文章
- **特点**：
  - 详细的三段式结构要求
  - 明确的字数控制（900字以内）
  - 完善的语言风格和格式规范
  - 中英文混排规则说明
  - 包含输出示例

#### 2. gpt5_grounding_prompt.txt
- **用途**：识别和定位论文概览图
- **特点**：
  - 详细的概览图识别标准
  - 精确的坐标系统说明
  - 边界框质量检查清单
  - 决策流程指导
  - 多场景示例

## 使用方法

在 `config.yml` 中配置：

```yaml
prompt:
  # 播客生成提示词
  custom_file: "prompts/gpt5_podcast_prompt.txt"

  # 概览图识别提示词
  grounding_file: "prompts/gpt5_grounding_prompt.txt"
```

## 自定义提示词

你可以：
1. 复制现有模板并修改
2. 创建全新的提示词文件
3. 针对不同模型优化提示词

创建自定义提示词后，在 `config.yml` 中指定文件路径即可使用。

## 提示词设计建议

### 播客生成提示词
- 明确输出格式和结构要求
- 指定字数限制和内容深度
- 说明语言风格（通俗易懂 vs 学术严谨）
- 提供格式规范（Markdown、中英文混排等）

### 概览图识别提示词
- 清晰定义目标图表的特征
- 详细说明坐标系统和输出格式
- 提供正反例说明
- 包含质量检查标准
