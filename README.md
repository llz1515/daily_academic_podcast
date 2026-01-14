# arXiv Daily Podcast 自动化生成器

这是一个 Python 项目，旨在自动化地将一篇或多篇 arXiv 论文转化为通俗易懂、适合播客的博客文章。项目通过下载论文 PDF 文件，并直接将 PDF 传递给支持文件输入的大型语言模型（如 GPT-4.1-mini、Gemini），生成结构化的三段式解读文章。

## 主要功能

- **PDF 直接输入**: 将论文 PDF 文件直接传递给 LLM，保留完整的论文结构、图表和公式信息。
- **自动获取论文**: 
  - 从 arXiv URL 自动抓取论文元信息并下载 PDF 文件
  - 从 Huggingface 每日论文 API 自动获取当日热门论文
  - 支持从文件读取 URL 列表
- **概览图提取**: 自动从 PDF 中提取论文概览图（Overview Figure）
- **高质量内容生成**: 调用支持文件输入的 LLM 生成符合特定格式和风格的播客文章
- **批量处理**: 支持一次性处理多个 URL，支持并行处理以提高效率
- **灵活配置**: 用户可以轻松更换 LLM 模型、自定义生成 Prompt、调整并行数量等
- **结构化输出**: 为每篇论文生成独立的 Markdown 文件，并为每次批量任务生成包含所有文章的汇总报告

## 项目结构

```
daily_academic_podcast/
├── main.py                    # 主程序入口，负责命令行解析和任务调度
├── arxiv_fetcher.py           # 模块：负责从 arXiv 获取论文数据和下载 PDF
├── podcast_generator.py       # 模块：负责调用 LLM 生成播客文章（PDF 直接输入）
├── paper_crawler.py           # 模块：负责从 Huggingface 爬取每日论文列表
├── pdf_image_extractor.py     # 模块：负责从 PDF 中提取概览图
├── requirements.txt           # Python 依赖包列表
├── utils/                     # 工具函数模块
│   ├── __init__.py
│   ├── file_utils.py
│   └── pdf_utils.py
├── pdfs/                      # PDF 文件保存目录（临时，处理完成后自动删除）
├── output/                    # 默认输出目录，存放生成的文章和图片
│   └── images/                # 提取的概览图保存目录
└── logs/                      # 日志目录，记录程序运行信息
```

## 安装与配置

### 1. 环境准备

确保你的环境中已安装 Python 3.8+ 和 pip。

### 2. 安装依赖

在项目根目录下，运行以下命令安装所需的 Python 库：

```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥

本项目通过 `openai` 库调用语言模型，使用 `python-dotenv` 从 `.env` 文件中读取 API 密钥。

请创建 `.env` 文件（项目根目录下），并设置相应的环境变量：

```bash
# OpenAI API 配置（或兼容 OpenAI API 的服务）
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_url_key_here  # 可选，如果使用第三方兼容服务
```

**注意**: `.env` 文件已添加到 `.gitignore`，不会被提交到版本控制系统。

## 使用方法

你可以通过 `main.py` 脚本来运行此项目。脚本支持多种输入方式，提供了灵活的使用方式。

### 命令行帮助

要查看所有可用的命令和选项，请运行：

```bash
python main.py --help
```

### 输入方式

项目支持三种输入方式，可以单独使用或组合使用：

1. **从 Huggingface 爬取每日论文** (`-d, --daily`)
2. **从文件读取 URL 列表** (`-f, --file`)
3. **直接输入 arXiv 链接** (`-a, --arxiv`)

### 示例

#### 1. 从 Huggingface 获取每日论文（推荐）

```bash
# 获取当日论文（默认保留前 5 篇）
python main.py -d

# 获取当日论文，保留前 7 篇
python main.py -d --daily-limit 7
```

#### 2. 从文件读取 URL 列表

创建一个文本文件（例如 `test_urls.txt`），每行包含一个 arXiv 链接：

```
https://arxiv.org/abs/2310.08560
https://arxiv.org/abs/2312.00752
```

然后运行：

```bash
python main.py -f test_urls.txt
```

#### 3. 直接输入 arXiv 链接

```bash
# 处理单篇论文
python main.py -a https://arxiv.org/abs/2310.08560

# 处理多篇论文
python main.py -a https://arxiv.org/abs/2310.08560 https://arxiv.org/abs/2312.00752
```

#### 4. 组合使用多种输入方式

```bash
# 组合使用：爬取每日论文 + 文件中的额外论文
python main.py -d -f test_urls.txt

# 组合使用：爬取每日论文 + 直接输入的链接
python main.py -d -a https://arxiv.org/abs/2310.08560

# 三种方式组合使用
python main.py -d -f test_urls.txt -a https://arxiv.org/abs/2310.08560
```

#### 5. 指定输出目录和模型

```bash
# 使用默认模型
python main.py -d -o ./my_podcasts -m gpt-4.1-mini

# 使用推荐模型组合（最佳效果）
python main.py -d -o ./my_podcasts -m gemini-3-pro-preview --grounding-model Qwen/Qwen3-VL-235B-A22B-Instruct
```

#### 6. 使用自定义 Prompt

```bash
python main.py -a https://arxiv.org/abs/2310.08560 -p my_prompt.txt
```

#### 7. 调整并行处理数量

```bash
# 使用 10 个并行线程处理
python main.py -d --max-workers 10
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `-d, --daily` | 从 Huggingface 爬取每日论文 |
| `-f, --file` | 从文件读取 URL 列表（每行一个 URL） |
| `-a, --arxiv` | 直接输入 arXiv 论文链接（可输入多个） |
| `-o, --output` | 输出目录（默认: `output`） |
| `--pdf-dir` | PDF 文件保存目录（默认: `pdfs`） |
| `-m, --model` | LLM 模型名称（默认: `gpt-4.1-mini`，可选: `gemini-3-pro-preview`, `gpt-4.1-mini`, `gpt-5.2`） |
| `-p, --prompt-file` | 自定义 prompt 文件路径 |
| `--daily-limit` | 保留 Huggingface 每日论文的前 N 篇（默认: 5） |
| `--grounding-model` | 用于识别概览图的视觉模型（默认: `gpt-4.1-mini`，可选: `gpt-4.1-mini`, `Qwen/Qwen3-VL-235B-A22B-Instruct`） |
| `--max-pages` | 搜索概览图的最大页数（默认: 5） |
| `--max-workers` | 最大并行处理数量（默认: 5） |

### 输出文件

- **单篇文章**: 在输出目录中，会为每篇成功处理的论文生成一个 `podcast_{arxiv_id}_{timestamp}_{model}.md` 文件，包含：
  - 论文标题、arXiv ID、作者信息
  - 提取的概览图（如果有）
  - 生成的播客文章内容

- **PDF 文件**: 下载的 PDF 文件保存在 `pdfs/` 目录中，处理完成后会自动删除以节省空间。

- **概览图**: 提取的概览图保存在 `output/images/` 目录中。

- **汇总报告**: 批量处理任务完成后，会生成两个汇总文件：
  - `daily_podcast_{timestamp}_{model}.md`: 一个 Markdown 文件，将所有生成的播客文章整合在一起。
  - `summary_{timestamp}_{model}.json`: 一个 JSON 文件，包含本次任务的详细信息（时间戳、模型、成功/失败数量、所有结果等）。

- **日志文件**: 运行日志保存在 `logs/` 目录中，文件名格式为 `podcast_{timestamp}.log`。

## 技术实现

### PDF 文件输入

本项目使用 OpenAI API 的文件输入功能，将 PDF 文件以 Base64 编码的方式直接传递给模型：

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "filename": "paper.pdf",
                    "file_data": f"data:application/pdf;base64,{base64_string}",
                }
            },
            {
                "type": "text",
                "text": "你的 prompt...",
            }
        ],
    },
]
```

这种方式相比提取文本有以下优势：

- 保留论文的完整结构和格式
- 图表、公式等视觉元素可被模型理解
- 无需依赖 PDF 文本提取工具

### 概览图提取

项目使用视觉模型（如 GPT-4.1-mini）自动识别 PDF 中的概览图（Overview Figure），通常位于论文的前几页。提取的图片会保存到输出目录，并在生成的 Markdown 文件中引用。

### 支持的模型

需要使用支持文件/多模态输入的模型，例如：

- `gpt-4.1-mini`（默认）
- `gemini-3-pro-preview`
- `gpt-5.2`
- 其他支持 PDF 输入的 OpenAI 兼容模型

### 推荐模型组合

为了获得最佳效果，推荐使用以下模型组合：

- **文本生成模型**: `gemini-3-pro-preview` - 用于生成播客文章内容
- **Grounding 模型**: `Qwen/Qwen3-VL-235B-A22B-Instruct` - 用于识别和提取论文概览图

使用推荐组合的示例命令：

```bash
python main.py -d -m gemini-3-pro-preview --grounding-model Qwen/Qwen3-VL-235B-A22B-Instruct
```

### 并行处理

项目使用 `ThreadPoolExecutor` 实现并行处理，可以同时处理多篇论文，大大提高处理效率。默认并行数为 5，可通过 `--max-workers` 参数调整。

## 自定义

### 修改 Prompt

默认的生成 Prompt 定义在 `podcast_generator.py` 文件的 `PODCAST_PROMPT` 变量中。你可以直接修改此字符串以调整生成文章的风格、结构或要求。

默认 Prompt 要求生成三段式文章：

1. **问题背景**（约 200 字）：包括当前该领域的方法、这种方法会造成什么问题、本文动机等内容
2. **核心贡献与关键技术**（约 400 字）：介绍论文提出的新方法、框架、模型或数据集，以及解决方案的核心创新点、架构设计或训练策略
3. **实验结果与意义**（约 300 字）：简略的实验设置、重点说结论的实验结果（辅以小部分重要数据）、该工作的意义与影响

总字数严格控制在 900 字以内。

你也可以通过 `-p` 参数指定自定义的 prompt 文件。

### 更换模型

通过 `-m` 参数指定模型名称。确保所选模型支持 PDF 文件输入。

### 调整并行数量

根据你的 API 配额和网络情况，可以通过 `--max-workers` 参数调整并行处理数量。建议值：
- 小规模（1-5 篇）：3-5
- 中规模（5-10 篇）：5-8
- 大规模（10+ 篇）：8-10

## 注意事项

1. **文件大小限制**: 单个 PDF 文件不应超过模型支持的最大文件大小。如果文件过大，程序会自动尝试压缩后重试。

2. **API 配额**: 由于 PDF 文件会被转换为图像和文本，token 消耗可能较高。请确保你的 API 账户有足够的配额。

3. **网络要求**: 
   - 需要能够访问 arXiv 网站下载论文
   - 需要能够访问 Huggingface API 获取每日论文列表
   - 需要能够访问你配置的 LLM API 服务

4. **PDF 处理**: 处理完成后，PDF 文件会自动删除以节省空间。如果需要保留 PDF，可以修改 `main.py` 中的相关代码。

5. **日志记录**: 所有操作都会记录到日志文件中，方便排查问题。日志文件会自动轮转和压缩。

## 故障排除

### 常见问题

1. **API 密钥错误**: 确保 `.env` 文件中的 `OPENAI_API_KEY` 设置正确。

2. **网络连接失败**: 检查网络连接，确保可以访问 arXiv、Huggingface 和 API 服务。

3. **PDF 下载失败**: 检查 arXiv URL 是否正确，论文是否存在。

4. **模型不支持文件输入**: 确保使用的模型支持 PDF 文件输入功能。

5. **并行处理导致 API 限流**: 减少 `--max-workers` 的值，或增加 API 请求间隔。

### 查看日志

所有运行日志都保存在 `logs/` 目录中，可以通过查看日志文件来诊断问题：

```bash
# 查看最新的日志文件
ls -lt logs/ | head -1
```
