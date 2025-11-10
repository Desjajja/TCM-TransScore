# TCM-TransScore：中医文献翻译质量评估框架

## 项目简介

TCM-TransScore 是一个专门用于中医文献英文翻译质量评估的综合框架，提供多维度、可量化的翻译质量分析。

### 评估维度

1. **术语准确性**（Terminology Accuracy）：基于中医术语库的精确匹配与上下文验证
2. **语义保真度**（Semantic Fidelity）：使用NLI模型进行双向语义蕴含检测
3. **语言质量**（Linguistic Quality）：通过OpenAI格式API或启发式规则评估语法流畅性
4. **结构完整性**（Structural Integrity）：段落和句子数量的对齐度分析

### 评估流程

```
源文本/参考译文/机器译文
         ↓
    术语提取与匹配
         ↓
    NLI语义一致性评估
         ↓
    语法流畅性检查
         ↓
    结构完整性分析
         ↓
    加权融合与诊断报告
```

---

## 环境需求与安装

### 1. 安装 uv 包管理器

#### macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 验证安装
```bash
uv --version
```

### 2. 安装项目

#### Python版本
- **要求**：Python 3.13
- **管理**：通过 uv 自动管理

#### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd TCM-TransScore
```

2. **安装所有依赖**
```bash
uv sync
```

**就是这样！** uv 会自动：
- 安装正确的 Python 版本（3.13）
- 创建虚拟环境
- 安装所有项目依赖（包括 PyTorch、SpaCy 模型等）
- 无需手动安装任何包

---

## 模型与API配置

### 1. NLI模型（本地调用）

- **模型名称**：`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **自动下载**：首次运行时自动从Hugging Face下载到本地缓存
- **缓存路径**：默认为 `~/.cache/huggingface/`
- **自定义缓存**：设置环境变量 `HF_HOME`

```bash
export HF_HOME=/your/custom/path
```

- **设备选择**：
  - 自动检测：优先使用CUDA GPU，否则使用CPU
  - 强制指定：设置环境变量 `FORCE_DEVICE=cpu` 或 `FORCE_DEVICE=cuda`

### 2. 语言模型API（OpenAI格式）

用于语法和流畅性评估，支持官方OpenAI API及兼容OpenAI格式的各类API网关。

#### 必需环境变量

```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 可选环境变量

```bash
# API基础URL（默认：https://api.openai.com/v1）
export OPENAI_BASE_URL="https://api.openai.com/v1"

# 使用的模型（默认：gpt-4o-mini）
export OPENAI_MODEL="gpt-4o-mini"
```

#### 配置示例

```bash
# 使用官方OpenAI
export OPENAI_API_KEY="sk-xxx"
export OPENAI_MODEL="gpt-4o-mini"

# 使用兼容网关
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-gateway.com/v1"
export OPENAI_MODEL="gpt-4"
```

**注意**：如果未配置API，语法检查将自动降级为启发式规则，评估权重会相应调整。

---
## 快速开始

### 1. 准备术语库

创建 JSON格式的术语库文件（例如 `terminology.json`）：

```json
{
  "Yin deficiency": "陰虧",
  "Yang deficiency": "阳虧",
  "Qi deficiency": "气虧",
  "liver depression": "肉郆",
  "spleen deficiency": "脚虧",
  "damp-heat": "湿热"
}
```

### 2. 运行示例

项目内置了演示示例，使用 uv 运行：

```bash
uv run main.py
```

运行后，评估结果会自动保存到 `data/evaluation_result.json`（目录会自动创建）。

### 3. 使用API进行评估
### 3. 使用API进行评估

```python
from main import TCMTransScoreEvaluator

# 初始化评估器
evaluator = TCMTransScoreEvaluator(
    terminology_file="terminology.json"
)

# 评估翻译质量
result = evaluator.evaluate(
    reference="参考译文...",
    machine_translation="机器译文...",
    source="源文本（可选）"
)

# 查看结果
print(f"总分: {result['overall_score']}")
print(f"各维度得分: {result['dimension_scores']}")
print(f"诊断信息: {result['diagnostics']}")
```

---

## 评分维度详解

### 术语准确性（权重：0.4）

**评分方法**：
- 提取参考译文和机器译文中的中医术语
- 计算术语F1分数（精确率和召回率的调和平均）
- 使用SpaCy验证共同术语的上下文相似度

**特殊情况处理**：
- 双方无术语：得分1.0
- 参考无术语但机器译文有术语：
  - 有源文本：使用源文本术语计算精度代理（×0.6）
  - 无源文本：该维度权重降为0

### 语义保真度（权重：0.35）

**评分方法**：
- 使用NLI模型进行双向蕴含检测
  - 前向：参考 → 机器译文
  - 后向：机器译文 → 参考
- 计算平均蕴含概率和矛盾概率
- 最终得分 = 平均蕴含 × (1 - 平均矛盾)

**输出范围**：0.0（完全矛盾）~ 1.0（完全蕴含）

### 语言质量（权重：0.15）

**评分方法**：
1. **优先方案**：使用语言模型API进行评估
   - 根据语法正确性和流畅性打分
   - 输出0.0-1.0的标准化分数

2. **降级方案**：启发式规则
   - 术语密度检查（理想范围5%-20%）
   - 基本格式问题检测（多余空格、缺失标点等）
   - 权重自动减半

### 结构完整性（权重：0.1）

**评分方法**：
- 段落数匹配度（权重0.7）
- 句子数匹配度（权重0.3）
- 空文本返回0.0

---

## 输出结果格式

### 保存位置

运行 `python main.py` 时，评估结果自动保存到：
```
data/evaluation_result.json
```

`data/` 目录会自动创建（如果不存在），评估结果以JSON格式保存。

### 结果结构

```json
{
  "overall_score": 0.756,
  "dimension_scores": {
    "terminology": 0.833,
    "semantic": 0.712,
    "linguistic": 0.650,
    "structural": 0.857
  },
  "active_weights": {
    "terminology": 0.400,
    "semantic": 0.350,
    "linguistic": 0.150,
    "structural": 0.100
  },
  "diagnostics": {
    "missing_terms": ["fire excess"],
    "extra_terms": ["internal heat"],
    "term_count": {
      "reference": 6,
      "machine": 5
    },
    "text_length": {
      "reference": 45,
      "machine": 42
    }
  }
}
```

---

## 常见问题

### 1. 首次加载NLI模型时间很长？

**原因**：模型需要从Hugging Face下载（约1.5GB）

**解决方案**：
- 耐心等待首次下载完成
- 后续运行会直接从缓存加载，速度很快
- 可提前运行一次进行预热

### 2. CUDA内存不足？

**解决方案**：
```bash
# 强制使用CPU
export FORCE_DEVICE=cpu
python main.py
```

### 3. API调用失败（429错误）？

**原因**：速率限制

**解决方案**：
- 程序内置了自动重试机制（最多3次）
- 遇到429时会指数退避重试
- 如果持续失败，会自动降级为启发式评估

### 4. SpaCy模型未找到？

**症状**：警告 "SpaCy模型未找到"

**影响**：术语上下文验证功能降级，不影响主要功能

**解决方案**：
```bash
python -m spacy download en_core_web_lg
```

### 5. 如何调整评分权重？

```python
evaluator = TCMTransScoreEvaluator(terminology_file="...")

# 自定义权重
evaluator.weights = {
    "terminology": 0.5,    # 增加术语权重
    "semantic": 0.3,
    "linguistic": 0.1,
    "structural": 0.1
}
```

---

## 性能建议

1. **首次运行预燭**
   ```bash
   # 预燭 NLI 模型
   uv run main.py
   ```

2. **批量评估优化**
   - 复用同一个evaluator实例
   - 避免重复加载模型

3. **大文本处理**
   - 当前版本对单个文本没有长度限制
   - NLI模型会自动截断到512 tokens
   - 建议分段评估超长文本

4. **GPU加速**
   - 使用CUDA可显著提升NLI推理速度
   - 推荐至少4GB显存

---

## 技术栈

- **NLI模型**：[MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
- **深度学习框架**：PyTorch + Transformers
- **句法分析**：SpaCy
- **API调用**：OpenAI Python SDK

---

## 许可证

本项目使用的依赖模型和库遵循各自的开源许可证：
- DeBERTa-v3-large-mnli：MIT License
- Transformers：Apache 2.0
- SpaCy：MIT License

---

## 更新日志

### v0.1.0（当前版本）
- ✅ 修正NLI模型路径为正式版本
- ✅ 实现OpenAI格式API集成用于语法评估
- ✅ 优化术语提取逻辑，支持大小写保留
- ✅ 修复NLI标签索引硬编码问题
- ✅ 增加完善的异常处理和降级机制
- ✅ 添加详细的中文注释和文档

---

## 贡献与反馈

如有问题或建议，欢迎提Issue或Pull Request。
