# 翻译理论评分模块使用指南

## 功能概述

TCM-TransScore v0.2.0 新增**翻译理论评分**（Theoretical Evaluation）模块，提供：

1. **双重评分体系**：
   - **TCM-TransScore**：客观指标评分（术语、语义、语法、结构）
   - **理论评分**：基于翻译理论的主观评分（由LLM专家评审）

2. **灵活的准则系统**：支持自定义翻译理论准则（奈达、Skopos等）

3. **LiteLLM代理支持**：通过代理统一管理待测模型（translator）和评审模型（critic）

---

## 快速开始

### 1. 使用LiteLLM Proxy（推荐）

#### Step 1: 安装LiteLLM

```bash
pip install litellm[proxy]
```

#### Step 2: 配置环境变量

```bash
# OpenAI API密钥（必需）
export OPENAI_API_KEY="sk-your-key-here"

# 如使用其他模型，配置相应API密钥
# export ANTHROPIC_API_KEY="sk-ant-xxx"
```

#### Step 3: 启动LiteLLM Proxy

```bash
# 使用项目提供的配置文件
litellm --config litellm_proxy.yaml --port 4000

# 或使用docker
docker run -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/litellm_proxy.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml --port 4000
```

#### Step 4: 配置环境变量指向Proxy

```bash
# Critic模型（理论评分）
export CRITIC_BASE_URL="http://localhost:4000"
export CRITIC_MODEL="critic"

# 可选：如果语法检查也使用proxy
export OPENAI_BASE_URL="http://localhost:4000"
export OPENAI_MODEL="gpt-4o-mini"
```

#### Step 5: 运行评估

```python
from main import TCMTransScoreEvaluator
from theoretical_scorer import TheoreticalScorer, integrate_scores, save_evaluation_report

# 初始化评估器
evaluator = TCMTransScoreEvaluator(terminology_file="terminology.json")

# TCM评分
tcm_result = evaluator.evaluate(
    reference="参考译文...",
    machine_translation="机器译文...",
    source="源文本..."
)

# 理论评分（自动使用critic模型）
scorer = TheoreticalScorer(criteria_file="criteria_example.json")
theoretical_result = scorer.evaluate(
    source="源文本...",
    translation="机器译文...",
    reference="参考译文..."
)

# 整合结果
final_result = integrate_scores(tcm_result, theoretical_result)
save_evaluation_report(final_result, "result.json")
```

---

## 配置文件详解

### litellm_proxy.yaml 结构

```yaml
model_list:
  # Translator：待测翻译模型
  - model_name: translator
    litellm_params:
      model: openai/gpt-4o-mini  # 可替换为任意待测模型
      api_key: os.environ/OPENAI_API_KEY
      temperature: 0.3
      max_tokens: 2000

  # Critic：理论评分模型
  - model_name: critic
    litellm_params:
      model: openai/gpt-4o  # 推荐使用更强的模型
      api_key: os.environ/OPENAI_API_KEY
      temperature: 0.3
      max_tokens: 1500
```

### 环境变量说明

| 变量名 | 用途 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | 通用API密钥 | - | 是 |
| `OPENAI_BASE_URL` | 通用API地址 | https://api.openai.com/v1 | 否 |
| `OPENAI_MODEL` | 语法检查模型 | gpt-4o-mini | 否 |
| `CRITIC_API_KEY` | Critic专用密钥 | 回退到OPENAI_API_KEY | 否 |
| `CRITIC_BASE_URL` | Critic专用地址 | 回退到OPENAI_BASE_URL | 否 |
| `CRITIC_MODEL` | Critic模型名 | critic | 否 |

---

## 自定义评分准则

项目支持三种格式的准则文件，均在**初始化时加载**，避免重复读取，保证性能。

### 格式1：JSON文件（推荐）

```json
{
  "name": "奈达功能对等理论评分",
  "criteria": "## 评分准则...\\n\\n### 1. 语义对等 - 40%\\n..."
}
```

优点：格式严格，易于程序化处理

### 格式2：YAML文件（推荐）

```yaml
name: "Skopos理论评分"

criteria: |
  ## 评分准则：Skopos理论
  
  根据德国功能主义翻译理论...
  
  ### 1. 目的性原则 - 40%
  - ...
```

优点：可读性更好，支持多行文本，不需转义换行符

**注意**：使用YAML格式需安装pyyaml：
```bash
pip install pyyaml
# 或
pip install -e ".[yaml]"
```

### 格式3：纯文本文件

```text
## 评分准则：Skopos理论

根据德国功能主义翻译理论...

### 1. 目的性原则 - 40%
- ...
```

优点：简单直接，但无法指定准则名称（使用默认名称）

### 使用自定义准则

```python
scorer = TheoreticalScorer(
    criteria_file="my_criteria.json"  # 或 my_criteria.txt
)
```

---

## 输出报告格式

完整的评估报告（`result.json`）包含两部分：

```json
{
  "metadata": {
    "evaluation_framework": "TCM-TransScore + Theoretical Evaluation",
    "tcm_weight": 0.6,
    "theoretical_weight": 0.4,
    "criteria_name": "奈达功能对等理论评分"
  },
  "combined_score": 0.756,
  "tcm_score": {
    "overall": 0.723,
    "dimensions": {
      "terminology": 0.833,
      "semantic": 0.712,
      "linguistic": 0.650,
      "structural": 0.857
    },
    "weights": {...},
    "diagnostics": {...}
  },
  "theoretical_score": {
    "score": 8.5,
    "normalized_score": 0.850,
    "feedback": "### 评价详情\n...",
    "raw_response": "完整的LLM响应..."
  }
}
```

### 分数说明

- **combined_score**: 综合分数（范围0-1）
  - 默认权重：TCM 60% + 理论 40%
  
- **tcm_score.overall**: TCM-TransScore总分（范围0-1）
  - 基于四个维度的加权平均
  
- **theoretical_score.score**: 理论评分原始分数（范围0-10）
  - 由LLM根据准则打分
  
- **theoretical_score.normalized_score**: 归一化理论分数（范围0-1）
  - score / 10

---

## 模型选择建议

### 场景1：开发测试

```yaml
# 经济型配置
translator: gpt-4o-mini
critic: gpt-4o-mini
```

估计成本：~$0.01/评估

### 场景2：日常评估

```yaml
# 标准配置
translator: 待测模型（如Claude、GPT等）
critic: gpt-4o
```

估计成本：~$0.05/评估

### 场景3：论文发表

```yaml
# 高精度配置
translator: 待测模型
critic: gpt-4o 或 claude-3-5-sonnet-20241022
```

估计成本：~$0.10/评估

---

## 常见问题

### Q1: 如何评估Claude模型的翻译质量？

修改 `litellm_proxy.yaml`:

```yaml
- model_name: translator
  litellm_params:
    model: anthropic/claude-3-5-sonnet-20241022
    api_key: os.environ/ANTHROPIC_API_KEY
```

然后正常运行评估即可。

### Q2: 如何对比多个模型？

为每个模型配置不同的 `model_name`:

```yaml
- model_name: translator-gpt4
  litellm_params:
    model: openai/gpt-4o-mini

- model_name: translator-claude
  litellm_params:
    model: anthropic/claude-3-5-sonnet-20241022
```

分别调用：

```python
# 设置环境变量指定模型
os.environ["TRANSLATOR_MODEL"] = "translator-gpt4"
# 运行评估...

os.environ["TRANSLATOR_MODEL"] = "translator-claude"
# 运行评估...
```

### Q3: Critic模型必须使用LiteLLM吗？

不是必须的。如果不使用LiteLLM，直接配置：

```bash
export CRITIC_BASE_URL="https://api.openai.com/v1"
export CRITIC_MODEL="gpt-4o"
export CRITIC_API_KEY="sk-xxx"
```

### Q4: 理论评分的准则如何制定？

建议参考经典翻译理论：

1. **奈达功能对等理论**：语义、风格、文化、交际效果
2. **Skopos理论**：目的性、连贯性、忠实性
3. **描述性翻译理论**：规范、习惯、可接受性
4. **关联理论**：认知环境、语境效果、处理努力

项目提供了奈达理论的默认准则（`criteria_example.json`）。

### Q5: 如何确保评分一致性？

1. 使用**较低的temperature**（0.1-0.3）
2. 使用**能力更强的critic模型**（gpt-4o, claude-3-opus等）
3. **多次评估取平均**（建议3-5次）
4. 在准则中**明确评分标准和范围**

---

## 高级用法

### 批量评估

```python
import json
from tqdm import tqdm

# 加载测试集
with open("test_set.json") as f:
    test_cases = json.load(f)

# 批量评估
results = []
for case in tqdm(test_cases):
    tcm_result = evaluator.evaluate(
        reference=case["reference"],
        machine_translation=case["translation"],
        source=case["source"]
    )
    
    theoretical_result = scorer.evaluate(
        source=case["source"],
        translation=case["translation"],
        reference=case["reference"]
    )
    
    integrated = integrate_scores(tcm_result, theoretical_result)
    integrated["case_id"] = case["id"]
    results.append(integrated)

# 保存所有结果
with open("batch_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 自定义权重

```python
# 调整TCM和理论评分的权重
integrated = integrate_scores(
    tcm_result, 
    theoretical_result,
    tcm_weight=0.7,        # TCM权重70%
    theoretical_weight=0.3  # 理论评分30%
)
```

### 仅使用理论评分

```python
# 不需要TCM评估器
scorer = TheoreticalScorer(criteria_file="criteria.json")

result = scorer.evaluate(
    source="源文本...",
    translation="译文...",
    reference="参考译文..."
)

print(f"理论评分: {result['score']}/10")
print(f"评价反馈:\n{result['feedback']}")
```

---

## 更新日志

### v0.2.0 (2025-01-10)
- ✅ 新增翻译理论评分模块
- ✅ 支持LiteLLM Proxy代理
- ✅ 支持自定义评分准则
- ✅ 双重评分整合与报告生成
- ✅ Translator/Critic模型分离

---

## 参考资料

- [LiteLLM文档](https://docs.litellm.ai/)
- [OpenAI API文档](https://platform.openai.com/docs/api-reference)
- [奈达功能对等理论](https://en.wikipedia.org/wiki/Dynamic_and_formal_equivalence)
- [Skopos理论](https://en.wikipedia.org/wiki/Skopos_theory)
