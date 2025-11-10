"""
翻译理论评分模块（Theoretical Evaluation Scorer）

基于翻译理论准则的LLM评分系统，提供：
1. 加载自定义评分准则文件
2. 构建带准则的评分提示词
3. 提取模型反馈和\\boxed{}包裹的分数
4. 整合到TCM-TransScore体系中
"""

import json
import logging
import os
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TheoreticalScorer:
    """
    翻译理论评分器
    
    根据用户提供的翻译理论准则，使用LLM对译文进行评分
    分数用\\boxed{}包裹，便于提取
    """
    
    def __init__(self, lm_client=None, criteria_file: Optional[str] = None):
        """
        初始化理论评分器
        
        参数:
            lm_client: LMClient实例，用于API调用
                      如果为None，会自动创建critic模型客户端
            criteria_file: 评分准则文件路径（文本或JSON格式）
        """
        # 如果未提供lm_client，创建专用的critic模型客户端
        if lm_client is None:
            from main import LMClient
            self.lm_client = LMClient(model_type="critic")
            logger.info("使用专用的Critic模型进行理论评分")
        else:
            self.lm_client = lm_client
        self.criteria = ""
        self.criteria_name = "翻译理论评分"
        
        if criteria_file and os.path.exists(criteria_file):
            self._load_criteria(criteria_file)
        else:
            logger.info("未提供评分准则文件，使用默认准则")
            self._load_default_criteria()
    
    def _load_criteria(self, file_path: str):
        """
        加载评分准则文件
        
        支持格式：
        - .txt: 纯文本准则
        - .json: JSON格式，包含name和criteria字段
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    self.criteria = data.get('criteria', '')
                    self.criteria_name = data.get('name', '翻译理论评分')
                else:
                    # 纯文本格式
                    self.criteria = f.read().strip()
            
            logger.info(f"成功加载评分准则: {self.criteria_name}")
            logger.debug(f"准则内容长度: {len(self.criteria)} 字符")
        
        except Exception as e:
            logger.error(f"加载评分准则失败: {e}")
            logger.info("使用默认准则")
            self._load_default_criteria()
    
    def _load_default_criteria(self):
        """加载默认评分准则"""
        self.criteria_name = "奈达功能对等理论评分"
        self.criteria = """
## 评分准则：奈达功能对等理论（Nida's Functional Equivalence Theory）

根据奈达的功能对等理论，优秀的翻译应达到以下标准：

### 1. 语义对等（Semantic Equivalence）- 40%
- 译文是否准确传达了原文的核心意义
- 专业术语和概念是否得到正确转换
- 是否避免了语义失真或遗漏

### 2. 风格对等（Stylistic Equivalence）- 30%
- 译文语言风格是否与原文一致（学术性、正式性）
- 句式结构是否符合目标语言习惯
- 专业性与可读性的平衡

### 3. 文化对等（Cultural Equivalence）- 20%
- 文化特定概念（如中医理论）是否得到恰当处理
- 是否保留了必要的文化信息
- 目标读者是否能理解文化内涵

### 4. 交际效果对等（Communicative Effect）- 10%
- 译文是否能在目标读者中产生与原文相似的效果
- 信息传递是否清晰有效

评分范围：0-10分
- 9-10分：优秀，各方面均达到高度对等
- 7-8分：良好，主要方面对等，有小瑕疵
- 5-6分：及格，基本对等但有明显不足
- 3-4分：较差，对等性明显不足
- 0-2分：很差，严重偏离对等原则
"""
    
    def evaluate(self, 
                 source: str, 
                 translation: str, 
                 reference: Optional[str] = None) -> Dict:
        """
        执行理论评分
        
        参数:
            source: 源文本
            translation: 待评估的译文
            reference: 参考译文（可选）
        
        返回:
            {
                "score": 8.5,  # 提取的分数
                "feedback": "详细的评价反馈...",
                "raw_response": "模型的完整响应"
            }
        """
        if not self.lm_client or not self.lm_client.available:
            logger.warning("LM客户端不可用，无法执行理论评分")
            return {
                "score": None,
                "feedback": "LM客户端不可用",
                "raw_response": ""
            }
        
        # 构建评分提示词
        prompt = self._build_prompt(source, translation, reference)
        
        # 调用LLM
        response = self.lm_client.generate(
            prompt=prompt,
            system="你是一位精通翻译理论和中医翻译的专家评审。请根据给定的理论准则对译文进行客观、专业的评价。",
            temperature=0.3,
            max_tokens=800,
            timeout=30
        )
        
        if not response:
            logger.warning("LLM调用失败，无法获取理论评分")
            return {
                "score": None,
                "feedback": "LLM调用失败",
                "raw_response": ""
            }
        
        # 提取分数和反馈
        score, feedback = self._extract_score_and_feedback(response)
        
        return {
            "score": score,
            "feedback": feedback,
            "raw_response": response,
            "criteria_name": self.criteria_name
        }
    
    def _build_prompt(self, 
                      source: str, 
                      translation: str, 
                      reference: Optional[str] = None) -> str:
        """
        构建评分提示词
        
        将准则、源文本、译文拼接成完整的prompt
        """
        prompt_parts = [
            "# 任务：根据翻译理论准则评估译文质量",
            "",
            "## 评分准则",
            self.criteria,
            "",
            "## 源文本（中文）",
            f"```\n{source}\n```",
            "",
            "## 待评估译文（英文）",
            f"```\n{translation}\n```"
        ]
        
        # 如果有参考译文，添加到提示词中
        if reference and reference.strip():
            prompt_parts.extend([
                "",
                "## 参考译文（英文）",
                f"```\n{reference}\n```",
                "",
                "注：参考译文仅供参照，不代表标准答案。"
            ])
        
        # 添加输出要求
        prompt_parts.extend([
            "",
            "---",
            "",
            "## 评价要求",
            "",
            "1. **按照上述准则的各个维度**进行详细分析",
            "2. **指出译文的优点和不足**，给出具体例子",
            "3. **给出改进建议**（如有必要）",
            "4. **最后给出总分**，用 \\boxed{分数} 格式包裹，例如：\\boxed{8.5}",
            "",
            "请开始你的评价："
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_score_and_feedback(self, response: str) -> Tuple[Optional[float], str]:
        """
        从LLM响应中提取分数和反馈
        
        分数格式：\\boxed{8.5}
        反馈：去除\\boxed{}后的完整文本
        
        返回:
            (分数, 反馈文本)
        """
        # 提取\\boxed{}中的分数
        boxed_pattern = r'\\boxed\{([0-9]+\.?[0-9]*)\}'
        match = re.search(boxed_pattern, response)
        
        score = None
        if match:
            try:
                score = float(match.group(1))
                # 确保分数在合理范围内
                score = max(0.0, min(10.0, score))
                logger.info(f"提取到理论评分: {score}")
            except ValueError:
                logger.warning(f"无法解析分数: {match.group(1)}")
        else:
            logger.warning("未找到\\boxed{}格式的分数")
            # 尝试备用模式：查找数字/10
            alt_pattern = r'([0-9]+\.?[0-9]*)\s*/\s*10|总分[:：]\s*([0-9]+\.?[0-9]*)'
            alt_match = re.search(alt_pattern, response)
            if alt_match:
                try:
                    score_str = alt_match.group(1) or alt_match.group(2)
                    score = float(score_str)
                    score = max(0.0, min(10.0, score))
                    logger.info(f"使用备用模式提取到分数: {score}")
                except (ValueError, AttributeError):
                    pass
        
        # 提取反馈（移除\\boxed{}部分）
        feedback = re.sub(boxed_pattern, '', response).strip()
        
        return score, feedback


def integrate_scores(tcm_score_result: Dict, 
                     theoretical_score_result: Dict,
                     tcm_weight: float = 0.6,
                     theoretical_weight: float = 0.4) -> Dict:
    """
    整合TCM-TransScore和理论评分
    
    参数:
        tcm_score_result: TCM-TransScore的评估结果
        theoretical_score_result: 理论评分的结果
        tcm_weight: TCM-TransScore权重（默认0.6）
        theoretical_weight: 理论评分权重（默认0.4）
    
    返回:
        整合后的完整评估报告
    """
    # 提取TCM总分（范围0-1）
    tcm_overall = tcm_score_result.get('overall_score', 0)
    
    # 提取理论评分（范围0-10），归一化到0-1
    theoretical_raw = theoretical_score_result.get('score')
    theoretical_normalized = theoretical_raw / 10.0 if theoretical_raw is not None else None
    
    # 计算综合分数
    combined_score = None
    if theoretical_normalized is not None:
        combined_score = (
            tcm_weight * tcm_overall + 
            theoretical_weight * theoretical_normalized
        )
    else:
        # 如果理论评分不可用，仅使用TCM分数
        combined_score = tcm_overall
        logger.warning("理论评分不可用，综合分数仅基于TCM-TransScore")
    
    # 构建整合报告
    result = {
        "metadata": {
            "evaluation_framework": "TCM-TransScore + Theoretical Evaluation",
            "tcm_weight": tcm_weight,
            "theoretical_weight": theoretical_weight,
            "criteria_name": theoretical_score_result.get('criteria_name', '未命名准则')
        },
        "combined_score": round(combined_score, 3) if combined_score else None,
        "tcm_score": {
            "overall": round(tcm_overall, 3),
            "dimensions": tcm_score_result.get('dimension_scores', {}),
            "weights": tcm_score_result.get('active_weights', {}),
            "diagnostics": tcm_score_result.get('diagnostics', {})
        },
        "theoretical_score": {
            "score": round(theoretical_raw, 2) if theoretical_raw else None,
            "normalized_score": round(theoretical_normalized, 3) if theoretical_normalized else None,
            "feedback": theoretical_score_result.get('feedback', ''),
            "raw_response": theoretical_score_result.get('raw_response', '')
        }
    }
    
    return result


def save_evaluation_report(result: Dict, output_file: str):
    """
    保存评估报告为JSON文件
    
    参数:
        result: 整合后的评估结果
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存至: {output_file}")
    except Exception as e:
        logger.error(f"保存评估报告失败: {e}")


# ========== 使用示例 ==========
if __name__ == "__main__":
    """
    演示理论评分模块的使用方法
    """
    from main import TCMTransScoreEvaluator, LMClient
    import tempfile
    
    # 1. 准备测试数据
    source_text = """
    患者表现为舌红少苔，脉细数，
    属于阴虚火旺证。
    治疗原则是滋阴清热。
    """
    
    translation_text = """
    The patient presents with a red tongue with scanty coating and a thready, rapid pulse,
    indicating a pattern of Yin deficiency with fire excess.
    The treatment principle is to nourish Yin and clear heat.
    """
    
    reference_text = """
    The patient shows a red tongue with little coating and a thin, rapid pulse,
    which belongs to the syndrome of Yin deficiency with fire excess.
    The therapeutic principle is to tonify Yin and reduce heat.
    """
    
    # 2. 准备术语库
    demo_terminology = {
        "Yin deficiency": "阴虚",
        "fire excess": "火旺",
        "red tongue": "红舌",
        "rapid pulse": "数脉",
        "pattern": "证型",
        "syndrome": "证"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(demo_terminology, f, ensure_ascii=False)
        term_file = f.name
    
    # 3. 准备自定义评分准则（可选）
    custom_criteria = {
        "name": "Skopos理论评分",
        "criteria": """
## 评分准则：Skopos理论（目的论）

根据德国功能主义翻译理论，优秀的翻译应达到以下标准：

### 1. 目的性原则（Skopos Rule）- 40%
- 译文是否实现了预期的交际目的
- 是否满足目标读者的需求和期待
- 专业信息传递是否有效

### 2. 连贯性原则（Coherence Rule）- 30%
- 译文内部逻辑是否连贯
- 目标读者是否能理解
- 专业性与可读性的平衡

### 3. 忠实性原则（Fidelity Rule）- 30%
- 在满足目的前提下，是否忠实于源文本
- 专业内容是否准确传达
- 关键信息是否完整保留

评分范围：0-10分
"""
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(custom_criteria, f, ensure_ascii=False)
        criteria_file = f.name
    
    try:
        print("=" * 70)
        print("TCM-TransScore + 翻译理论评分 - 综合评估演示")
        print("=" * 70)
        
        # 4. 初始化TCM评估器
        print("\n[1/3] 初始化TCM-TransScore评估器...")
        tcm_evaluator = TCMTransScoreEvaluator(terminology_file=term_file)
        
        # 5. 执行TCM评估
        print("\n[2/3] 执行TCM-TransScore评估...")
        tcm_result = tcm_evaluator.evaluate(
            reference=reference_text.strip(),
            machine_translation=translation_text.strip(),
            source=source_text.strip()
        )
        
        print(f"TCM总分: {tcm_result['overall_score']}")
        
        # 6. 初始化理论评分器
        print("\n[3/3] 执行翻译理论评分...")
        theoretical_scorer = TheoreticalScorer(
            lm_client=tcm_evaluator.lm_client,
            criteria_file=criteria_file  # 使用自定义准则
        )
        
        # 7. 执行理论评分
        theoretical_result = theoretical_scorer.evaluate(
            source=source_text.strip(),
            translation=translation_text.strip(),
            reference=reference_text.strip()
        )
        
        if theoretical_result['score']:
            print(f"理论评分: {theoretical_result['score']}/10")
        else:
            print("理论评分: 不可用（需要配置OPENAI_API_KEY）")
        
        # 8. 整合结果
        print("\n" + "=" * 70)
        print("整合评估报告")
        print("=" * 70)
        
        integrated_result = integrate_scores(
            tcm_score_result=tcm_result,
            theoretical_score_result=theoretical_result,
            tcm_weight=0.6,
            theoretical_weight=0.4
        )
        
        # 9. 显示综合结果
        print(f"\n综合分数: {integrated_result['combined_score']}")
        print(f"  - TCM-TransScore: {integrated_result['tcm_score']['overall']} (权重: 60%)")
        if theoretical_result['score']:
            print(f"  - 理论评分: {theoretical_result['score']}/10 (权重: 40%)")
            print(f"\n理论评分反馈摘要:")
            feedback_preview = theoretical_result['feedback'][:200]
            print(f"  {feedback_preview}...")
        
        # 10. 保存报告
        output_file = "evaluation_report.json"
        save_evaluation_report(integrated_result, output_file)
        print(f"\n完整报告已保存至: {output_file}")
        
        print("\n" + "=" * 70)
        print("评估完成")
        print("=" * 70)
    
    finally:
        # 清理临时文件
        import os
        for f in [term_file, criteria_file]:
            if os.path.exists(f):
                os.unlink(f)
