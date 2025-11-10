"""
TCM-TransScore：中医文献翻译质量评估框架
主要功能：
1. 术语准确性评估（基于术语库匹配）
2. 语义保真度评估（基于本地NLI模型）
3. 语言质量评估（基于OpenAI格式API）
4. 结构完整性评估（基于段落和句子对齐）
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import pipeline
import spacy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LMClient:
    """
    OpenAI格式API客户端，支持多用途模型调用
    
    用途：
    1. 语法质量评估（默认使用通用配置）
    2. 翻译理论评分（可指定critic模型）
    
    支持LiteLLM Proxy代理和直接API调用
    """
    
    def __init__(self, model_type="default"):
        """
        初始化OpenAI客户端，读取环境变量配置
        
        参数:
            model_type: 模型类型
                - "default": 默认配置（OPENAI_*环境变量）
                - "critic": 理论评分模型（CRITIC_*环境变量）
        """
        self.model_type = model_type
        
        # 根据模型类型选择不同的环境变量
        if model_type == "critic":
            # Critic模型配置（用于翻译理论评分）
            self.api_key = os.getenv("CRITIC_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("CRITIC_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            self.model = os.getenv("CRITIC_MODEL", "critic")  # LiteLLM proxy中的critic模型
            logger.info(f"初始化Critic模型客户端...")
        else:
            # 默认配置（用于语法检查等通用任务）
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.client = None
        self.available = False
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self.available = True
                logger.info(f"LMClient初始化成功 [类型: {model_type}, 模型: {self.model}]")
            except ImportError:
                logger.warning("未安装openai库，LM功能将不可用")
            except Exception as e:
                logger.warning(f"LMClient初始化失败: {e}")
        else:
            logger.info(f"未配置API密钥 [类型: {model_type}]，相关功能将不可用")
    
    def generate(self, 
                 prompt: str, 
                 system: str = "You are a helpful assistant.",
                 temperature: float = 0.3,
                 max_tokens: int = 100,
                 timeout: int = 30) -> Optional[str]:
        """
        调用语言模型生成响应
        
        参数:
            prompt: 用户提示词
            system: 系统提示词
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            timeout: 超时时间（秒）
        
        返回:
            生成的文本，失败时返回None
        """
        if not self.available or not self.client:
            return None
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 速率限制时增加延迟
                if "rate" in error_msg.lower() or "429" in error_msg:
                    time.sleep(retry_delay * (attempt + 1))
                elif attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None
        
        return None


class TCMTransScoreEvaluator:
    """中医翻译质量评估器主类"""
    
    def __init__(self, 
                 terminology_file: str,
                 nli_model_path: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                 device: Optional[str] = None):
        """
        初始化评估器
        
        参数:
            terminology_file: 中医术语库JSON文件路径，格式 {"term_en": "term_zh"}
            nli_model_path: NLI模型路径，本地调用
            device: 计算设备，None时自动选择
        """
        # 设备选择
        self.device = self._select_device(device)
        logger.info(f"使用计算设备: {self.device}")
        
        # 加载术语库
        self.terminology = {}
        self.term_pattern = None
        self._load_terminology(terminology_file)
        
        # 初始化NLI模型
        self.nli_pipeline = None
        self.label_indices = {}
        self._init_nli_model(nli_model_path)
        
        # 初始化语言模型客户端
        self.lm_client = LMClient()
        
        # 初始化句法分析器
        self.nlp_en = self._init_syntax_parser()
        
        # 评分权重配置
        self.weights = {
            "terminology": 0.4,
            "semantic": 0.35,
            "linguistic": 0.15,
            "structural": 0.1
        }
    
    def _select_device(self, device: Optional[str]) -> str:
        """
        选择计算设备
        
        优先级：环境变量FORCE_DEVICE > 参数device > CUDA自动探测 > CPU
        """
        force_device = os.getenv("FORCE_DEVICE")
        if force_device:
            return force_device
        
        if device:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
    
    def _load_terminology(self, file_path: str):
        """
        加载中医术语库并构建正则表达式模式
        
        术语库格式: {"term_en": "term_zh", ...}
        支持英文术语边界匹配，保留原始大小写
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.terminology = json.load(f)
            
            if not self.terminology:
                logger.warning("术语库为空，术语评估将不可用")
                return
            
            # 按长度降序排序，优先匹配长术语（避免"Qi deficiency"被"Qi"覆盖）
            sorted_terms = sorted(self.terminology.keys(), key=len, reverse=True)
            
            # 构建正则表达式：使用负向前瞻/后顾确保词边界
            # (?<![A-Za-z0-9_]) 确保术语前不是字母、数字或下划线
            # (?![A-Za-z0-9_]) 确保术语后不是字母、数字或下划线
            patterns = []
            for term in sorted_terms:
                # 转义特殊字符
                escaped_term = re.escape(term)
                # 添加边界控制
                pattern = f"(?<![A-Za-z0-9_]){escaped_term}(?![A-Za-z0-9_])"
                patterns.append(pattern)
            
            # 组合所有模式，不区分大小写
            combined_pattern = '|'.join(patterns)
            self.term_pattern = re.compile(combined_pattern, re.IGNORECASE)
            
            logger.info(f"成功加载 {len(self.terminology)} 个术语")
        
        except FileNotFoundError:
            logger.error(f"术语库文件未找到: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"术语库JSON格式错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载术语库时发生错误: {e}")
            raise
    
    def _init_nli_model(self, model_path: str):
        """
        初始化NLI（自然语言推理）模型
        
        使用transformers pipeline进行文本分类，支持文本对推理
        模型输出三个标签：ENTAILMENT（蕴含）、NEUTRAL（中立）、CONTRADICTION（矛盾）
        """
        try:
            logger.info(f"正在加载NLI模型: {model_path}")
            
            # 使用pipeline简化加载流程
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model_path,
                device=0 if self.device == "cuda" else -1,  # pipeline使用0表示cuda，-1表示cpu
                return_all_scores=True  # 返回所有标签的概率
            )
            
            # 动态解析标签索引（不同模型的标签顺序可能不同）
            # 获取模型配置中的id2label映射
            model = self.nli_pipeline.model
            if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                id2label = model.config.id2label
                
                # 查找各标签对应的索引
                for idx, label in id2label.items():
                    label_upper = label.upper()
                    if 'ENTAIL' in label_upper:
                        self.label_indices['entailment'] = idx
                    elif 'CONTRA' in label_upper:
                        self.label_indices['contradiction'] = idx
                    elif 'NEUTRAL' in label_upper:
                        self.label_indices['neutral'] = idx
                
                logger.info(f"标签索引映射: {self.label_indices}")
            else:
                logger.warning("无法获取标签映射，使用默认索引（可能不准确）")
                self.label_indices = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            
            logger.info("NLI模型加载成功")
        
        except Exception as e:
            logger.error(f"NLI模型加载失败: {e}")
            logger.error("请确保已安装transformers库，且网络连接正常")
            raise
    
    def _init_syntax_parser(self) -> Optional[object]:
        """
        初始化句法分析器（SpaCy）
        
        用于术语上下文相似度计算
        如果加载失败，相关功能将降级
        """
        try:
            nlp = spacy.load("en_core_web_lg")
            logger.info("SpaCy句法分析器加载成功")
            return nlp
        except OSError:
            logger.warning("SpaCy模型未找到，请运行: python -m spacy download en_core_web_lg")
            logger.warning("术语上下文验证功能将降级")
            return None
        except Exception as e:
            logger.warning(f"SpaCy加载失败: {e}")
            return None
    
    def extract_terminology(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取中医术语
        
        返回:
            包含原始形式和规范化形式的字典
            {
                "canonical": ["yin deficiency", "qi deficiency"],  # 小写规范形式
                "original": ["Yin deficiency", "Qi deficiency"]    # 原始大小写
            }
        """
        if not self.term_pattern:
            return {"canonical": [], "original": []}
        
        matches = self.term_pattern.findall(text)
        
        # 去重但保留原始大小写
        canonical_set = set()
        original_forms = []
        
        for match in matches:
            canonical = match.lower()
            if canonical not in canonical_set:
                canonical_set.add(canonical)
                original_forms.append(match)
        
        return {
            "canonical": list(canonical_set),
            "original": original_forms
        }
    
    def terminology_accuracy(self, ref: str, mt: str, source: str = "") -> Tuple[float, bool]:
        """
        计算术语准确性得分
        
        参数:
            ref: 参考翻译
            mt: 机器翻译
            source: 源文本（可选，用于特殊情况处理）
        
        返回:
            (得分, 是否需要降权)
            得分范围0-1，降权标志用于动态调整权重
        """
        ref_terms_data = self.extract_terminology(ref)
        mt_terms_data = self.extract_terminology(mt)
        
        ref_terms = set(ref_terms_data["canonical"])
        mt_terms = set(mt_terms_data["canonical"])
        
        # 情况1：参考翻译和机器翻译都没有术语
        if not ref_terms and not mt_terms:
            return 1.0, False  # 完美匹配
        
        # 情况2：参考翻译没有术语，但机器翻译有术语
        if not ref_terms and mt_terms:
            # 如果提供了源文本，使用源文本术语作为代理
            if source:
                source_terms = set(self.extract_terminology(source)["canonical"])
                if source_terms:
                    # 计算精度代理：机器翻译术语中有多少在源文本中
                    common_with_source = mt_terms & source_terms
                    precision_proxy = len(common_with_source) / len(mt_terms) if mt_terms else 0
                    # 降低权重惩罚（0.6倍）
                    return precision_proxy * 0.6, False
            
            # 无源文本或源文本也无术语：建议降权该维度
            return 0.5, True  # 返回中等分数，标记需要降权
        
        # 情况3：正常情况，计算F1分数
        common = ref_terms & mt_terms
        
        if not common:
            return 0.0, False
        
        precision = len(common) / len(mt_terms) if mt_terms else 0
        recall = len(common) / len(ref_terms) if ref_terms else 0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        # 如果SpaCy可用，增加上下文相似度验证
        if self.nlp_en and common:
            context_score = self._validate_term_contexts(ref, mt, common)
            # 组合F1和上下文分数
            final_score = f1 * 0.7 + context_score * 0.3
        else:
            final_score = f1
        
        return final_score, False
    
    def _validate_term_contexts(self, ref: str, mt: str, terms: set) -> float:
        """
        验证共同术语在两个文本中的上下文相似度
        
        使用SpaCy的词向量计算上下文窗口的相似度
        """
        if not self.nlp_en:
            return 1.0
        
        scores = []
        for term in terms:
            ref_context = self._get_context_window(ref, term, window=50)
            mt_context = self._get_context_window(mt, term, window=50)
            
            if ref_context and mt_context:
                try:
                    ref_doc = self.nlp_en(ref_context)
                    mt_doc = self.nlp_en(mt_context)
                    
                    # 检查向量是否有效
                    if ref_doc.vector_norm > 0 and mt_doc.vector_norm > 0:
                        sim = ref_doc.similarity(mt_doc)
                        # 相似度范围是[-1, 1]，转换到[0, 1]
                        sim_normalized = (sim + 1) / 2
                        scores.append(sim_normalized)
                except Exception as e:
                    logger.debug(f"计算上下文相似度时出错: {e}")
                    continue
        
        return np.mean(scores) if scores else 1.0
    
    def _get_context_window(self, text: str, term: str, window: int = 50) -> str:
        """
        获取术语周围的上下文窗口（字符级）
        
        参数:
            text: 文本
            term: 术语
            window: 窗口大小（字符数）
        """
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return ""
        
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        
        return text[start:end]
    
    def semantic_fidelity(self, ref: str, mt: str) -> float:
        """
        计算语义保真度得分
        
        使用NLI模型进行双向蕴含检测：
        1. 参考 -> 机器翻译 的蕴含概率
        2. 机器翻译 -> 参考 的蕴含概率
        
        同时考虑矛盾概率作为惩罚项
        
        得分公式: average(entailment) * (1 - average(contradiction))
        """
        if not self.nli_pipeline:
            logger.warning("NLI模型不可用，返回默认分数")
            return 0.5
        
        try:
            # 双向NLI评估
            forward_probs = self._nli_inference(ref, mt)  # premise=参考, hypothesis=机器翻译
            backward_probs = self._nli_inference(mt, ref)  # premise=机器翻译, hypothesis=参考
            
            # 提取蕴含和矛盾概率
            ent_idx = self.label_indices.get('entailment', 0)
            contra_idx = self.label_indices.get('contradiction', 2)
            
            forward_ent = forward_probs[ent_idx]
            backward_ent = backward_probs[ent_idx]
            forward_contra = forward_probs[contra_idx]
            backward_contra = backward_probs[contra_idx]
            
            # 计算平均蕴含和矛盾概率
            avg_entailment = (forward_ent + backward_ent) / 2
            avg_contradiction = (forward_contra + backward_contra) / 2
            
            # 组合得分：蕴含越高越好，矛盾越低越好
            score = avg_entailment * (1 - avg_contradiction)
            
            logger.debug(f"NLI得分详情 - 蕴含: {avg_entailment:.3f}, 矛盾: {avg_contradiction:.3f}, 最终: {score:.3f}")
            
            return score
        
        except Exception as e:
            logger.error(f"NLI推理失败: {e}")
            return 0.5
    
    def _nli_inference(self, premise: str, hypothesis: str) -> List[float]:
        """
        执行NLI推理，返回三个标签的概率
        
        参数:
            premise: 前提文本
            hypothesis: 假设文本
        
        返回:
            [prob_label0, prob_label1, prob_label2]
        """
        # pipeline的输入格式
        input_text = f"{premise} [SEP] {hypothesis}"
        
        # 执行推理
        results = self.nli_pipeline(input_text, truncation=True, max_length=512)
        
        # results格式: [[{'label': 'LABEL_0', 'score': 0.x}, ...]]
        # 提取概率并按标签索引排序
        probs = [0.0, 0.0, 0.0]
        for item in results[0]:
            label = item['label']
            score = item['score']
            # 标签格式可能是 'LABEL_0', 'LABEL_1', 'LABEL_2' 或直接的标签名
            if label.startswith('LABEL_'):
                idx = int(label.split('_')[1])
                probs[idx] = score
            else:
                # 根据标签名映射到索引
                label_upper = label.upper()
                if 'ENTAIL' in label_upper:
                    idx = self.label_indices['entailment']
                elif 'CONTRA' in label_upper:
                    idx = self.label_indices['contradiction']
                elif 'NEUTRAL' in label_upper:
                    idx = self.label_indices['neutral']
                else:
                    continue
                probs[idx] = score
        
        return probs
    
    def linguistic_quality(self, mt: str) -> Tuple[float, bool]:
        """
        计算语言质量得分
        
        优先使用API进行语法和流畅性评估
        降级方案：启发式规则
        
        返回:
            (得分, 是否需要降权)
        """
        # 方案1：使用LM API进行评估
        if self.lm_client.available:
            score = self._lm_grammar_check(mt)
            if score is not None:
                return score, False
        
        # 方案2：启发式规则（降级）
        logger.debug("使用启发式规则进行语法评估")
        
        # 检查术语密度
        mt_terms = self.extract_terminology(mt)
        term_count = len(mt_terms["canonical"])
        word_count = len(mt.split())
        
        if word_count == 0:
            return 0.0, True  # 空文本
        
        term_ratio = term_count / word_count
        
        # 理想术语比例：5%-20%
        if 0.05 <= term_ratio <= 0.20:
            term_score = 1.0
        elif term_ratio < 0.05:
            # 术语太少
            term_score = term_ratio / 0.05
        else:
            # 术语太多
            term_score = max(0, 1 - (term_ratio - 0.20) / 0.30)
        
        # 检查基本格式问题
        format_score = 1.0
        if mt.count('  ') > word_count * 0.1:  # 过多的双空格
            format_score -= 0.2
        if not any(mt.endswith(p) for p in ['.', '!', '?', '。', '！', '？']):  # 缺少结束标点
            format_score -= 0.1
        
        format_score = max(0, format_score)
        
        # 组合得分
        heuristic_score = 0.6 * term_score + 0.4 * format_score
        
        # 建议降权，因为这是降级方案
        return heuristic_score, True
    
    def _lm_grammar_check(self, text: str) -> Optional[float]:
        """
        使用语言模型API进行语法和流畅性评估
        
        返回:
            0-1的分数，失败时返回None
        """
        system_prompt = "You are an expert in evaluating English text quality for medical translations."
        
        user_prompt = f"""Rate the grammatical correctness and fluency of the following English text on a scale of 0.0 to 1.0, where:
- 1.0 = Perfect grammar and natural fluency
- 0.7-0.9 = Minor grammatical issues but generally fluent
- 0.4-0.6 = Noticeable grammatical errors or awkward phrasing
- 0.0-0.3 = Severe grammatical problems

Text: "{text}"

Respond with ONLY a number between 0.0 and 1.0, no explanation."""
        
        response = self.lm_client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.3,
            max_tokens=10,
            timeout=15
        )
        
        if response:
            try:
                # 提取数字
                score = float(re.search(r'0?\.\d+|1\.0|[01]', response).group())
                score = max(0.0, min(1.0, score))  # 限制范围
                return score
            except (AttributeError, ValueError) as e:
                logger.warning(f"无法解析API响应: {response}, 错误: {e}")
                return None
        
        return None
    
    def structural_integrity(self, ref: str, mt: str) -> float:
        """
        计算结构完整性得分
        
        评估段落和句子数量的匹配程度
        """
        # 段落分割
        ref_paras = [p.strip() for p in ref.split('\n') if p.strip()]
        mt_paras = [p.strip() for p in mt.split('\n') if p.strip()]
        
        # 空文本处理
        if not ref_paras or not mt_paras:
            return 0.0
        
        # 段落数匹配度
        para_match = min(len(ref_paras), len(mt_paras)) / max(len(ref_paras), len(mt_paras))
        
        # 句子数匹配度（简单基于句号分割）
        ref_sents = self._split_sentences(ref)
        mt_sents = self._split_sentences(mt)
        
        if len(ref_sents) == 0 or len(mt_sents) == 0:
            sent_match = 0.0
        else:
            sent_match = min(len(ref_sents), len(mt_sents)) / max(len(ref_sents), len(mt_sents))
        
        # 组合得分
        score = 0.7 * para_match + 0.3 * sent_match
        
        return score
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        简单的句子分割
        
        基于常见句子结束标点
        """
        # 替换换行为空格
        text = text.replace('\n', ' ')
        
        # 基于句号、问号、感叹号分割
        sentences = re.split(r'[.!?。！？]+', text)
        
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def evaluate(self, reference: str, machine_translation: str, source: str = "") -> Dict:
        """
        评估机器翻译质量
        
        参数:
            reference: 参考翻译
            machine_translation: 机器翻译
            source: 源文本（可选）
        
        返回:
            评估结果字典，包含各维度得分和总分
        """
        # 1. 各维度评分
        term_score, term_downweight = self.terminology_accuracy(reference, machine_translation, source)
        sem_score = self.semantic_fidelity(reference, machine_translation)
        lang_score, lang_downweight = self.linguistic_quality(machine_translation)
        struct_score = self.structural_integrity(reference, machine_translation)
        
        # 2. 动态权重调整
        active_weights = self.weights.copy()
        
        if term_downweight:
            logger.info("术语维度权重降为0（参考翻译无术语且无源文本）")
            active_weights["terminology"] = 0.0
        
        if lang_downweight:
            logger.info("语言质量维度使用降级评估，权重减半")
            active_weights["linguistic"] *= 0.5
        
        # 重新归一化权重
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            for key in active_weights:
                active_weights[key] /= total_weight
        
        # 3. 加权融合
        raw_score = (
            active_weights["terminology"] * term_score +
            active_weights["semantic"] * sem_score +
            active_weights["linguistic"] * lang_score +
            active_weights["structural"] * struct_score
        )
        
        # 4. Sigmoid归一化（可选，这里保持原始加权分数）
        # final_score = self._sigmoid_normalize(raw_score)
        final_score = raw_score
        
        # 5. 生成诊断信息
        diagnostics = self._generate_diagnostics(reference, machine_translation)
        
        return {
            "overall_score": round(final_score, 3),
            "dimension_scores": {
                "terminology": round(term_score, 3),
                "semantic": round(sem_score, 3),
                "linguistic": round(lang_score, 3),
                "structural": round(struct_score, 3)
            },
            "active_weights": {k: round(v, 3) for k, v in active_weights.items()},
            "diagnostics": diagnostics
        }
    
    def _sigmoid_normalize(self, raw_score: float) -> float:
        """
        Sigmoid归一化（可选）
        
        将分数映射到更陡峭的S曲线上
        """
        k = 10  # 曲线陡峭度
        return 1 / (1 + np.exp(-k * (raw_score - 0.5)))
    
    def _generate_diagnostics(self, ref: str, mt: str) -> Dict:
        """
        生成诊断信息
        
        帮助用户理解评分结果，识别翻译问题
        """
        ref_terms_data = self.extract_terminology(ref)
        mt_terms_data = self.extract_terminology(mt)
        
        ref_terms = set(ref_terms_data["canonical"])
        mt_terms = set(mt_terms_data["canonical"])
        
        missing = list(ref_terms - mt_terms)
        extra = list(mt_terms - ref_terms)
        
        return {
            "missing_terms": missing[:10],  # 限制数量，避免输出过长
            "extra_terms": extra[:10],
            "term_count": {
                "reference": len(ref_terms),
                "machine": len(mt_terms)
            },
            "text_length": {
                "reference": len(ref.split()),
                "machine": len(mt.split())
            }
        }


# ========== 使用示例 ==========
if __name__ == "__main__":
    """
    使用示例：演示如何使用TCM-TransScore进行翻译质量评估
    """
    
    # 1. 准备术语库（实际使用时应加载完整术语库）
    demo_terminology = {
        "Yin deficiency": "阴虚",
        "Yang deficiency": "阳虚",
        "Qi deficiency": "气虚",
        "Blood deficiency": "血虚",
        "liver depression": "肝郁",
        "spleen deficiency": "脾虚",
        "kidney deficiency": "肾虚",
        "damp-heat": "湿热",
        "wind-cold": "风寒",
        "wind-heat": "风热",
        "fire excess": "火旺",
        "floating pulse": "浮脉",
        "deep pulse": "沉脉",
        "thin pulse": "细脉",
        "rapid pulse": "数脉",
        "slippery pulse": "滑脉",
        "red tongue": "红舌",
        "pale tongue": "淡舌",
        "syndrome": "证型"
    }
    
    # 2. 保存临时术语库
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(demo_terminology, f, ensure_ascii=False, indent=2)
        term_file = f.name
    
    try:
        # 3. 初始化评估器
        print("=" * 60)
        print("TCM-TransScore 翻译质量评估系统")
        print("=" * 60)
        
        evaluator = TCMTransScoreEvaluator(terminology_file=term_file)
        
        # 4. 准备测试文本
        reference = """
        The patient presents with a red tongue with little coating, thin and rapid pulse, 
        which belongs to the syndrome of Yin deficiency with fire excess. 
        Liver depression and spleen deficiency are also observed.
        The treatment principle is to nourish Yin and clear heat.
        """
        
        machine_translation = """
        The patient has a red tongue with scanty coating, thready and rapid pulse, 
        indicating a pattern of Yin deficiency with internal heat. 
        Symptoms of depressed liver and deficient spleen are also present.
        The therapeutic approach focuses on tonifying Yin and reducing heat.
        """
        
        source_text = """
        患者表现为舌红少苔，脉细数，
        属于阴虚火旺证。
        还观察到肝郁脾虚。
        治疗原则是滋阴清热。
        """
        
        # 5. 执行评估
        print("\n正在评估翻译质量...\n")
        result = evaluator.evaluate(
            reference=reference.strip(),
            machine_translation=machine_translation.strip(),
            source=source_text.strip()
        )
        
        # 6. 输出结果
        print("=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"\n总分: {result['overall_score']:.3f}")
        print("\n各维度得分:")
        for dim, score in result['dimension_scores'].items():
            dim_name = {
                "terminology": "术语准确性",
                "semantic": "语义保真度",
                "linguistic": "语言质量",
                "structural": "结构完整性"
            }.get(dim, dim)
            weight = result['active_weights'][dim]
            print(f"  {dim_name:12s}: {score:.3f} (权重: {weight:.3f})")
        
        print("\n诊断信息:")
        diag = result['diagnostics']
        if diag['missing_terms']:
            print(f"  缺失术语: {', '.join(diag['missing_terms'][:5])}")
        else:
            print("  缺失术语: 无")
        
        if diag['extra_terms']:
            print(f"  额外术语: {', '.join(diag['extra_terms'][:5])}")
        else:
            print("  额外术语: 无")
        
        print(f"  术语数量 - 参考: {diag['term_count']['reference']}, "
              f"机器翻译: {diag['term_count']['machine']}")
        print(f"  文本长度 - 参考: {diag['text_length']['reference']} 词, "
              f"机器翻译: {diag['text_length']['machine']} 词")
        
        # 导出结果为JSON
        import os
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, "evaluation_result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")
        
        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)
    
    finally:
        # 7. 清理临时文件
        import os
        if os.path.exists(term_file):
            os.unlink(term_file)
