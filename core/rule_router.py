"""RuleRouter — 15维规则评分引擎（ClawRouter 风格）。

纯正则匹配 + 计数，无 LLM 调用，<1ms 完成分类。
基于规则评分将消息归入四档：SIMPLE / MEDIUM / COMPLEX / REASONING。
"""

import re
from dataclasses import dataclass
from typing import List, Optional

_log = __import__("logging").getLogger(__name__)


# ── 评分维度定义 ──

@dataclass
class Dimension:
    name: str
    weight: float
    patterns: Optional[List[str]] = None  # regex patterns
    fn: str = "regex"                     # regex / count / length_reward / length_penalty

    def score(self, text: str) -> float:
        if self.fn == "regex" and self.patterns:
            return 1.0 if any(re.search(p, text) for p in self.patterns) else 0.0
        if self.fn == "count" and self.patterns:
            total = sum(len(re.findall(p, text)) for p in self.patterns)
            return min(total / 3.0, 1.0)
        if self.fn == "length_reward":
            words = len(text)
            return min(words / 200.0, 1.0)
        if self.fn == "length_penalty":
            words = len(text)
            return 1.0 if words < 20 else 0.0
        return 0.0


# 15 个评分维度
DIMENSIONS: List[Dimension] = [
    # 代码与技术
    Dimension("code_presence", 3.0, [
        r"`[^`]+`", r"def\s+\w+\s*\(", r"class\s+\w+", r"import\s+\w+",
        r"function\s+\w+", r"<\w+>.*</\w+>", r"\bconst\b", r"\blet\b", r"\bvar\b",
        r"\bfor\s+\w+\s+in\b", r"\bwhile\b", r"\breturn\b",
        r"\bprint[f]?\(", r"\bif\s+__name__\b",
        # 中文代码相关
        r"写\s*(一个|个|一段).*(函数|方法|算法|代码|程序|脚本)",
        r"实现.*(排序|搜索|查找|算法|功能|接口)",
        r"用.*(写|实现|编写).*(代码|程序|脚本)",
        r"(写|实现).*(排序|算法|搜索|查找|函数|类)",
    ]),
    Dimension("technical_terms", 2.0, [
        r"\bAPI\b", r"\bSDK\b", r"\bJSON\b", r"\bHTTP\b", r"\bSQL\b",
        r"\bgit\b", r"\bdocker\b", r"\bdeploy\b", r"\bconfig\b",
        r"\balgorithm\b", r"\bdatabase\b", r"\bserver\b", r"\bport\b",
        # 中文技术术语（直接子串匹配，Python中中文为\w）
        "算法", "数据库", "接口", "协议", "框架", "函数", "排序", "搜索",
        "代码", "程序", "脚本", "参数", "配置", "命令",
        "Python", "Java", "js", "javascript", "typescript",
        "html", "css", "sql", "json", "yaml", "xml", "toml",
        "服务器", "架构", "前端", "后端", "部署",
    ]),
    # 推理与复杂度
    Dimension("reasoning_markers", 3.0, [
        "为什么", "怎么", "如何", "原因", "原理",
        "分析", "解释", "比较", "区别", "关系",
        "推理", "证明", "推导", "论证", "定理",
        "对比", "总结", "归纳", "概括",
        r"\bwhy\b", r"\bexplain\b", r"\bdifference\b",
    ]),
    Dimension("multi_step", 2.5, [
        r"首先.*然后", r"第一步.*第二步", r"先.*再.*最后",
        "流程", "步骤", "方案", "计划",
        r"first.*then", r"step\s+\d", r"\bprocess\b",
        r"分.*步", r"逐步", r"依次", r"分别",
        "重构", "优化", "改进", "升级",
    ]),
    # 指令与结构
    Dimension("imperative_verbs", 1.0, [
        r"请\s*(写|创建|生成|实现|修改|删除|添加|搜索|查|告诉|解释|翻译|设计|分析|计算|帮我)",
        r"帮我", r"给我", r"帮我把", r"把", r"将",
        r"(写|创建|生成|实现|翻译|解释|分析|计算|告诉|重构|比较|对比|画|设计|绘制)",
    ]),
    Dimension("constraint_count", 2.0, [
        r"要求", r"必须", r"需要", r"条件", r"限制", r"约束",
        r"不超过", r"不少于", r"在.*范围内",
        r"using\s", r"with\s", r"without\s",
    ], fn="count"),
    Dimension("output_format", 2.0, [
        r"以.*格式", r"返回.*JSON", r"输出.*列表",
        r"表格", r"列出", r"列举", r"用.*表示",
        r"output.*json", r"in.*format", r"as.*table",
    ]),
    # 意图与领域
    Dimension("simple_indicators", -2.0, patterns=[
        r"^(你好|您好|嗨|hi|hello|hey|早上好|下午好|晚上好|晚安)",
        r"^(哈哈|嘿嘿|好的|嗯嗯|是的|对|行吧|ok|好的吧|可以|好哒|好滴)",
        r"^(谢谢|感谢|多谢|辛苦|麻烦了|谢谢啦|thank你)",
        r"^(再见|拜拜|88|bye|明天见|下次聊)",
        r"^\.{3,}$", r"^\.+$",
        r"^(好|行|ok|嗯|哦|啊|诶|咦)$",
        r"^是$", r"^对$", r"^嗯嗯?$",
    ]),
    Dimension("creative_markers", 1.5, [
        "故事", "小说", "剧本", "诗歌", "歌词", "创意",
        r"\bstory\b", r"\bpoem\b", r"\bcreative\b", r"\bdesign\b",
    ]),
    Dimension("domain_specificity", 1.5, [
        "数学", "物理", "化学", "生物", "医学",
        "法律", "金融", "经济", "定理",
        r"\bmath\b", r"\bphysics\b", r"\bchemistry\b", r"\blaw\b",
    ]),
    # 其他
    Dimension("token_count", 1.5, fn="length_reward"),
    Dimension("question_complexity", 2.0, [
        r".*[？?]+\s*.*[？?]+",  # 多个问号
        r"(什么|哪里|谁|怎么|为什么|如何).*(什么|哪里|谁|怎么|为什么|如何)",  # 复合疑问
        r"并且|而且|同时|以及", r"还是|或者|要么",
        r"\.\s*[A-Z]",  # 多句
    ]),
    Dimension("reference_complexity", 2.0, [
        r"根据.*(之前|上面|前面|上文|刚才)", r"参考.*(文档|链接|文件|代码)",
        r"基于.*(情况|结果|分析)", r"在这个基础上",
        r"as\s+(mentioned|described|shown|stated)", r"based\s+on",
        r"following\s+the", r"according\s+to",
    ]),
    Dimension("negation_complexity", 1.5, [
        r"除了.*(不|没有|之外)", r"不要|不能|不可以|不允许",
        r"排除|忽略|跳过", r"但不包括|除了.*以外",
        r"except\s", r"excluding\s", r"without\s.*and\s",
        r"but\s+not", r"don't\s", r"shouldn't\s",
    ]),
    Dimension("agentic_task", 3.0, [
        r"(搜索|查|找)\s*(一下|一哈)?.*(表情|图片|用户|人|记忆|资料)",
        r"记(住|录|下)\s", r"记住", r"别忘了",
        r"用.*(工具|命令|脚本|函数)", r"执行(命令|脚本|任务)",
        r"run\s", r"execute\s", r"search\s", r"find\s",
        r"look\s+up", r"fetch\s", r"query\s",
        r"调用.*(工具|函数|接口|API)", r"给我.*(查|搜|看)",
    ]),
]

# 分档阈值
TIER_BOUNDS = [
    ("simple",    float("-inf"), 2.0),
    ("medium",    2.0,           4.0),
    ("complex",   4.0,           8.0),
    ("reasoning", 8.0,           float("inf")),
]

SIMPLE_SYSTEM_PROMPT = """你是一个友好的群聊助手。请简短自然地回复用户消息。
不要使用工具，不要搜索记忆，直接回复即可。保持语气亲切，符合你的角色设定。"""


class RuleRouter:
    """15 维规则评分路由引擎。纯本地计算，<1ms。

    用法:
        router = RuleRouter()
        tier = router.classify("你好呀")  # → "simple"
        tier = router.classify("写一个二分查找算法")  # → "complex"
    """

    def __init__(self):
        self._dimensions = DIMENSIONS
        self._bounds = TIER_BOUNDS

    def score(self, text: str) -> dict:
        """对文本进行 15 维评分，返回维度得分明细。"""
        results = {}
        total = 0.0
        for dim in self._dimensions:
            s = dim.score(text)
            weighted = s * dim.weight
            results[dim.name] = {"raw": s, "weight": dim.weight, "score": weighted}
            total += weighted
        results["total"] = total
        return results

    def classify(self, text: str) -> str:
        """评分 → 返回 tier 名称: simple / medium / complex / reasoning。"""
        total = 0.0
        for dim in self._dimensions:
            total += dim.score(text) * dim.weight

        for name, lo, hi in self._bounds:
            if lo <= total < hi:
                return name
        return "medium"

    def classify_with_detail(self, text: str) -> dict:
        """评分 + tier，返回完整信息。"""
        scores = self.score(text)
        tier = self.classify(text)
        return {"tier": tier, "total": scores.pop("total"), "dimensions": scores}


def is_simple_enough_for_direct(text: str) -> bool:
    """判断是否适合不走 ToolLoop 直接回复。

    条件：长度短 (< 50 字)、无代码、无技术术语、无复杂要求。
    """
    if len(text) > 50:
        return False
    if re.search(r"`[^`]+`|def\s+\w+|class\s+\w+", text):
        return False
    if re.search(r"为什么|怎么|如何|分析|解释|比较|区别|证明|推理|对比", text):
        return False
    return True
