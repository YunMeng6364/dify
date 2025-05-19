import re
from typing import Optional, cast


class JiebaKeywordTableHandler:
    def __init__(self):
        import jieba.analyse  # type: ignore

        from core.rag.datasource.keyword.jieba.stopwords import STOPWORDS

        # 初始化时设置Jieba的停用词表
        jieba.analyse.default_tfidf.stop_words = STOPWORDS  # type: ignore

    def extract_keywords(self, text: str, max_keywords_per_chunk: Optional[int] = 10) -> set[str]:
        """使用Jieba的TF-IDF算法从文本中提取关键词"""
        import jieba.analyse  # type: ignore

        # 提取指定数量的关键词
        keywords = jieba.analyse.extract_tags(
            sentence=text,
            topK=max_keywords_per_chunk,
        )
        # 将返回的关键词列表转换为字符串列表
        keywords = cast(list[str], keywords)

        # 返回扩展后的关键词集合
        return set(self._expand_tokens_with_subtokens(set(keywords)))

    def _expand_tokens_with_subtokens(self, tokens: set[str]) -> set[str]:
        """将关键词拆分为子词，并过滤掉停用词"""
        from core.rag.datasource.keyword.jieba.stopwords import STOPWORDS

        results = set()
        for token in tokens:
            results.add(token)
            # 使用正则表达式将关键词拆分为子词
            sub_tokens = re.findall(r"\w+", token)
            if len(sub_tokens) > 1:
                # 过滤掉停用词，并将子词加入结果集合
                results.update({w for w in sub_tokens if w not in list(STOPWORDS)})

        return results
