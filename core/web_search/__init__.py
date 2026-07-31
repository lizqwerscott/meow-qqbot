"""网页搜索 / 抓取服务包。"""

from core.web_search.config import WebFetchConfig, WebSearchConfig
from core.web_search.service import WebService

__all__ = ["WebFetchConfig", "WebSearchConfig", "WebService"]
