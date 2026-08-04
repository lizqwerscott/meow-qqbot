from datetime import datetime, timedelta, timezone


class TimeBlockBuilder:
    """构建当前时间动态块。"""

    def build(self) -> str:
        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        weekday_names = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]
        time_info = now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday_names[now.weekday()]})")
        return (
            f"当前时间: {time_info} (CST/UTC+8)\n\n"
            "注意：创建定时任务时请使用北京时间 (CST/UTC+8)，不要使用 UTC。"
        )
