# -*- coding: utf-8 -*-
"""多轮对话记忆：记住上一轮分析的城市/污染物/年份，支持追问。"""
from dataclasses import dataclass, field


@dataclass
class Memory:
    last_cities: list = field(default_factory=list)
    last_pollutant: str = None
    last_year: int = None
    last_intent: str = None
    history: list = field(default_factory=list)   # [(question, intent, seconds)]

    def update(self, plan, seconds=0.0, result_cities=None):
        # 优先记忆「本轮实际分析的城市」，支撑「这几个城市」式追问
        if result_cities:
            self.last_cities = list(result_cities)
        elif plan.cities:
            self.last_cities = list(plan.cities)
        self.last_pollutant = plan.pollutant
        if plan.year:
            self.last_year = plan.year
        self.last_intent = plan.intent
        self.history.append({
            "question": plan.raw, "intent": plan.intent,
            "pollutant": plan.pollutant, "year": plan.year,
            "seconds": round(seconds, 2),
        })

    def note_out_of_scope(self, plan):
        """拒答轮只记历史（意图统计用），不污染城市/污染物记忆。"""
        self.history.append({
            "question": plan.raw, "intent": "out_of_scope",
            "pollutant": None, "year": None,
            "seconds": 0.0,
        })

    def stats(self):
        """任务日志看板：高频任务、平均耗时。"""
        if not self.history:
            return None
        from collections import Counter
        cnt = Counter(h["intent"] for h in self.history)
        avg = sum(h["seconds"] for h in self.history) / len(self.history)
        return {"总任务数": len(self.history),
                "高频任务": cnt.most_common(3),
                "平均耗时秒": round(avg, 2)}
