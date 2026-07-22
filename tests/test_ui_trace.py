"""测试智能体执行事件的界面展示状态。"""

import unittest

from services.ui_trace import TraceDisplay, apply_trace_event, format_runtime_error


class TraceDisplayTests(unittest.TestCase):
    def test_extracts_intent_from_supervisor_step(self):
        display = apply_trace_event(
            TraceDisplay(),
            {"supervisor_step": "问题分类结果: couplet"},
        )

        self.assertEqual(display.intent, "couplet")

    def test_formats_top_three_rag_references(self):
        display = apply_trace_event(
            TraceDisplay(),
            {"rag_references": ["第一条", "第二条", "第三条", "第四条"]},
        )

        self.assertEqual(display.rag_references, "1. 第一条\n2. 第二条\n3. 第三条")

    def test_formats_free_quota_runtime_error(self):
        message = format_runtime_error(Exception("AllocationQuota.FreeTierOnly Free quota exhausted"))

        self.assertIn("API Key、业务空间、模型名", message)


if __name__ == "__main__":
    unittest.main()
