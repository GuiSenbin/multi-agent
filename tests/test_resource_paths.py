"""测试本地资源路径解析。"""

import unittest
from pathlib import Path

from config.paths import resource_path


class ResourcePathTests(unittest.TestCase):
    def test_couplet_resource_resolves_inside_project(self):
        path = resource_path("couplettest.csv")

        self.assertEqual(path, Path(__file__).resolve().parents[1] / "resource" / "couplettest.csv")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
