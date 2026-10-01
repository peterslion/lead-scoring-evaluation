"""Lock the homework answers computed from the pinned 2026 dataset."""

from __future__ import annotations

import unittest

from lead_scoring.__main__ import main
from lead_scoring.prepare import prepare


class HomeworkAnswersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare()
        cls.payload = main()

    def test_split_and_imputation(self):
        data = self.data
        self.assertEqual(len(data.full), 5000)
        self.assertEqual(len(data.full_train), 4000)
        self.assertEqual(len(data.train), 3000)
        self.assertEqual(len(data.validation), 1000)
        self.assertEqual(len(data.test), 1000)
        self.assertEqual(int(data.missing_counts.sum()), 148 + 240 + 194 + 208 + 369 + 35)
        self.assertEqual(int(data.full.isna().sum().sum()), 0)

    def test_question_answers(self):
        payload = self.payload
        self.assertEqual(payload["question_1"]["answer"], "lead_score")
        self.assertEqual(payload["question_2"]["answer"], 0.732)
        self.assertEqual(payload["question_3"]["answer"], 0.63)
        self.assertEqual(payload["question_4"]["answer"], 0.41)
        self.assertEqual(payload["question_5"]["answer"], 0.007)
        self.assertEqual(payload["question_6"]["answer"], 0.001)
        self.assertTrue(payload["question_2"]["matches_option"])
        self.assertTrue(payload["question_3"]["matches_option"])
        self.assertTrue(payload["question_4"]["matches_option"])
        self.assertTrue(payload["question_5"]["matches_option"])


if __name__ == "__main__":
    unittest.main()
