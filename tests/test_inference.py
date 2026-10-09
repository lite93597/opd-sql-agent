import unittest

from opd_sql.prompts import format_messages, extract_sql


class PromptTests(unittest.TestCase):
    def test_reference_sql_is_not_exposed(self):
        record = {"schema": "CREATE TABLE x(a INT)", "question": "Count rows", "evidence": "",
                  "gold_sql": "SECRET_REFERENCE_SQL"}
        messages = format_messages(record, [{"sql": "SELECT bad FROM x", "error": "no such column: bad"}])
        text = repr(messages)
        self.assertNotIn("SECRET_REFERENCE_SQL", text)
        self.assertIn("no such column: bad", text)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_extract_does_not_silently_repair(self):
        self.assertEqual(extract_sql("<think>analysis</think>```sql\nSELECT 1;\n```"), "SELECT 1;")
        self.assertEqual(extract_sql("This is not SQL"), "This is not SQL")


if __name__ == "__main__":
    unittest.main()
