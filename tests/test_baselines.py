import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location('server_baseline', Path(__file__).parents[1]/'scripts/server/run_baselines.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def test_overflow_keeps_input_and_never_calls_model(self):
        with patch.object(baseline.requests, 'post') as post:
            result = baseline.generate({'id':'dev:1','db_id':'db'}, [1]*9, 'http://localhost', 'model', 2, 10)
        post.assert_not_called()
        self.assertEqual(result['input_tokens'], 9)
        self.assertEqual(result['finish_reason'], 'context_limit')
        self.assertTrue(result['error'])

    def test_token_ids_and_greedy_settings_are_preserved(self):
        response = Mock()
        response.json.return_value = {'choices':[{'text':'SELECT 1;', 'finish_reason':'stop'}],
                                     'usage':{'prompt_tokens':3,'completion_tokens':4}}
        with patch.object(baseline.requests, 'post', return_value=response) as post:
            result = baseline.generate({'id':'dev:1','db_id':'db','gold_sql':'SECRET'}, [2,3,4],
                                       'http://localhost', 'model', 512, 16384)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['prompt'], [2,3,4])
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['repetition_penalty'], 1)
        self.assertNotIn('SECRET', str(payload))
        self.assertEqual(result['sql'], 'SELECT 1;')
        self.assertIsNone(result['error'])

    def test_changed_prompt_length_is_reported_as_error(self):
        response = Mock()
        response.json.return_value = {'choices':[{'text':'SELECT 1;', 'finish_reason':'stop'}],
                                     'usage':{'prompt_tokens':99,'completion_tokens':4}}
        with patch.object(baseline.requests, 'post', return_value=response):
            result = baseline.generate({'id':'dev:1','db_id':'db'}, [2,3,4], 'http://localhost', 'model', 10, 100)
        self.assertIn('differs', result['error'])
        self.assertEqual(result['sql'], '')


if __name__ == '__main__':
    unittest.main()
