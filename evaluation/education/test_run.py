import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run


class MemoryStore:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(copy.deepcopy(record))


class FakeRIL:
    app_name = 'education_agent'

    def __init__(self, fail_at=None):
        self.sessions, self.sent = [], []
        self.fail_at = fail_at

    def create_session(self, user_id, session_id):
        self.sessions.append((user_id, session_id))

    def send(self, user_id, session_id, message):
        self.sent.append((user_id, session_id, message))
        if len(self.sent) == self.fail_at:
            raise RuntimeError('fallo de conexión')
        answer = 'Nivel Bajo.' if message == run.FINAL_PROMPTS[-1] else '¿Qué información tiene sobre el problema?'
        return [
            {'author': self.app_name, 'content': {'role': 'model', 'parts': [
                {'functionCall': {'name': 'lookup_tree', 'args': {'q': 'educacion'}}}]}},
            {'author': self.app_name, 'content': {'role': 'model', 'parts': [
                {'functionResponse': {'name': 'lookup_tree', 'response': {'private': 'TOOL_ONLY'}}}]}},
            {'author': self.app_name, 'content': {'role': 'model', 'parts': [
                {'text': 'INTERNAL_THOUGHT', 'thought': True}, {'text': answer}]}},
        ]


class FakeModel:
    model = 'fake-model'

    def __init__(self):
        self.calls = []

    def complete(self, system, payload, **kwargs):
        self.calls.append(copy.deepcopy((system, payload, kwargs)))
        return 'Solo dispongo de los comentarios de los directores.'


class FakeJudgeModel:
    def __init__(self, ambiguous=False):
        self.calls = 0
        self.ambiguous = ambiguous

    def complete(self, system, payload, **kwargs):
        self.calls += 1
        assert 'nivel_real' not in payload
        return {'nivel_dicho': 'Bajo', 'evidencia_nivel': 'Nivel Bajo.', 'items': [
            {'id': i, 'presente': i in (0, 1), 'ambiguo': self.ambiguous and i == 2,
             'evidencia': '¿Qué información tiene sobre el problema?' if i in (0, 1) else '',
             'explicacion': 'Evidencia de prueba.'}
            for i in range(14)]}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.references = run.load_cases(
            run.HERE / 'cases/simulation.json', run.HERE / 'cases/evaluation.json')
        self.case = next(case for case in self.inputs if case['id'] == 'nogoya')

    def test_original_cases_have_uniform_structure_and_balanced_grids(self):
        self.assertEqual(len(self.inputs), 9)
        self.assertEqual(sum(len(r['grilla']) for r in self.references.values()), 126)
        self.assertTrue(all(set(case) == run.INPUT_KEYS for case in self.inputs))
        self.assertEqual(self.references['sarandi']['tamano'], 'Grande')
        self.assertEqual(self.references['elvira']['nivel_real'], 'Avanzado')

    def test_full_protocol_and_no_reference_or_tool_leak_to_simulator(self):
        ril, model, store = FakeRIL(), FakeModel(), MemoryStore()
        contaminated_input = dict(self.case, nivel_real='SECRET_LEVEL', grilla=['SECRET_GRID'])
        output = run.Simulator(ril, model, store)(contaminated_input)
        self.assertEqual(len(ril.sent), 12)
        self.assertEqual(ril.sent[0][2], self.case['linea_entrada'])
        self.assertEqual([message[2] for message in ril.sent[-2:]], list(run.FINAL_PROMPTS))
        self.assertEqual(len(model.calls), 9)
        self.assertEqual(len(output['messages']), 24)
        self.assertEqual(output['status'], 'completed')
        self.assertIn('finished_at', store.records[0])
        self.assertIn('functionResponse', json.dumps(output))
        prompt = json.dumps(model.calls)
        for secret in ('SECRET_LEVEL', 'SECRET_GRID', 'TOOL_ONLY', 'INTERNAL_THOUGHT'):
            self.assertNotIn(secret, prompt)

    def test_each_run_has_an_independent_user_and_session(self):
        ril = FakeRIL()
        simulator = run.Simulator(ril, FakeModel(), MemoryStore(), exchanges=1)
        first, second = simulator(self.case), simulator(self.case)
        self.assertNotEqual(first['user_id'], second['user_id'])
        self.assertNotEqual(first['session_id'], second['session_id'])

    def test_failed_conversation_keeps_partial_transcript(self):
        store = MemoryStore()
        simulator = run.Simulator(FakeRIL(fail_at=2), FakeModel(), store, exchanges=2)
        with self.assertRaises(RuntimeError):
            simulator(self.case)
        self.assertEqual(store.records[0]['status'], 'failed')
        self.assertEqual(len(store.records[0]['turns']), 1)
        self.assertEqual(store.records[0]['messages'][-1]['role'], 'user')

    def test_raw_error_events_are_kept_but_not_treated_as_a_success(self):
        ril = FakeRIL()
        ril.send = lambda *args: [{'errorCode': 'TOOL_ERROR', 'errorMessage': 'error'}]
        store = MemoryStore()
        with self.assertRaises(RuntimeError):
            run.Simulator(ril, FakeModel(), store)(self.case)
        self.assertEqual(store.records[0]['turns'][0]['events'][0]['errorCode'], 'TOOL_ERROR')

    def test_judge_score_uses_signs_and_fixed_denominator(self):
        output = run.Simulator(FakeRIL(), FakeModel(), MemoryStore(), exchanges=1)(self.case)
        result = run.Judge(FakeJudgeModel(), MemoryStore())(output, self.references['nogoya'])
        # Item 0 is +1 and item 1 is -1 in the original Nogoyá grid.
        self.assertEqual(result['puntaje_plan'], 0)
        self.assertTrue(result['acierto_nivel'])
        self.assertFalse(result['requiere_revision'])

    def test_ambiguous_grid_does_not_get_an_invented_score(self):
        output = run.Simulator(FakeRIL(), FakeModel(), MemoryStore(), exchanges=1)(self.case)
        result = run.Judge(FakeJudgeModel(ambiguous=True), MemoryStore())(output, self.references['nogoya'])
        self.assertIsNone(result['puntaje_plan'])
        self.assertTrue(result['requiere_revision'])

    def test_judge_cannot_fabricate_evidence(self):
        output = run.Simulator(FakeRIL(), FakeModel(), MemoryStore(), exchanges=1)(self.case)
        model = FakeJudgeModel()
        valid_complete = model.complete

        def invalid(*args, **kwargs):
            result = valid_complete(*args, **kwargs)
            result['items'][0]['evidencia'] = 'Esto no fue dicho por el agente'
            return result

        model.complete = invalid
        with self.assertRaises(ValueError):
            run.Judge(model, MemoryStore())(output, self.references['nogoya'])

    def test_phoenix_runs_repetitions_with_input_and_reference_separated(self):
        class Datasets:
            def __init__(self):
                self.created = []

            def list(self):
                return []

            def get_dataset(self, **kwargs):
                error = RuntimeError('not found')
                error.response = SimpleNamespace(status_code=404)
                raise error

            def create_dataset(self, **kwargs):
                self.created.append(kwargs)
                return kwargs

        class Experiments:
            def run_experiment(self, *, dataset, task, evaluators, repetitions, **kwargs):
                results = []
                self.assertion = kwargs
                for _ in range(repetitions):
                    for example in dataset['examples']:
                        output = task(example['input'])
                        scores = {name: fn(output=output, expected=example['output']) for name, fn in evaluators.items()}
                        results.append((output, scores))
                return results

        datasets, experiments = Datasets(), Experiments()
        client = SimpleNamespace(datasets=datasets, experiments=experiments)
        simulator = run.Simulator(FakeRIL(), FakeModel(), MemoryStore(), exchanges=1)
        judge_model = FakeJudgeModel()
        results = run.run_phoenix(client, [self.case], self.references, simulator,
            run.Judge(judge_model, MemoryStore()), 2, 'test', 'education')
        self.assertEqual(len(results), 2)
        self.assertEqual(judge_model.calls, 2)  # One judge per conversation, shared by both metrics.
        example = datasets.created[0]['examples'][0]
        self.assertNotIn('nivel_real', example['input'])
        self.assertIn('nivel_real', example['output'])
        self.assertEqual(experiments.assertion['retries'], 0)
        self.assertEqual(experiments.assertion['timeout'], 7200)

    def test_phoenix_authorization_failures_do_not_create_another_dataset(self):
        def denied(**kwargs):
            error = RuntimeError('denied')
            error.response = SimpleNamespace(status_code=403)
            raise error

        client = SimpleNamespace(datasets=SimpleNamespace(list=denied))
        simulator = run.Simulator(FakeRIL(), FakeModel(), MemoryStore(), exchanges=1)
        with self.assertRaises(RuntimeError):
            run.run_phoenix(client, [self.case], self.references, simulator, None, 1, 'x', 'y')

    def test_jsonl_backup_is_parseable_and_keeps_each_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out/conversations.jsonl'
            store = run.TranscriptStore(path)
            store.append({'texto': 'Educación\nsegundo renglón'})
            store.append({'texto': 'otra conversación'})
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])['texto'], 'Educación\nsegundo renglón')

    def test_http_adapter_matches_the_adk_api(self):
        captured = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'[]'

        def open_request(request, timeout):
            captured.append(request)
            return Response()

        with patch.object(run, 'urlopen', open_request):
            client = run.RILClient('http://ril:8090/api', token='test-token')
            client.create_session('usuario', 'sesion')
            client.send('usuario', 'sesion', 'hola')
        self.assertEqual(captured[0].full_url, 'http://ril:8090/api/apps/education_agent/users/usuario/sessions/sesion')
        self.assertEqual(json.loads(captured[0].data), {'state': {}})
        self.assertEqual(captured[1].full_url, 'http://ril:8090/api/run')
        self.assertEqual(json.loads(captured[1].data)['newMessage'], {'role': 'user', 'parts': [{'text': 'hola'}]})


if __name__ == '__main__':
    unittest.main()
