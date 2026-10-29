"""Simulación de conversaciones de Educación y experimentos en Phoenix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
FINAL_PROMPTS = ('dame un plan', '¿en qué nivel de madurez ubicás a mi municipio?')
INPUT_KEYS = {'id', 'municipio', 'provincia', 'poblacion_habitantes', 'problema',
              'rol_usuario', 'linea_entrada', 'situacion'}
LEVELS = ('Bajo', 'Intermedio', 'Avanzado')
SIZES = ('Chico', 'Mediano', 'Grande')


def now():
    return datetime.now(timezone.utc).isoformat()


def load_env(path):
    """Read simple KEY=VALUE config without executing shell expressions."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        if not separator or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError(f'Configuración inválida en {path.name}')
        os.environ.setdefault(key, value.strip().strip('\"\''))


def load_cases(simulation_path, evaluation_path):
    inputs = json.loads(simulation_path.read_text())
    references = json.loads(evaluation_path.read_text())
    if not isinstance(inputs, list) or not isinstance(references, list) or not inputs:
        raise ValueError('Cada JSON debe ser un array de casos no vacío')
    ids = [case['id'] for case in inputs]
    ref_ids = [case['id'] for case in references]
    if len(set(ids)) != len(ids) or len(set(ref_ids)) != len(ref_ids) or set(ids) != set(ref_ids):
        raise ValueError('Los identificadores deben ser únicos y coincidir en ambos archivos')
    expected = {case['id']: case for case in references}
    for case in inputs:
        if set(case) != INPUT_KEYS:
            raise ValueError(f'{case["id"]}: campos inesperados o faltantes en simulación')
        if any(not isinstance(case[k], str) or not case[k].strip() for k in INPUT_KEYS - {'poblacion_habitantes'}):
            raise ValueError(f'{case["id"]}: datos de simulación incompletos')
        if type(case['poblacion_habitantes']) is not int or case['poblacion_habitantes'] <= 0:
            raise ValueError(f'{case["id"]}: población inválida')
        ref = expected[case['id']]
        if set(ref) != {'id', 'nivel_real', 'tamano', 'grilla'}:
            raise ValueError(f'{case["id"]}: referencia de evaluación inválida')
        if ref['nivel_real'] not in LEVELS or ref['tamano'] not in SIZES:
            raise ValueError(f'{case["id"]}: nivel o tamaño inválido')
        grid = ref['grilla']
        if not isinstance(grid, list) or not grid:
            raise ValueError(f'{case["id"]}: la grilla debe ser una lista no vacía')
        if any(not isinstance(item, dict) or set(item) != {'id', 'texto', 'signo', 'fuente'}
               or not isinstance(item['texto'], str) or not item['texto'].strip()
               or not isinstance(item['fuente'], str) or not item['fuente'].strip() for item in grid):
            raise ValueError(f'{case["id"]}: grilla incompleta')
        criterion_ids = [item['id'] for item in grid]
        if any(type(criterion_id) is not int for criterion_id in criterion_ids) or len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError(f'{case["id"]}: los IDs de los criterios deben ser enteros únicos')
        if any(type(item['signo']) is not int or item['signo'] not in (-1, 1) for item in grid):
            raise ValueError(f'{case["id"]}: cada criterio debe tener signo +1 o -1')
    return inputs, expected


class RILClient:
    def __init__(self, base_url, app_name='education_agent', timeout=180, token=None):
        self.base_url = base_url.rstrip('/')
        self.app_name = app_name
        self.timeout = timeout
        self.token = token

    def request(self, method, path, payload=None):
        headers = {'Accept': 'application/json'}
        body = None
        if payload is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(payload, ensure_ascii=False).encode()
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        req = Request(self.base_url + path, body, headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            # Do not include server bodies: they can contain credentials/context.
            raise RuntimeError(f'RIL {method} {path}: HTTP {exc.code}') from None

    def session_path(self, user_id, session_id):
        parts = [quote(value, safe='') for value in (self.app_name, user_id, session_id)]
        return f'/apps/{parts[0]}/users/{parts[1]}/sessions/{parts[2]}'

    def create_session(self, user_id, session_id):
        # No case facts or answer keys are injected into the RIL session state.
        return self.request('POST', self.session_path(user_id, session_id), {'state': {}})

    def send(self, user_id, session_id, message):
        # /run uses the same runner as /run_sse and returns complete raw events.
        return self.request('POST', '/run', {
            'appName': self.app_name, 'userId': user_id, 'sessionId': session_id,
            'newMessage': {'role': 'user', 'parts': [{'text': message}]},
            'streaming': False,
        })


def public_reply(events):
    if not isinstance(events, list):
        raise ValueError('RIL no devolvió una lista de eventos')
    texts = []
    for event in events:
        if event.get('errorCode') or event.get('errorMessage'):
            raise RuntimeError('RIL informó un error durante el turno')
        content = event.get('content') or {}
        if event.get('partial') or content.get('role') in ('user', 'tool'):
            continue
        for part in content.get('parts') or []:
            if part.get('text') and not part.get('thought'):
                texts.append(part['text'])
    if not texts:
        raise ValueError('El turno de RIL terminó sin respuesta pública')
    return '\n'.join(texts)


class GeminiModel:
    def __init__(self, model):
        from google import genai
        from google.genai import types
        self.types = types
        self.model = model
        http_options = types.HttpOptions(timeout=120_000)
        if os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'true').lower() == 'true':
            project = os.environ.get('GOOGLE_CLOUD_PROJECT')
            if not project:
                raise ValueError('Falta GOOGLE_CLOUD_PROJECT para usar Vertex AI')
            self.client = genai.Client(vertexai=True, project=project,
                                       location=os.getenv('GOOGLE_CLOUD_LOCATION', 'global'),
                                       http_options=http_options)
        else:
            key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not key:
                raise ValueError('Falta GEMINI_API_KEY o GOOGLE_API_KEY')
            self.client = genai.Client(api_key=key, http_options=http_options)

    def complete(self, system, payload, schema=None, temperature=0):
        config = {'system_instruction': system, 'temperature': temperature}
        if schema is not None:
            config.update(response_mime_type='application/json', response_json_schema=schema)
        import httpx
        # Only retry stateless model calls; RIL turns may already have changed the session.
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=json.dumps(payload, ensure_ascii=False),
                    config=self.types.GenerateContentConfig(**config),
                )
                break
            except (httpx.TransportError, ConnectionError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if not response.text or not response.text.strip():
            raise ValueError('El modelo no devolvió texto')
        return json.loads(response.text) if schema is not None else response.text.strip()


SIMULATOR_INSTRUCTION = (HERE / "instructions/simulation.tmpl").read_text(encoding="utf-8").strip()


class TranscriptStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record):
        with self.lock, self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


class Simulator:
    def __init__(self, ril, model, store, exchanges=10):
        if not 1 <= exchanges <= 10:
            raise ValueError('Los intercambios deben estar entre 1 y 10')
        self.ril, self.model, self.store, self.exchanges = ril, model, store, exchanges

    def __call__(self, input):
        # Strict allowlist even if a caller supplies an entire dataset example.
        case = {key: input[key] for key in INPUT_KEYS}
        run_id = str(uuid.uuid4())
        record = {
            'run_id': run_id, 'caso_id': case['id'], 'session_id': run_id,
            'user_id': f'eval-education-{run_id}', 'started_at': now(),
            'app_name': self.ril.app_name,
            'simulator_model': getattr(self.model, 'model', 'test'),
            'messages': [], 'turns': [], 'status': 'running',
        }
        try:
            self.ril.create_session(record['user_id'], record['session_id'])
            for index in range(self.exchanges):
                message = case['linea_entrada'] if index == 0 else self.model.complete(
                    SIMULATOR_INSTRUCTION, {'ficha': case, 'historial': record['messages']},
                    temperature=0.7)
                self._turn(record, message, 'dialogue')
            for message in FINAL_PROMPTS:
                self._turn(record, message, 'final')
            record['status'] = 'completed'
            return record
        except Exception as exc:
            record['status'] = 'failed'
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['finished_at'] = now()
            self.store.append(record)

    def _turn(self, record, message, phase):
        record['messages'].append({'role': 'user', 'content': message, 'phase': phase})
        events = self.ril.send(record['user_id'], record['session_id'], message)
        # Keep all tool calls, results, usage and state deltas, including errors.
        record['turns'].append({'index': len(record['turns']), 'phase': phase,
                                'user_message': message, 'events': events})
        answer = public_reply(events)
        record['messages'].append({'role': 'assistant', 'content': answer, 'phase': phase})


JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'nivel_dicho': {'type': 'string', 'enum': [*LEVELS, 'No determinado']},
        'evidencia_nivel': {'type': 'string'},
        'items': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer'}, 'presente': {'type': 'boolean'},
                'ambiguo': {'type': 'boolean'}, 'evidencia': {'type': 'string'},
                'explicacion': {'type': 'string'},
            },
            'required': ['id', 'presente', 'ambiguo', 'evidencia', 'explicacion'],
        }},
    },
    'required': ['nivel_dicho', 'evidencia_nivel', 'items'],
}
JUDGE_INSTRUCTION = (HERE / "instructions/judge.tmpl").read_text(encoding="utf-8").strip()


class Judge:
    def __init__(self, model, store):
        self.model, self.store = model, store

    def __call__(self, output, expected):
        if not isinstance(output, dict) or output.get('status') != 'completed':
            raise ValueError('No se evalúa una simulación fallida como si fuera completa')
        result = self.model.complete(JUDGE_INSTRUCTION,
            {'historial': output['messages'], 'grilla': expected['grilla']},
            schema=JUDGE_SCHEMA)
        ids = [item['id'] for item in result['items']]
        criterion_ids = {item['id'] for item in expected['grilla']}
        if (any(type(criterion_id) is not int for criterion_id in ids)
                or len(ids) != len(criterion_ids) or set(ids) != criterion_ids):
            raise ValueError('El juez debe devolver todos los ítems de la grilla, sin duplicados ni agregados')
        agent_messages = [m['content'] for m in output['messages'] if m['role'] == 'assistant']
        validation_errors = []
        for item in result['items']:
            if type(item['presente']) is not bool or type(item['ambiguo']) is not bool:
                raise ValueError('El juez devolvió un marcado inválido')
            if item['presente'] and (not item['evidencia'] or not any(item['evidencia'] in text for text in agent_messages)):
                # Keep the judge's citation for review, but never score unsupported evidence.
                item['ambiguo'] = True
                validation_errors.append(f'Ítem {item["id"]}: evidencia no encontrada en los mensajes del agente')
        level = result['nivel_dicho']
        if level not in (*LEVELS, 'No determinado'):
            raise ValueError('El juez devolvió un nivel inválido')
        level_valid = level == 'No determinado' or bool(agent_messages and result['evidencia_nivel']
            and result['evidencia_nivel'] in agent_messages[-1])
        if not level_valid:
            validation_errors.append('El nivel carece de evidencia en la respuesta final')
        ambiguous = any(item['ambiguo'] for item in result['items'])
        signs = {item['id']: item['signo'] for item in expected['grilla']}
        # Normalize using this case's grid, which may differ from other cases.
        score = sum(signs[item['id']] for item in result['items'] if item['presente']) / len(signs)
        result.update(run_id=output['run_id'], caso_id=output['caso_id'],
                      nivel_real=expected['nivel_real'], tamano=expected['tamano'],
                      acierto_nivel=level == expected['nivel_real'] if level_valid else None,
                      puntaje_plan=None if ambiguous else score,
                      requiere_revision=ambiguous or not level_valid,
                      errores_validacion=validation_errors)
        self.store.append(result)
        return result


def create_evaluators(judge):
    """Share one judge result or error between both metrics of a conversation."""
    cache = {}
    judge_lock = threading.Lock()
    unavailable_evaluation = {
        'label': 'no_evaluable',
        'explanation': 'La conversación no produjo un resultado completo. Revise el error de la tarea y su respaldo JSONL.',
    }

    def judged(output, expected):
        # Phoenix also invokes evaluators for tasks that failed with output=None.
        if not isinstance(output, dict) or output.get('status') != 'completed':
            return None
        with judge_lock:
            if output['run_id'] not in cache:
                try:
                    cache[output['run_id']] = judge(output, expected)
                except Exception as exc:
                    cache[output['run_id']] = exc
            result = cache[output['run_id']]
            if isinstance(result, Exception):
                raise result
            return result

    def acierto_nivel(output, expected):
        result = judged(output, expected)
        if result is None:
            return unavailable_evaluation.copy()
        evaluation = {'label': result['nivel_dicho'], 'explanation': result['evidencia_nivel']}
        if result['acierto_nivel'] is None:
            evaluation['label'] = 'requiere_revision'
        else:
            evaluation['score'] = int(result['acierto_nivel'])
        return evaluation

    def puntaje_plan(output, expected):
        result = judged(output, expected)
        if result is None:
            return unavailable_evaluation.copy()
        evaluation = {'label': 'requiere_revision' if result['requiere_revision'] else 'evaluado',
                      'explanation': json.dumps(result['items'], ensure_ascii=False)}
        if result['puntaje_plan'] is not None:
            evaluation['score'] = result['puntaje_plan']
        return evaluation

    return {'acierto_nivel': acierto_nivel, 'puntaje_plan': puntaje_plan}


def reevaluate_experiment(client, experiment_id, judge):
    """Use Phoenix's original outputs and references without calling RIL again."""
    experiment = client.experiments.get_experiment(experiment_id=experiment_id)
    result = client.experiments.evaluate_experiment(
        experiment=experiment, evaluators=create_evaluators(judge), retries=0, timeout=7200)
    return result


def run_phoenix(client, inputs, references, simulator, judge, repetitions, experiment_name, dataset_name):
    """Client injection also permits integration-contract tests without credentials."""
    payload = [{'input': case, 'output': references[case['id']]} for case in inputs]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    # Immutable-by-content name: repeated runs reuse the same cases and answer keys.
    name = f'{dataset_name}-{digest}'
    existing = next((dataset for dataset in client.datasets.list() if dataset['name'] == name), None)
    if existing is not None:
        dataset = client.datasets.get_dataset(dataset={'id': existing['id']})
    else:
        dataset = client.datasets.create_dataset(name=name, examples=payload,
            dataset_description=f'{len(inputs)} casos de Educación: input para simulación; output de referencia para evaluación.')

    def task(input):
        return simulator(input)

    evaluators = create_evaluators(judge) if judge is not None else None
    return client.experiments.run_experiment(dataset=dataset, task=task,
        evaluators=evaluators, repetitions=repetitions, retries=0,
        timeout=7200, experiment_name=experiment_name,
        experiment_metadata={'dataset_sha256': digest, 'app_name': simulator.ril.app_name,
                             'intercambios': simulator.exchanges})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=HERE / '.env')
    parser.add_argument('--simulation', type=Path, default=HERE / 'cases/simulation.json')
    parser.add_argument('--evaluation', type=Path, default=HERE / 'cases/evaluation.json')
    parser.add_argument('--validate', action='store_true', help='Valida los casos sin red ni llamadas a modelos')
    parser.add_argument('--case', action='append', default=[], help='ID de caso; repetible')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--exchanges', type=int, default=10)
    parser.add_argument('--no-judge', action='store_true', help='Solo generar conversaciones')
    parser.add_argument('--reevaluate-experiment', help='ID de Phoenix: evalúa conversaciones existentes sin ejecutar RIL')
    parser.add_argument('--experiment-name', default='education-simulation')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args(argv)
    load_env(args.env_file)
    if args.reevaluate_experiment and (args.no_judge or args.validate or args.case):
        parser.error('La reevaluación no admite --no-judge, --validate ni --case; utiliza el experimento original completo')
    if args.reevaluate_experiment:
        from phoenix.client import Client
        client = Client(base_url=os.getenv('PHOENIX_ENDPOINT', 'http://localhost:6006'),
                        api_key=os.getenv('PHOENIX_API_KEY') or None)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        output_dir = args.output_dir / ('reevaluation-' + stamp)
        store = TranscriptStore(output_dir / 'evaluations.jsonl')
        judge = Judge(GeminiModel(os.getenv('JUDGE_MODEL', 'gemini-2.5-flash')), store)
        experiment = reevaluate_experiment(client, args.reevaluate_experiment, judge)
        print(f'Reevaluación registrada; respaldo: {output_dir}')
        # Count unique runs: both metrics share one judge call and one backup record.
        completed_ids = {record['output']['run_id'] for record in experiment['task_runs']
                         if isinstance(record.get('output'), dict)
                         and record['output'].get('status') == 'completed'}
        evaluated_ids = {json.loads(line)['run_id'] for line in store.path.read_text().splitlines()} if store.path.exists() else set()
        return 0 if completed_ids == evaluated_ids else 1
    if args.repetitions < 1 or not 1 <= args.exchanges <= 10:
        parser.error('repetitions debe ser positivo; exchanges debe estar entre 1 y 10')
    inputs, references = load_cases(args.simulation, args.evaluation)
    if args.case:
        unknown = set(args.case) - {case['id'] for case in inputs}
        if unknown:
            parser.error(f'Casos desconocidos: {sorted(unknown)}')
        inputs = [case for case in inputs if case['id'] in args.case]
    if args.validate:
        criteria_count = sum(len(references[case['id']]['grilla']) for case in inputs)
        print(f'OK: {len(inputs)} casos; {criteria_count} criterios en total; sin etiquetas de evaluación en simulación.')
        return 0
    from phoenix.client import Client
    endpoint = os.getenv('PHOENIX_ENDPOINT', 'http://localhost:6006')
    client = Client(base_url=endpoint, api_key=os.getenv('PHOENIX_API_KEY') or None)
    model = GeminiModel(os.getenv('SIMULATOR_MODEL', 'gemini-2.5-flash'))
    ril = RILClient(os.getenv('RIL_API_URL', 'http://localhost:8090/api'),
                    os.getenv('RIL_APP_NAME', 'education_agent'),
                    int(os.getenv('RIL_TIMEOUT_SECONDS', '180')), os.getenv('RIL_API_TOKEN'))
    apps = ril.request('GET', '/list-apps')
    if ril.app_name not in apps:
        raise ValueError(f'La API ADK no ofrece {ril.app_name}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output_dir = args.output_dir / stamp
    simulator = Simulator(ril, model, TranscriptStore(output_dir / 'conversations.jsonl'), args.exchanges)
    judge = None if args.no_judge else Judge(
        GeminiModel(os.getenv('JUDGE_MODEL', 'gemini-2.5-flash')),
        TranscriptStore(output_dir / 'evaluations.jsonl'))
    experiment = run_phoenix(client, inputs, references, simulator, judge, args.repetitions,
                             args.experiment_name, 'ril-education')
    print(f'Experimento registrado en {endpoint}; respaldo: {output_dir}')
    if isinstance(experiment, dict):
        experiment_id = experiment.get('id') or experiment.get('experiment_id')
    else:
        experiment_id = getattr(experiment, 'id', None)
    if experiment_id:
        print(f'ID: {experiment_id}')
    transcripts = output_dir / 'conversations.jsonl'
    records = [json.loads(line) for line in transcripts.read_text().splitlines()] if transcripts.exists() else []
    failed = len(records) != len(inputs) * args.repetitions or any(record['status'] != 'completed' for record in records)
    if judge is not None:
        evaluations = output_dir / 'evaluations.jsonl'
        count = len(evaluations.read_text().splitlines()) if evaluations.exists() else 0
        failed = failed or count != len(inputs) * args.repetitions
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
