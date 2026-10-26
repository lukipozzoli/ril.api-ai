# Evaluación de Educación con Phoenix

Simula funcionarios municipales contra `education_agent` a través de la API ADK de RIL.
Phoenix autoalojado registra el dataset, las conversaciones como resultados de experimento
y las evaluaciones. No requiere suscripción de Arize. Las llamadas a los modelos de Google
utilizan las credenciales y condiciones del proyecto configurado.

## Datos

- `cases/simulation.json`: array de nueve fichas con identidad, población, situación y frase inicial.
- `cases/evaluation.json`: array de nueve referencias, enlazadas por `id`, con nivel, tamaño y grilla.

El simulador recibe exclusivamente los campos de `simulation.json`. El agente de RIL solo
recibe los mensajes del funcionario; no recibe la situación completa, las grillas ni etiquetas.
Cada ejecución utiliza un usuario y una sesión UUID nuevos para aislar la memoria entre corridas.

Se preservan los 126 ítems originales, con sus signos y fuentes. Aunque la introducción del
documento dice que la falla genérica «respuesta aplicable a cualquier municipio» está en las
nueve fichas, no aparece en las grillas de `lules` ni `alamos`; no se agregó ni se modificó ningún ítem.
La revisión del equipo de Educación sigue pendiente según el documento original.

## Preparación

Desde esta carpeta:

```sh
cp .env.example .env
```

Complete `GOOGLE_CLOUD_PROJECT` y configure el acceso a RIL en `.env`.
El sandbox ADK debe estar operativo y ofrecer `education_agent` en `/api/list-apps`.
Esta integración utiliza `/api/run`: ejecuta el mismo agente que `/api/run_sse` y conserva
todos los eventos de herramientas, respuestas, consumo y cambios de estado devueltos por ADK.
No utiliza la API de producción `:8080/run-sse`.

Con Vertex AI, necesita credenciales ADC válidas en `~/.config/gcloud`:

```sh
gcloud auth application-default login
docker compose up -d phoenix
```

Abra Phoenix en http://localhost:6006. Los datos se conservan en el volumen `phoenix-data`.
El servicio escucha solo en la interfaz local del host; en un servidor remoto, utilice un
túnel SSH o la configuración de autenticación de su despliegue. No hace falta publicar el puerto.

## Ejecución

Comience con una conversación y sin juez:

```sh
docker compose run --build --rm simulator --case nogoya --no-judge
```

Luego ejecute los nueve casos, tres veces cada uno, con evaluación:

```sh
docker compose run --rm simulator --repetitions 3 --experiment-name education-v1
```

Por defecto hay diez intercambios (cada uno incluye mensaje del usuario y respuesta del agente)
y dos turnos finales fijos: «dame un plan» y «¿en qué nivel de madurez ubicás a mi municipio?».
`--exchanges` permite de 1 a 10 para pruebas breves; registre ese valor al comparar experimentos.

Los casos se cargan en un dataset cuyo nombre incluye un hash de las fichas y referencias.
Repetir una corrida con los mismos casos reutiliza ese dataset, sin duplicar sus ejemplos.
Cada conversación es un resultado del experimento, no un nuevo ejemplo del dataset de casos.
También se exportan `results/<corrida>/conversations.jsonl` y, con juez, `evaluations.jsonl`.
Estos respaldos incluyen las ejecuciones fallidas y permanecen disponibles si falla el registro en Phoenix.

El juez devuelve evidencias por ítem. El puntaje se calcula con código: suma de los signos de
los ítems presentes dividida por 14. Si algún ítem es ambiguo, el puntaje queda sin valor y la
corrida requiere revisión; no se inventan criterios para resolverlo. El acierto de nivel se
mide por separado. El tamaño se usa para segmentar resultados, no se infiere de una respuesta
que el protocolo original nunca solicita al agente.

## Fuera de Docker

También puede ejecutar el programa desde un servidor, una VM o un entorno de CI con acceso
a RIL y Phoenix. No depende de que ambos servicios corran en la misma computadora:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
RIL_API_URL=http://localhost:8090/api .venv/bin/python run.py --case nogoya --no-judge
```

`PHOENIX_ENDPOINT`, `RIL_API_URL` y las credenciales se pueden suministrar como variables de
entorno. Las variables ya definidas tienen prioridad sobre `.env`.

## Verificación sin red ni credenciales

```sh
python3 run.py --validate
python3 -m unittest discover -s . -p 'test_*.py' -v
```
