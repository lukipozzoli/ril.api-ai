# Evaluación de Educación con Phoenix

Simula funcionarios municipales contra `education_agent` a través de la API ADK de RIL.
Phoenix autoalojado registra el dataset, las conversaciones como resultados de experimento
y las evaluaciones. No requiere suscripción de Arize. Las llamadas a los modelos de Google
utilizan las credenciales y condiciones del proyecto configurado.

## Datos

Los prompts se editan por separado en `instructions/simulation.tmpl` y
`instructions/judge.tmpl`. El programa los carga en UTF-8 al iniciar, con rutas relativas
a `run.py`, independientemente de la carpeta desde la que se ejecute.

- `cases/simulation.json`: array de nueve fichas con identidad, población, situación y frase inicial.
- `cases/evaluation.json`: array de nueve referencias, enlazadas por `id`, con nivel, tamaño y grilla.

El simulador recibe exclusivamente los campos de `simulation.json`. El agente de RIL solo
recibe los mensajes del funcionario; no recibe la situación completa, las grillas ni etiquetas.
Cada ejecución utiliza un usuario y una sesión UUID nuevos para aislar la memoria entre corridas.

Puede usar otras parejas de archivos con `--simulation RUTA --evaluation RUTA`.
Los IDs de los casos deben ser únicos y coincidir entre ambos archivos.
Cada caso puede tener una cantidad distinta de criterios y cualquier proporción de signos
positivos y negativos, incluso todos del mismo signo. La grilla debe tener al menos un
criterio. Cada criterio conserva los campos `id` (entero único dentro de su grilla, sin
necesidad de ser consecutivo), `texto`, `signo` (`+1` o `-1`) y `fuente`.

Se preservan los 126 ítems originales, con sus signos y fuentes. Aunque la introducción del
documento dice que la falla genérica «respuesta aplicable a cualquier municipio» está en las
nueve fichas, no aparece en las grillas de `lules` ni `alamos`; no se agregó ni se modificó ningún ítem.
La revisión del equipo de Educación sigue pendiente según el documento original.

## Preparación

Desde esta carpeta:

```sh
cp .env.example .env
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

El simulador y el juez se ejecutan con Python local (3.10 o superior). El Compose de
esta carpeta agrega solamente Phoenix; reutilice el sandbox del Docker de RIL existente,
con su puerto 8090 publicado. No necesita reconstruir ni duplicar RIL.

Complete `GOOGLE_CLOUD_PROJECT` y configure el acceso a RIL en `.env`.
El sandbox ADK debe estar operativo y ofrecer `education_agent` en `/api/list-apps`.
Esta integración utiliza `/api/run`: ejecuta el mismo agente que `/api/run_sse` y conserva
todos los eventos de herramientas, respuestas, consumo y cambios de estado devueltos por ADK.
No utiliza la API de producción `:8080/run-sse`.
El simulador accede desde el host a RIL por `http://localhost:8090/api` y a Phoenix por
`http://localhost:6006`. Los contenedores no necesitan compartir una red entre sí.

El endpoint `/api/run` espera a que termine el turno antes de responder. El sandbox debe
arrancar con `web -port 8090 -write-timeout 3m api ...`: el límite predeterminado de 15
segundos puede cerrar la conexión durante una respuesta del modelo, incluso si el cliente
tiene un timeout mayor. El Compose de desarrollo de este fork ya incluye ese ajuste.
Para reutilizar el sandbox de otro checkout sin reconstruir su imagen, desde esta carpeta:

```sh
docker compose -p rilapi-ai-main \
  -f /ruta/al/repo-original/docker-compose.dev.yml \
  -f "$PWD/compose.ril-timeout.yaml" up -d --no-build --no-deps sandbox
```

Adapte el nombre del proyecto (`-p`) y la ruta al checkout que usa su contenedor existente.
Recrear el sandbox borra sus sesiones en memoria; detenga las corridas antes de hacerlo.

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
.venv/bin/python run.py --case nogoya --no-judge
```

Luego ejecute los nueve casos, tres veces cada uno, con evaluación:

```sh
.venv/bin/python run.py --repetitions 3 --experiment-name education-v1
```

Por defecto hay diez intercambios (cada uno incluye mensaje del usuario y respuesta del agente)
y dos turnos finales fijos: «dame un plan» y «¿en qué nivel de madurez ubicás a mi municipio?».
`--exchanges` permite de 1 a 10 para pruebas breves; registre ese valor al comparar experimentos.

Los casos se cargan en un dataset cuyo nombre incluye un hash de las fichas y referencias.
Repetir una corrida con los mismos casos reutiliza ese dataset, sin duplicar sus ejemplos.
Cada conversación es un resultado del experimento, no un nuevo ejemplo del dataset de casos.
También se exportan `results/<corrida>/conversations.jsonl` y, con juez, `evaluations.jsonl`.
Estos respaldos incluyen las ejecuciones fallidas y permanecen disponibles si falla el registro en Phoenix.
Si una tarea falla o no devuelve una conversación completa, sus métricas se etiquetan
`no_evaluable`, sin puntaje ni llamada al juez. El error original permanece en la tarea;
la corrida sigue considerándose fallida. Los experimentos anteriores conservan sus errores.

El juez devuelve evidencias por ítem. El puntaje se calcula con código: suma de los signos de
los ítems presentes dividida por la cantidad total de criterios de la grilla de ese caso
(14 en las fichas originales). Si algún ítem es ambiguo, el puntaje queda sin valor y la
corrida requiere revisión; no se inventan criterios para resolverlo. El acierto de nivel se
mide por separado. El tamaño se usa para segmentar resultados, no se infiere de una respuesta
que el protocolo original nunca solicita al agente.

Si una cita del juez no aparece literalmente en los mensajes del agente, se conserva
para revisión y ese criterio se marca ambiguo: no se publica un puntaje de plan.
Si la cita del nivel no está en la respuesta final, su métrica tampoco recibe puntaje.
Las dos métricas comparten una llamada al juez y también comparten sus errores, para
evitar repetir llamadas fallidas. Las llamadas locales a Gemini tienen un timeout de
120 segundos por intento y hasta tres intentos ante errores de transporte. No se
reintentan automáticamente los turnos de RIL porque podrían duplicar acciones en la sesión.

## Reevaluar sin repetir conversaciones

Detenga o espere a que termine la corrida anterior antes de reevaluar el mismo experimento.
Para ejecutar solamente el juez sobre las conversaciones guardadas en Phoenix:

```sh
.venv/bin/python run.py --reevaluate-experiment RXhwZXJpbWVudDo1
```

Reemplace el ID por el de su experimento. Se reutilizan sus respuestas y referencias
originales, sin llamar a RIL ni al simulador. Se actualizan las métricas del experimento
y se guardan nuevos respaldos en `results/reevaluation-*/evaluations.jsonl`.
Las conversaciones fallidas siguen sin puntaje; este comando no las vuelve a simular.

## Servicios remotos y otras parejas de casos

También puede ejecutar el programa desde un servidor, una VM o un entorno de CI con acceso
a RIL y Phoenix. No depende de que ambos servicios corran en la misma computadora:

```sh
.venv/bin/python run.py \
  --simulation cases/simulation.json \
  --evaluation cases/evaluation.json \
  --experiment-name education-v2
```

`PHOENIX_ENDPOINT`, `RIL_API_URL` y las credenciales se pueden suministrar como variables de
entorno. Las variables ya definidas tienen prioridad sobre `.env`.

## Verificación sin red ni credenciales

```sh
python3 run.py --validate
python3 -m unittest discover -s . -p 'test_*.py' -v
```
