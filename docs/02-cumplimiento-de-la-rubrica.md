# Cumplimiento de la rúbrica (Módulo 2)

Total 100 pts · Aprobación 70 pts. Para cada criterio: qué pide, dónde está en el repo y cómo
comprobarlo.

| Criterio (peso) | Qué se evalúa | Evidencia en el repo | Cómo comprobarlo |
|---|---|---|---|
| **Definición del esquema y salida estructurada (30%)** | Pydantic como contrato de datos e implementación de `with_structured_output` | `schemas.py`: `EntidadesTecnicas` con `tecnologias` (lista, mínimo 1, con salida explícita `SIN_TECNOLOGIAS`), `nivel_de_criticidad` (enum `baja/media/alta`) y `resumen_tecnico`; validadores y `extra="forbid"`. `chain.py` → `build_chain`: `model.with_structured_output(EntidadesTecnicas, ...)` | `pytest tests/test_schemas.py` |
| **Implementación de cadena LCEL y prompting (30%)** | Sintaxis pipe `\|` y diseño de `ChatPromptTemplate` para la extracción técnica | `chain.py` → `PROMPT_EXTRACCION \| modelo_validado`. `prompts.py`: `ChatPromptTemplate.from_messages` con roles system/human, variables `{texto}` e `{instrucciones_de_formato}`, sin f-strings | `pytest tests/test_prompts.py` y `test_la_cadena_es_prompt_pipe_modelo_con_retry` |
| **Ejecución asíncrona y resiliencia (25%)** | `.ainvoke()` y logs para observar validación y reintentos | `chain.py` → `process_text()` (`async`, usa `.ainvoke()`), `.with_retry()` con backoff exponencial, detección de `finish_reason`, `PipelineLogHandler` y logs en `_verificar_respuesta` | `python main.py --simular reintento` muestra los logs de cada intento; `pytest tests/test_chain.py` |
| **Estructura del repositorio y script de prueba (15%)** | Módulos (`schemas.py`, `chain.py`) y script de prueba | `schemas.py`, `chain.py`, `prompts.py`, `main.py` (mini-script de prueba asíncrono), `tests/`, `README.md` con ejemplo de salida JSON e instrucciones, `.env.example` | Revisar el árbol del repo |

## Checklist de entrega de la consigna

- [x] Repositorio de GitHub con `schemas.py` (modelo Pydantic) y `chain.py` (cadena LCEL).
- [x] Cadena compuesta con LCEL: `prompt | model.with_structured_output(Schema)`.
- [x] Lógica de reintento (`.with_retry()`) ante JSON mal formado o incompleto.
- [x] Función asíncrona `process_text()` con `.ainvoke()` y logs de validación.
- [x] Mini-script de prueba que ejecute un ejemplo (`main.py`).
- [x] README con el ejemplo de salida JSON y las instrucciones (la consigna indica que no hace falta PDF ni informe).
- [x] Prueba de estrés (paso sugerido 5 de la consigna): `python main.py --estres` con un texto ambiguo, uno sin
  tecnologías y uno casi vacío. La corrida real y lo que reveló están en el README ("Prueba de estrés").
- [x] Repositorio público en GitHub, sin `.env`. Verificado con un `git clone` limpio, entorno nuevo y `pytest` (80 passed).

## Errores comunes de la consigna, uno por uno

| Error común | Cómo lo evitamos |
|---|---|
| Ignorar el `finish_reason` | `include_raw=True` + `_verificar_respuesta` lee `stop_reason` / `finish_reason` y lanza `RespuestaTruncadaError` si el modelo se quedó sin tokens |
| Hardcoding de prompts (f-strings) | `ChatPromptTemplate` con variables; test con un log JSON que trae `{ }` |
| Confiar ciegamente en el JSON del modelo (unidad) | Pydantic valida todo; nunca se hace `json.loads` a mano |
| Reintentar errores permanentes (unidad) | `retry_if_exception_type` solo con errores recuperables; la API key inválida, el `max_tokens` y el texto sin tecnologías no se reintentan (tests lo comprueban) |
| Confundir "forma válida" con "dato verdadero" | Hallado en la prueba de estrés real: el modelo rellenó la lista con "No aplica" y Pydantic la aprobó. Se agregó la salida explícita `SIN_TECNOLOGIAS`, `es_marcador_de_relleno` y `TextoSinEntidadesError` |

## Extras que suman (más allá de lo pedido)

- Soporta Anthropic **y** OpenAI cambiando una variable de entorno (reutiliza la idea del Módulo 1).
- `ModeloSimulado` + `main.py --simular ...`: demostración reproducible de reintentos sin API.
- Logs con el **consumo de tokens** por intento y en total (un reintento es otra llamada que se paga).
- 80 tests, incluidos casos de falla (se verificó que los tests fallan si se rompe la lógica de reintentos o
  la detección de `finish_reason`).
