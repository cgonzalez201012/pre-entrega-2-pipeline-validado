# Pipeline de extracción de entidades técnicas (LCEL + Pydantic)

Pre-entrega 2 · Módulo 2 · AI Engineering (Coderhouse)

Recibe un párrafo de texto sin procesar (una descripción de arquitectura, un log de error, un
informe de incidente) y devuelve un **objeto validado** con las tecnologías mencionadas, el nivel
de criticidad y un resumen técnico. Si el LLM responde algo que no cumple el contrato (JSON mal
formado, campos faltantes, valores fuera del enum), el pipeline **reintenta automáticamente**; si no
logra recuperarse, falla de forma **controlada** (nunca con un crash inesperado).

## Ejemplo de salida

Salida real de `python main.py` (proveedor Anthropic, modelo `claude-haiku-4-5`; ejemplo de arquitectura):

```json
{
  "tecnologias": [
    "FastAPI",
    "PostgreSQL",
    "Redis"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Backend en FastAPI con PostgreSQL y Redis experimenta timeouts durante picos de tráfico porque el pool de conexiones a PostgreSQL se satura con concurrencia alta, impactando directamente la disponibilidad del servicio en producción."
}
```

Y los logs de esa misma ejecución (validación, intentos y consumo de tokens):

```
15:36:26 INFO    Procesando texto de 250 caracteres.
15:36:26 INFO    Llamando al modelo (intento 1)...
15:36:28 INFO    Tokens del intento 1: entrada=1311, salida=140.
15:36:28 INFO    Validación OK (finish_reason=tool_use).
15:36:28 INFO    Extracción OK en 2.08s (intentos usados: 1; tokens: entrada=1311, salida=140, total=1451).
```

Con un log de errores (`python main.py --ejemplo log`) el resultado real fue:

```json
{
  "tecnologias": ["psycopg2", "PostgreSQL", "Redis", "Kubernetes"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de órdenes experimenta una caída crítica por agotamiento de conexiones en PostgreSQL y cache miss del 98% en Redis, con 3 réplicas en Kubernetes fuera de servicio, resultando en healthcheck fallido (503)."
}
```

## Cómo está armado

```
{"texto": "..."}
      │
      ▼
ChatPromptTemplate  ──►  modelo.with_structured_output(EntidadesTecnicas)  ──►  verificación
   (prompts.py)              (el LLM "rellena" el esquema Pydantic)             (finish_reason + Pydantic)
                                         └──────────────  .with_retry()  ──────────────┘
                                                                   │
                                                                   ▼
                                                        EntidadesTecnicas validado
```

La cadena es una sola línea LCEL (en `chain.py`, función `build_chain`):

```python
modelo_estructurado = model.with_structured_output(EntidadesTecnicas, include_raw=True, method="function_calling")
modelo_validado = (modelo_estructurado | RunnableLambda(_verificar_respuesta)).with_retry(...)
chain = PROMPT_EXTRACCION | modelo_validado          # prompt | model.with_structured_output(...)
```

| Archivo | Qué contiene |
|---|---|
| `schemas.py` | Contrato Pydantic: `EntidadesTecnicas` (`tecnologias`, `nivel_de_criticidad`, `resumen_tecnico`) y el enum `NivelCriticidad` |
| `prompts.py` | `ChatPromptTemplate` modular (variables `{texto}` e `{instrucciones_de_formato}`; sin f-strings) |
| `chain.py` | Cadena LCEL, `.with_retry()`, verificación de `finish_reason`, errores propios, logs y la función asíncrona `process_text()` |
| `modelo_simulado.py` | "LLM de mentira" con respuestas guionadas, para probar los fallos sin API |
| `main.py` | Mini-script de prueba (ejemplos, prueba de estrés y simulaciones) |
| `tests/` | 80 tests (`pytest`) sin llamadas a ninguna API |
| `docs/` | Guía de lo construido, cumplimiento de la rúbrica y preguntas para defender la entrega |

### Decisiones de diseño

- **`include_raw=True`**: además del objeto parseado, devuelve el mensaje crudo del modelo. Es la
  única forma de leer el `finish_reason` / `stop_reason` y detectar respuestas **cortadas por falta
  de tokens** antes de intentar usarlas.
- **Qué se reintenta y qué no** (el curso lo marca como error común: *"reintentar errores permanentes"*):

  | Situación | ¿Reintenta? | Por qué |
  |---|---|---|
  | JSON incompleto, campo faltante, valor fuera del enum, lista vacía, sin datos estructurados | Sí | Otra llamada puede salir bien |
  | Red caída, timeout, rate limit (429), error 5xx | Sí | Fallos transitorios |
  | Corte por `max_tokens` / `length` | **No** | Con el mismo `max_tokens` volvería a cortarse: se informa con un error claro |
  | El texto no menciona ninguna tecnología (`SIN_TECNOLOGIAS` o un relleno como "No aplica") | **No** | El texto no cambia: otra llamada diría lo mismo y se pagaría de nuevo |
  | API key inválida (401) | **No** | Es permanente; reintentar solo gasta tiempo |

- **Una sola capa de reintentos**: los clientes se crean con `max_retries=0` (los SDK reintentan por
  su cuenta por defecto) para que la política sea únicamente la de `.with_retry()`: visible,
  configurable (`LLM_MAX_INTENTOS`) y con logs. Entre intentos hay backoff exponencial con jitter.
- **Errores controlados**: `process_text()` solo lanza `ValueError` (texto vacío) o subclases de
  `PipelineError`: `RespuestaInvalidaError`, `RespuestaTruncadaError`, `TextoSinEntidadesError`,
  `ProveedorError`.
- **Lista de tecnologías no vacía, y qué hacer cuando el texto no tiene ninguna**: `min_length=1` es una
  restricción de negocio del contrato, pero valida la *forma*, no la *verdad*. En la prueba de estrés con
  el modelo real, ante el texto sin tecnologías y ante `"ok"` el LLM llenó la lista con `"No aplica"` /
  `"<UNKNOWN>"` para cumplir la regla, y Pydantic lo aprobó (la lista no estaba vacía). Se resolvió en tres capas:
  1. **Salida explícita**: el prompt y la descripción del campo le ofrecen al modelo un valor honesto,
     `["SIN_TECNOLOGIAS"]`, en vez de empujarlo a inventar.
  2. **Detección de rellenos** (`es_marcador_de_relleno`): si igual aparece `"No aplica"`, `"N/A"`,
     `"<UNKNOWN>"`..., se descarta de las listas mixtas y, si es lo único que hay, se lo trata como
     "texto sin tecnologías". Es una heurística de lista corta, no una garantía.
  3. **Error propio y no reintentable**: `TextoSinEntidadesError`. El texto no cambia, así que reintentar
     solo gastaría tokens.

## Instalación

Requiere Python 3.12 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # y completá ANTHROPIC_API_KEY (o OPENAI_API_KEY)
```

## Uso

```bash
python main.py                         # ejemplo de arquitectura con el modelo real
python main.py --ejemplo log           # ejemplo de log de error
python main.py --texto "Tu texto..."   # texto propio
python main.py --estres                # prueba de estrés con textos ambiguos
python main.py --provider openai       # cambia de proveedor

# Sin API key ni internet (modelo simulado que falla a propósito):
python main.py --simular reintento         # 2 respuestas inválidas y a la 3ra se recupera
python main.py --simular sin-recuperacion  # siempre inválido: agota los reintentos
python main.py --simular truncado          # corte por max_tokens: NO reintenta
python main.py --simular sin-entidades     # texto sin tecnologías: falla controlada, NO reintenta
```

Desde código:

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("La API en FastAPI usa Redis y se cae bajo carga."))
print(resultado.model_dump_json(indent=2))
```

### El reintento en acción (`python main.py --simular reintento`)

```
19:45:46 INFO    Procesando texto de 250 caracteres.
19:45:46 INFO    Llamando al modelo (intento 1)...
19:45:46 INFO    Tokens del intento 1: entrada=120, salida=45.
19:45:46 WARNING Validación FALLIDA (finish_reason=tool_use): resumen_tecnico: Field required
19:45:46 WARNING REINTENTO: llamando al modelo (intento 2)...
19:45:46 INFO    Tokens del intento 2: entrada=120, salida=45.
19:45:46 WARNING Validación FALLIDA (finish_reason=tool_use): nivel_de_criticidad: Input should be 'baja', 'media' or 'alta'
19:45:46 WARNING REINTENTO: llamando al modelo (intento 3)...
19:45:46 INFO    Tokens del intento 3: entrada=120, salida=45.
19:45:46 INFO    Validación OK (finish_reason=tool_use).
19:45:46 INFO    Extracción OK en 0.03s (intentos usados: 3; tokens: entrada=360, salida=135, total=495).
```

Los logs también informan el **consumo de tokens**: cada reintento es una llamada más al modelo (y se
paga), así que el total suma todos los intentos, no solo el exitoso. Si el pipeline falla, el mensaje de
error indica igual los tokens consumidos.

### Prueba de estrés

`python main.py --estres` envía tres textos que tensionan el contrato: uno **ambiguo** ("se cayó lo de
siempre, creo que es la base o la red"), uno **sin tecnologías** (una convocatoria a reunión) y uno
**casi vacío** (`"ok"`). El objetivo es comprobar que el pipeline nunca se cae: o devuelve datos
válidos, o falla con un error controlado.

**Hallazgo con el modelo real.** La primera corrida contradijo una suposición del diseño. Se esperaba
que el texto sin tecnologías hiciera fallar la validación (lista vacía → reintentos → error). Pasó otra
cosa: para cumplir `min_length=1` el modelo **inventó relleno** (`"No aplica"`, `"<UNKNOWN>"`) y la
validación lo aprobó, porque una lista con un texto adentro tiene la forma correcta. Una restricción de
esquema no le dice al modelo qué hacer cuando no tiene nada que poner; lo empuja a fabricar algo que
cumpla la regla. Lección: **validar la forma no es validar la verdad**.

**Corrección** (ver "Decisiones de diseño"): salida explícita `SIN_TECNOLOGIAS` en el prompt y en el
esquema, detección de rellenos y un error propio sin reintentos. Se puede ver la parte determinista sin
API con `python main.py --simular sin-entidades`:

```
WARNING El texto no menciona tecnologías identificables (el modelo devolvió ['SIN_TECNOLOGIAS']). No se reintenta.
ERROR   Pipeline falló tras 1 intento(s) en 0.02s (tokens consumidos: 165): El texto no menciona ninguna tecnología identificable: no hay nada que extraer.

❌ Falla controlada (TextoSinEntidadesError): El texto no menciona ninguna tecnología identificable: no hay nada que extraer.
```

**Corrida real después de la corrección** (`claude-haiku-4-5`, `python main.py --estres`):

| Texto | Qué devolvió el modelo | Resultado | Intentos | Tokens | Tiempo |
|---|---|---|---|---|---|
| Ambiguo ("...es la base o la red, no sé...") | `["SIN_TECNOLOGIAS"]` | `TextoSinEntidadesError` | 1 | 1415 | 2.05 s |
| Sin tecnologías (reunión de presupuesto) | `["SIN_TECNOLOGIAS"]` | `TextoSinEntidadesError` | 1 | 1397 | 1.62 s |
| Casi vacío (`"ok"`) | `["SIN_TECNOLOGIAS"]` | `TextoSinEntidadesError` | 1 | 1357 | 1.33 s |

```
PRUEBA DE ESTRÉS: sin tecnologías
Texto: Mañana a las 10 hay reunión de equipo para revisar el presupuesto del trimestre.

15:31:21 INFO    Procesando texto de 80 caracteres.
15:31:21 INFO    Llamando al modelo (intento 1)...
15:31:23 INFO    Tokens del intento 1: entrada=1275, salida=122.
15:31:23 WARNING El texto no menciona tecnologías identificables (el modelo devolvió ['SIN_TECNOLOGIAS']). No se reintenta.
15:31:23 ERROR   Pipeline falló tras 1 intento(s) en 1.62s (tokens consumidos: 1397): El texto no menciona ninguna tecnología identificable: no hay nada que extraer.

❌ Falla controlada (TextoSinEntidadesError): El texto no menciona ninguna tecnología identificable: no hay nada que extraer.
```

El texto ambiguo nombra "la base" y "la red" sin decir cuál, y el modelo lo trató como "sin tecnologías":
es coherente con la regla del prompt de extraer solo lo mencionado *explícitamente*. Ese criterio es una
decisión de prompt: un prompt más permisivo habría devuelto `["base de datos", "red"]`.

Los logs muestran cada intento. Con `--simular` se reproducen de forma determinista los fallos que un
modelo real casi nunca produce a pedido.

## Tests

```bash
pytest
```

Cubren el contrato Pydantic, el prompt, la estructura LCEL, los reintentos (campo faltante, enum
inválido, sin herramienta, lista vacía), el corte por `max_tokens`/`length`, el texto sin tecnologías
(`SIN_TECNOLOGIAS` y rellenos: no reintenta), errores de red (reintenta) vs. API key inválida (no
reintenta), los logs y la construcción de los modelos reales.
Usan `ModeloSimulado`, así que no gastan API.

## Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` u `openai` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Clave del proveedor elegido |
| `LLM_MODEL` | `claude-haiku-4-5` / `gpt-4o-mini` | Modelo a usar |
| `LLM_MAX_TOKENS` | `1024` | Tokens máximos de la respuesta |
| `LLM_TIMEOUT_S` | `60` | Timeout por llamada, en segundos |
| `LLM_MAX_INTENTOS` | `3` | Intentos totales (el primero + los reintentos) |

> En Anthropic no se envía `temperature`: el SDK `anthropic` actual ya no lo acepta en
> `messages.create()` y varios modelos nuevos solo admiten el valor por defecto. En OpenAI se usa
> `temperature=0` para que la extracción sea lo más determinista posible.
