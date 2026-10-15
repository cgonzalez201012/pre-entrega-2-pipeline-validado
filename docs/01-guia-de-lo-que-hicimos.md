# Guía: qué construimos y por qué

Esta guía recorre la entrega en el orden en que conviene entenderla. Cada paso enlaza un concepto
del Módulo 2 con el archivo donde lo vas a ver.

## 1. El problema

Un LLM devuelve texto libre, y el texto libre es difícil de usar en un sistema: el formato cambia,
faltan campos, aparecen claves inventadas. La entrega pide un pipeline que convierta un párrafo
técnico en un **objeto validado** (`tecnologias`, `nivel_de_criticidad`, `resumen_tecnico`) y que
sea **resiliente**: si el modelo se equivoca de formato, que reintente.

## 2. El contrato (`schemas.py`) · Pydantic

`EntidadesTecnicas` cumple dos funciones a la vez:

1. LangChain lo convierte en un JSON Schema y se lo manda al modelo como una "herramienta" que
   debe rellenar. Así el modelo sabe qué forma tiene que tener la respuesta.
2. Cuando el modelo responde, Pydantic **valida** los datos. Esto cubre los tres tipos de
   validación de la unidad:
   - *Sintáctica*: los datos son parseables.
   - *Estructural*: están todos los campos (`resumen_tecnico` faltante → error).
   - *Semántica*: se cumplen reglas de negocio (`tecnologias` no vacía, criticidad ∈ {baja, media, alta}).

Detalles que conviene saber explicar:
- `Field(min_length=1)` es la restricción "la lista no puede venir vacía" que sugiere la consigna.
- `ConfigDict(extra="forbid")` hace que una clave inventada por el modelo (por ejemplo `confianza`) falle en
  lugar de ignorarse en silencio.
- **Forma ≠ verdad**: `min_length=1` obliga a que la lista no esté vacía, pero no garantiza que lo que haya
  adentro sea una tecnología. En la prueba de estrés real, ante un texto sin tecnologías el modelo puso
  `"No aplica"` para cumplir la regla. Por eso el contrato ofrece una salida explícita
  (`SIN_TECNOLOGIAS`) y `es_marcador_de_relleno` detecta los rellenos (sección 5, punto 4).
- El validador `normalizar_criticidad` acepta `"Alta"` o `" ALTA "` (no vale la pena gastar un reintento por una
  mayúscula) pero `"critica"` sigue siendo inválido: tolerante con la forma, estricto con el contenido.

## 3. El prompt (`prompts.py`) · `ChatPromptTemplate`

El prompt es un **artefacto de software**, no un string pegado en el código:
- Variables declaradas: `{texto}` (la entrada) e `{instrucciones_de_formato}` (precargada con
  `.partial(...)`, se puede reemplazar al invocar).
- **Sin f-strings**: el curso lo marca como error común ("Hardcoding de Prompts"). Hay un test con un
  log JSON (que trae `{ }`) para mostrar por qué: un f-string o `.format()` los interpretaría como variables.
- Separa roles: el `system` fija las reglas ("no inventes tecnologías") y el `human` trae el texto, marcado
  entre `<texto>` como *datos*, no como instrucciones.

## 4. La cadena LCEL (`chain.py`) · `build_chain`

```python
modelo_estructurado = model.with_structured_output(EntidadesTecnicas, include_raw=True, method="function_calling")
modelo_validado = (modelo_estructurado | RunnableLambda(_verificar_respuesta)).with_retry(...)
chain = PROMPT_EXTRACCION | modelo_validado
```

- `|` conecta *runnables*: la salida de uno es la entrada del siguiente (prompt → modelo → verificación).
- `with_structured_output(Schema)` es el "método preferido" de la consigna. Por debajo usa *tool calling*:
  fuerza al modelo a invocar una herramienta cuyo esquema es tu clase Pydantic y parsea los argumentos.
- **Por qué `include_raw=True`**: sin eso solo recibís el objeto parseado y se pierde el mensaje original.
  Con `True` recibís `{"raw": mensaje, "parsed": objeto|None, "parsing_error": error|None}`. Lo necesitamos
  para leer el `finish_reason` (siguiente punto) y para que un error de validación no rompa la cadena antes
  de que decidamos qué hacer con él.

## 5. El `finish_reason` · `_verificar_respuesta`

Error común de la consigna: *ignorar el `finish_reason`*. Si el modelo se queda sin tokens, la respuesta
queda cortada y el JSON incompleto. La función `_verificar_respuesta` lee el motivo de corte
(`stop_reason` en Anthropic, `finish_reason` en OpenAI) **antes** de mirar el resultado del parseo:

1. Si el motivo es `max_tokens` / `length` → `RespuestaTruncadaError`, **sin reintentar** (con el mismo
   `max_tokens` volvería a cortarse). El mensaje le dice al usuario qué hacer: subir `LLM_MAX_TOKENS`.
2. Si hubo error de validación → `RespuestaInvalidaError` (reintentable).
3. Si no hay datos estructurados (el modelo contestó texto libre) → `RespuestaInvalidaError` (reintentable).
4. Si la forma es válida pero la lista solo trae `SIN_TECNOLOGIAS` o un relleno ("No aplica", "<UNKNOWN>")
   → `TextoSinEntidadesError`, **sin reintentar**: el texto no cambia, otra llamada diría lo mismo.

## 6. La resiliencia · `.with_retry()`

```python
.with_retry(retry_if_exception_type=ERRORES_REINTENTABLES, wait_exponential_jitter=True, stop_after_attempt=3)
```

- **Backoff exponencial con jitter**: la espera crece entre intentos (≈1 s, ≈2 s...) con un componente
  aleatorio, para no saturar al servicio ni sincronizar muchos clientes que reintentan a la vez.
- **Solo errores recuperables** (`ERRORES_REINTENTABLES`): respuesta inválida, red caída, timeout, 429 y 5xx.
  Una API key inválida, un corte por `max_tokens` o un texto sin tecnologías **no** se reintentan: reintentar errores permanentes
  desperdicia plata y latencia (error de arquitectura que menciona el curso).
- **Qué envuelve**: solo el tramo "modelo + verificación". Cada reintento vuelve a llamar al modelo, sin
  rearmar el prompt.
- **`max_retries=0` en los clientes** (`build_model`): los SDK de OpenAI y Anthropic reintentan por su
  cuenta por defecto. Si no se apagan, habría dos capas de reintentos apiladas y no sabrías cuántas
  llamadas hiciste en realidad. Así hay una sola política, y es la que ves en el código.

## 7. Asincronía y logs · `process_text`

`process_text(text)` es `async` y ejecuta la cadena con `.ainvoke()`: mientras espera la respuesta de la
API, el event loop queda libre para otras tareas (la idea central del Módulo 1, ahora con LCEL).

Los logs salen de dos lugares:
- `PipelineLogHandler`: un *callback* de LangChain que cuenta las llamadas al modelo, avisa cuando una
  es un `REINTENTO` y **suma los tokens** de cada intento (`on_llm_end`). Un reintento es otra llamada y se
  paga, así que el total refleja el costo real de recuperarse de una respuesta inválida.
- `_verificar_respuesta`: registra `Validación OK` o `Validación FALLIDA` con el campo que falló y el
  `finish_reason`.

## 8. Cómo probamos sin gastar API (`modelo_simulado.py`)

`ModeloSimulado` es un chat model de LangChain de verdad con respuestas guionadas. Pasa por el mismo
`with_structured_output`, el mismo parser Pydantic y el mismo `with_retry` que un modelo real; lo único
falso es lo que "responde". Por eso podemos probar de forma **determinista** casos que con un LLM real
casi nunca ocurren a pedido: JSON incompleto, enum inválido, corte por `max_tokens`, 429.

Además de los tests, verificamos el pipeline con los SDK reales de Anthropic y OpenAI contra un
servidor HTTP local que respondía esos mismos escenarios (200, 429, 500, 401, `max_tokens`).

## 9. Qué te toca verificar vos

Todo lo anterior está probado sin llamar a un modelo real. Antes de entregar, en tu Mac:

```bash
pytest                       # 80 passed
python main.py               # modelo real (Anthropic): debería imprimir el JSON validado
python main.py --estres      # textos ambiguos: éxito o falla controlada, nunca un crash
python main.py --simular reintento     # el reintento, reproducible
python main.py --simular sin-entidades # texto sin tecnologías: falla controlada, sin reintentos
```

Los resultados reales de `python main.py` y de `--estres` ya están pegados en el README (secciones
"Ejemplo de salida" y "Prueba de estrés"). Si cambiás el prompt o el modelo, volvé a correrlos y actualizalos.
