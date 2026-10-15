# Preguntas para defender la entrega

Practicalas en voz alta. Cada respuesta es corta a propósito: lo importante es que puedas explicarla
con tus palabras y señalar el archivo donde está.

**1. ¿Qué hace `with_structured_output()` por debajo?**
Convierte la clase Pydantic en un JSON Schema, se lo manda al proveedor como una herramienta (*tool
calling*) que el modelo debe invocar, y parsea los argumentos de esa llamada a una instancia de la clase.
Se parece a pedirle al modelo que "complete un formulario" en vez de escribir texto libre.

**2. ¿Por qué usaste `include_raw=True`?**
Porque necesito el mensaje crudo del modelo para leer el `finish_reason`. Sin eso solo recibo el objeto
parseado (o una excepción) y no puedo distinguir una respuesta mal formada de una cortada por falta de
tokens. Además evita que el error de parseo rompa la cadena antes de que yo decida qué hacer.

**3. ¿Qué pasa si el modelo se queda sin tokens?**
El `stop_reason` (Anthropic) o `finish_reason` (OpenAI) viene como `max_tokens`/`length`. El pipeline
lanza `RespuestaTruncadaError` y **no reintenta**, porque con el mismo `max_tokens` el resultado sería el
mismo. El mensaje indica subir `LLM_MAX_TOKENS` o acortar el texto.

**4. ¿Qué errores reintentás y cuáles no? ¿Por qué?**
Reintento los recuperables: respuesta inválida (campo faltante, enum inválido, lista vacía, sin datos
estructurados), caídas de red, timeouts, 429 y 5xx. No reintento los permanentes: API key inválida, corte por
`max_tokens` y texto sin tecnologías. Reintentar un error permanente solo gasta plata y latencia.

**5. ¿Por qué `max_retries=0` en el cliente si ya tenés `.with_retry()`?**
Porque los SDK reintentan por su cuenta por defecto (2 veces). Con las dos capas, un solo `ainvoke` podría hacer
hasta 3 × 3 llamadas sin que lo notes. Dejo una única política, visible y configurable.

**6. ¿Para qué sirve el backoff exponencial con jitter?**
Cada reintento espera más que el anterior (≈1 s, ≈2 s...), así no se satura un servicio que ya está con
problemas. El jitter agrega un componente aleatorio para que muchos clientes no reintenten sincronizados.

**7. ¿Por qué `.with_retry()` envuelve solo el modelo y la verificación, y no toda la cadena?**
Porque el prompt es determinista: no tiene sentido rearmarlo. Lo que puede fallar (y cambiar entre
intentos) es la llamada al modelo y la validez de su respuesta.

**8. ¿Por qué un `ChatPromptTemplate` y no un f-string?**
Porque el prompt es un artefacto de software: tiene variables declaradas, roles separados y se puede
versionar y reemplazar (por ejemplo `instrucciones_de_formato`). Con un f-string, un texto con llaves (como un
log JSON) rompe el formato o inyecta variables sin querer. Hay un test que lo demuestra.

**9. ¿Por qué `extra="forbid"` y `min_length=1`?**
`extra="forbid"` hace que una clave inventada por el modelo falle en lugar de ignorarse. `min_length=1` es la
restricción de negocio de la consigna: una lista vacía no es una extracción válida. Pero ojo, tiene una
contracara que descubrí en la prueba de estrés con el modelo real (ver pregunta 14): ante un texto sin
tecnologías, `min_length=1` empuja al modelo a inventar relleno. Por eso el contrato le da una salida
explícita (`SIN_TECNOLOGIAS`) y el pipeline la trata como `TextoSinEntidadesError`.

**10. ¿Qué hace `process_text` si el pipeline no logra recuperarse?**
No se cae: lanza un error propio (`RespuestaInvalidaError`, `RespuestaTruncadaError`,
`TextoSinEntidadesError` o `ProveedorError`), todos subclases de `PipelineError`. Quien llama puede capturar `PipelineError` y decidir (mostrar un mensaje,
reencolar, usar un valor por defecto).

**11. ¿Por qué es asíncrono?**
Porque la espera de la API es I/O. Con `ainvoke()` el event loop queda libre mientras el modelo "piensa": un
servidor (por ejemplo FastAPI) puede atender a otros usuarios. Se podría procesar muchos textos a la vez con
`chain.abatch(...)`.

**12. ¿Cómo probaste los reintentos si el LLM real casi nunca devuelve JSON roto?**
Con `ModeloSimulado`: un chat model de LangChain con respuestas guionadas que pasa por el mismo
`with_structured_output`, el mismo parser y el mismo `with_retry`. También lo verifiqué con los SDK reales
contra un servidor HTTP local que devolvía 200, 429, 500, 401 y `max_tokens`.

**13. ¿Cómo mejorarías el pipeline?**
- Que el reintento le muestre al modelo el error de validación (*self-correction*), por ejemplo con un parser
  de reparación, en lugar de repetir la misma llamada.
- Reemplazar la heurística de rellenos (`es_marcador_de_relleno`, una lista corta de textos) por una señal
  explícita en el esquema, por ejemplo un campo opcional `hay_tecnologias: bool`. No lo hice para respetar los
  tres campos que pide la consigna.
- Un *fallback* a un modelo alternativo (`.with_fallbacks()`) cuando el primero falla repetidamente.
- Métricas (latencia, cantidad de reintentos) con LangSmith, como se ve en el Módulo 7.

**14. ¿Qué te enseñó la prueba de estrés con el modelo real?**
Que validar la **forma** no es validar la **verdad**. Yo había escrito que un texto sin tecnologías "fallaría en
forma controlada" por la regla de lista no vacía. Con el texto de la reunión y con `"ok"`, el modelo no falló:
puso `"No aplica"` y `"<UNKNOWN>"` para cumplir la regla, y Pydantic lo aprobó porque la lista no estaba vacía.
Lo corregí con tres cambios: (1) una salida explícita `SIN_TECNOLOGIAS` en el prompt y en la descripción del
campo; (2) `es_marcador_de_relleno`, que descarta rellenos en listas mixtas y detecta cuando es lo único que hay;
(3) `TextoSinEntidadesError`, que no se reintenta porque el texto no cambia. Se ve con
`python main.py --simular sin-entidades` y está cubierto por tests. Después de la corrección volví a correr la prueba con el modelo real: los tres textos (ambiguo, sin tecnologías
y `"ok"`) devolvieron `["SIN_TECNOLOGIAS"]` y terminaron en `TextoSinEntidadesError` con **un solo intento** cada
uno. Fue un caso donde el modelo real mostró algo que los tests con el modelo simulado, por sí solos, no
habrían mostrado.

**15. ¿Por qué `TextoSinEntidadesError` no se reintenta?**
Porque no es un problema de formato sino de contenido: el texto no menciona tecnologías, y volver a preguntar
con el mismo texto dará la misma respuesta. Reintentar solo sumaría llamadas (y tokens pagados) sin cambiar el
resultado. Hay un test que comprueba que el modelo se llama una sola vez.
