"""chain.py - Pipeline de extracción de entidades técnicas (LCEL + Pydantic + reintentos).

Flujo completo de una llamada:

    {"texto": ...}
        │
        ▼
    ChatPromptTemplate ──► modelo.with_structured_output(EntidadesTecnicas)
                                   │  (devuelve la respuesta cruda + el objeto parseado)
                                   ▼
                          _verificar_respuesta  (finish_reason + validación Pydantic)
                                   │
                                   ▼   si lanza un error "reintentable" → .with_retry()
                          EntidadesTecnicas ya validado       vuelve a llamar al modelo

Decisiones de diseño (se explican en el README y en docs/):
  * with_structured_output(..., include_raw=True): devuelve también el mensaje crudo
    del modelo, necesario para leer el finish_reason / stop_reason y detectar cortes.
  * .with_retry() envuelve SOLO el tramo "modelo + verificación": cada reintento vuelve
    a llamar al modelo, pero no se re-arma el prompt.
  * Solo se reintentan errores recuperables (respuesta mal formada, red, rate limit,
    5xx). Un corte por max_tokens, una API key inválida o un texto sin tecnologías NO se
    reintentan: darían lo mismo cada vez y solo gastarían plata y latencia.
  * Validar la forma no es validar la verdad: con "lista no vacía" el modelo real rellena
    con "No aplica". Por eso hay una salida explícita (SIN_TECNOLOGIAS) y se detectan rellenos.
  * El cliente del proveedor se crea con max_retries=0 para que la ÚNICA política de
    reintentos sea la de .with_retry() (visible, configurable y con logs).
"""

import logging
import os
import time
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from prompts import PROMPT_EXTRACCION
from schemas import EntidadesTecnicas

logger = logging.getLogger("pipeline")

# --------------------------------------------------------------------------- #
# Configuración por defecto
# --------------------------------------------------------------------------- #
MODELOS_POR_DEFECTO = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}
MAX_TOKENS_POR_DEFECTO = 1024
TIMEOUT_S_POR_DEFECTO = 60.0
MAX_INTENTOS_POR_DEFECTO = 3

# Motivos de corte que indican que la respuesta quedó INCOMPLETA por falta de tokens.
#   OpenAI: "length" · Anthropic: "max_tokens" · Gemini: "MAX_TOKENS"
MOTIVOS_DE_TRUNCADO = {"length", "max_tokens", "MAX_TOKENS", "model_context_window_exceeded"}


# --------------------------------------------------------------------------- #
# Errores propios del pipeline
# --------------------------------------------------------------------------- #
class PipelineError(Exception):
    """Base de todos los errores controlados del pipeline."""


class RespuestaInvalidaError(PipelineError):
    """El modelo respondió, pero NO cumple el contrato (JSON roto, campo faltante...).

    Es un error REINTENTABLE: otra llamada puede devolver una respuesta correcta.
    """


class RespuestaTruncadaError(PipelineError):
    """El modelo se quedó sin tokens (finish_reason = length / max_tokens).

    NO se reintenta: con el mismo max_tokens volvería a cortarse. Solución: subir
    LLM_MAX_TOKENS o acortar el texto.
    """


class TextoSinEntidadesError(PipelineError):
    """La respuesta tiene la FORMA correcta, pero el texto no menciona ninguna tecnología.

    El modelo lo señaló con SIN_TECNOLOGIAS (o con un relleno tipo "No aplica"). NO se
    reintenta: el texto no cambia, así que otra llamada diría lo mismo y solo gastaría plata.
    Es un hallazgo de la prueba de estrés real: Pydantic valida la forma, no la verdad.
    """


class ProveedorError(PipelineError):
    """Falla del proveedor que no se arregla reintentando (API key inválida, etc.)
    o que siguió fallando tras agotar los reintentos (red caída, rate limit)."""


def _errores_transitorios() -> tuple[type[BaseException], ...]:
    """Errores de transporte que SÍ vale la pena reintentar (red, 429, 5xx, timeout).

    Se importan de forma opcional: cada SDK solo se usa si está instalado.
    """
    tipos: list[type[BaseException]] = []
    try:
        import anthropic

        tipos += [
            anthropic.APIConnectionError,  # incluye APITimeoutError
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ]
    except ImportError:  # pragma: no cover
        pass
    try:
        import openai

        tipos += [
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ]
    except ImportError:  # pragma: no cover
        pass
    return tuple(tipos)


ERRORES_REINTENTABLES: tuple[type[BaseException], ...] = (
    RespuestaInvalidaError,
    ValidationError,
    OutputParserException,
    *_errores_transitorios(),
)


# --------------------------------------------------------------------------- #
# Construcción del modelo
# --------------------------------------------------------------------------- #
def build_model(
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = MAX_TOKENS_POR_DEFECTO,
    timeout_s: float = TIMEOUT_S_POR_DEFECTO,
) -> BaseChatModel:
    """Crea el chat model de LangChain para el proveedor elegido (como en el Módulo 1).

    Las API keys se leen del entorno (ANTHROPIC_API_KEY / OPENAI_API_KEY).
    """
    provider = provider.lower().strip()
    nombre = model or MODELOS_POR_DEFECTO.get(provider)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Sin temperature: el SDK de Anthropic instalado (1.14) no lo incluye entre los
        # parámetros de messages.create() (ver README).
        return ChatAnthropic(model=nombre, max_tokens=max_tokens, timeout=timeout_s, max_retries=0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # temperature=0: para extracción de datos queremos la respuesta más determinista.
        return ChatOpenAI(
            model=nombre,
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout_s,
            max_retries=0,
        )

    raise ValueError(f"Proveedor no soportado: '{provider}'. Usá 'anthropic' u 'openai'.")


# --------------------------------------------------------------------------- #
# Verificación de la respuesta (finish_reason + validación)
# --------------------------------------------------------------------------- #
def _motivo_de_corte(mensaje: BaseMessage) -> str | None:
    """Lee por qué terminó de generar el modelo (el nombre de la clave depende del proveedor)."""
    meta = getattr(mensaje, "response_metadata", None) or {}
    return meta.get("finish_reason") or meta.get("stop_reason")


def _verificar_respuesta(resultado: dict[str, Any]) -> EntidadesTecnicas:
    """Recibe {'raw', 'parsed', 'parsing_error'} y devuelve el objeto validado o lanza un error.

    Orden de las comprobaciones (importa):
      1. finish_reason: si el modelo se quedó sin tokens, el JSON está cortado; hay que
         decirlo con claridad en vez de dejar un error de parseo confuso.
      2. parsing_error: Pydantic / el parser rechazaron la respuesta → reintentable.
      3. parsed vacío: el modelo no invocó la herramienta de salida → reintentable.
      4. contenido: la forma es válida pero la lista solo trae SIN_TECNOLOGIAS / rellenos
         ("No aplica", "<UNKNOWN>") → el texto no tiene tecnologías → NO reintentable.
    """
    raw: BaseMessage = resultado["raw"]
    parsed: EntidadesTecnicas | None = resultado.get("parsed")
    error: BaseException | None = resultado.get("parsing_error")
    motivo = _motivo_de_corte(raw)

    if motivo in MOTIVOS_DE_TRUNCADO:
        logger.error("Respuesta truncada por falta de tokens (finish_reason=%s). No se reintenta.", motivo)
        raise RespuestaTruncadaError(
            f"El modelo cortó la respuesta (finish_reason={motivo}). Subí LLM_MAX_TOKENS o acortá el texto."
        )

    if error is not None:
        logger.warning("Validación FALLIDA (finish_reason=%s): %s", motivo, _resumir_error(error))
        raise RespuestaInvalidaError(f"La respuesta no cumple el contrato: {_resumir_error(error)}") from error

    if parsed is None:
        logger.warning("Validación FALLIDA (finish_reason=%s): el modelo no devolvió datos estructurados.", motivo)
        raise RespuestaInvalidaError("El modelo no devolvió datos estructurados (no invocó la herramienta de salida).")

    if not parsed.tiene_tecnologias_reales():
        logger.warning(
            "El texto no menciona tecnologías identificables (el modelo devolvió %s). No se reintenta.",
            parsed.tecnologias,
        )
        raise TextoSinEntidadesError(
            "El texto no menciona ninguna tecnología identificable: no hay nada que extraer."
        )

    logger.info("Validación OK (finish_reason=%s).", motivo)
    return parsed


def _resumir_error(error: BaseException) -> str:
    """Versión corta y legible de un ValidationError para los logs."""
    if isinstance(error, ValidationError):
        partes = []
        for e in error.errors():
            campo = ".".join(str(p) for p in e["loc"]) or "(raíz)"
            partes.append(f"{campo}: {e['msg']}")
        return "; ".join(partes)
    return str(error).splitlines()[0] if str(error) else type(error).__name__


# --------------------------------------------------------------------------- #
# Logs de cada intento
# --------------------------------------------------------------------------- #
class PipelineLogHandler(AsyncCallbackHandler):
    """Callback que registra cada llamada al modelo, cuenta los intentos y suma los tokens.

    Un reintento es una llamada más al modelo, y cada llamada se paga: por eso se acumulan
    los tokens de TODOS los intentos (no solo del exitoso). Así se ve cuánto cuesta realmente
    recuperarse de una respuesta inválida.
    """

    def __init__(self) -> None:
        self.intentos = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    @property
    def tokens_totales(self) -> int:
        return self.tokens_entrada + self.tokens_salida

    async def on_chat_model_start(self, serialized, messages, *, tags=None, **kwargs) -> None:
        self.intentos += 1
        if self.intentos == 1:
            logger.info("Llamando al modelo (intento 1)...")
        else:
            logger.warning("REINTENTO: llamando al modelo (intento %d)...", self.intentos)

    async def on_llm_end(self, response, **kwargs) -> None:
        """Se ejecuta cuando el modelo responde: lee el consumo de tokens del mensaje."""
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None)
                if not uso:
                    continue  # algunos proveedores/modelos no informan el consumo
                entrada, salida = uso.get("input_tokens", 0), uso.get("output_tokens", 0)
                self.tokens_entrada += entrada
                self.tokens_salida += salida
                logger.info("Tokens del intento %d: entrada=%d, salida=%d.", self.intentos, entrada, salida)


# --------------------------------------------------------------------------- #
# La cadena LCEL
# --------------------------------------------------------------------------- #
def build_chain(
    model: BaseChatModel,
    max_intentos: int = MAX_INTENTOS_POR_DEFECTO,
    esperar_entre_intentos: bool = True,
) -> Runnable:
    """Arma:  prompt | (modelo.with_structured_output(Schema) | verificación).with_retry()

    Args:
        model: cualquier chat model de LangChain que soporte tool calling.
        max_intentos: intentos totales (el primero + los reintentos).
        esperar_entre_intentos: backoff exponencial con jitter entre intentos
            (poner False en tests para que no esperen).
    """
    # method="function_calling": el modelo "rellena" una herramienta cuyo esquema sale de
    # EntidadesTecnicas. Es el mismo mecanismo en todos los proveedores y no depende de los
    # límites del modo "strict" de cada API.
    modelo_estructurado = model.with_structured_output(
        EntidadesTecnicas, include_raw=True, method="function_calling"
    )

    modelo_validado = (modelo_estructurado | RunnableLambda(_verificar_respuesta)).with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        wait_exponential_jitter=esperar_entre_intentos,
        stop_after_attempt=max_intentos,
    )

    return PROMPT_EXTRACCION | modelo_validado


@lru_cache(maxsize=1)
def get_default_chain() -> Runnable:
    """Cadena por defecto, configurada con variables de entorno (se crea una sola vez)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    modelo = build_model(
        provider=provider,
        model=os.getenv("LLM_MODEL") or None,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", MAX_TOKENS_POR_DEFECTO)),
        timeout_s=float(os.getenv("LLM_TIMEOUT_S", TIMEOUT_S_POR_DEFECTO)),
    )
    return build_chain(modelo, max_intentos=int(os.getenv("LLM_MAX_INTENTOS", MAX_INTENTOS_POR_DEFECTO)))


# --------------------------------------------------------------------------- #
# Punto de entrada asíncrono
# --------------------------------------------------------------------------- #
async def process_text(text: str, chain: Runnable | None = None) -> EntidadesTecnicas:
    """Extrae entidades técnicas de `text` y devuelve un objeto EntidadesTecnicas validado.

    Args:
        text: párrafo sin procesar (arquitectura, log de error, incidente...).
        chain: cadena a usar; si no se pasa, se usa la configurada por entorno.

    Raises:
        ValueError: el texto está vacío (error de quien llama, no del pipeline).
        RespuestaInvalidaError: tras agotar los reintentos, el modelo no cumplió el contrato.
        RespuestaTruncadaError: el modelo se quedó sin tokens.
        TextoSinEntidadesError: el texto no menciona ninguna tecnología (no se reintenta).
        ProveedorError: falla del proveedor (API key, red, rate limit tras reintentos).
    """
    if not text or not text.strip():
        raise ValueError("El texto de entrada está vacío.")

    cadena = chain if chain is not None else get_default_chain()
    handler = PipelineLogHandler()
    inicio = time.perf_counter()
    logger.info("Procesando texto de %d caracteres.", len(text))

    try:
        resultado = await cadena.ainvoke(
            {"texto": text},
            config={"callbacks": [handler], "run_name": "extraccion_entidades_tecnicas"},
        )
    except PipelineError as exc:
        logger.error(
            "Pipeline falló tras %d intento(s) en %.2fs (tokens consumidos: %d): %s",
            handler.intentos, time.perf_counter() - inicio, handler.tokens_totales, exc,
        )
        raise
    except Exception as exc:  # API key inválida, red caída tras los reintentos, etc.
        logger.error("Error del proveedor tras %d intento(s): %s: %s", handler.intentos, type(exc).__name__, exc)
        raise ProveedorError(f"{type(exc).__name__}: {exc}") from exc

    logger.info(
        "Extracción OK en %.2fs (intentos usados: %d; tokens: entrada=%d, salida=%d, total=%d).",
        time.perf_counter() - inicio, handler.intentos,
        handler.tokens_entrada, handler.tokens_salida, handler.tokens_totales,
    )
    return resultado
