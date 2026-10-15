"""Tests de la cadena (chain.py) usando el ModeloSimulado: sin API, sin red, deterministas.

Cada test cuenta cuántas veces se llamó al modelo (`modelo.llamadas`) para comprobar
cuándo el pipeline reintenta y cuándo NO.
"""

import logging

import anthropic
import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.retry import RunnableRetry

import chain as chain_module
from chain import (
    PipelineLogHandler,
    ProveedorError,
    RespuestaInvalidaError,
    RespuestaTruncadaError,
    TextoSinEntidadesError,
    ERRORES_REINTENTABLES,
    build_chain,
    build_model,
    process_text,
)
from modelo_simulado import (
    ModeloSimulado,
    respuesta_con_criticidad_invalida,
    respuesta_incompleta,
    respuesta_ok,
    respuesta_sin_herramienta,
    respuesta_sin_tecnologias,
    respuesta_truncada,
)
from schemas import EntidadesTecnicas, NivelCriticidad

TEXTO = "La API en FastAPI usa Redis y PostgreSQL y se satura con muchas conexiones."


def armar(respuestas, **kwargs):
    """Devuelve (modelo, cadena) sin esperas entre intentos."""
    modelo = ModeloSimulado(respuestas=respuestas)
    return modelo, build_chain(modelo, esperar_entre_intentos=False, **kwargs)


# --------------------------------------------------------------------------- #
# Estructura de la cadena LCEL
# --------------------------------------------------------------------------- #
def test_la_cadena_es_prompt_pipe_modelo_con_retry():
    _, cadena = armar([respuesta_ok()])
    assert isinstance(cadena.first, ChatPromptTemplate)  # prompt | ...
    assert isinstance(cadena.last, RunnableRetry)  # ... | (modelo estructurado).with_retry()


# --------------------------------------------------------------------------- #
# Camino feliz
# --------------------------------------------------------------------------- #
async def test_respuesta_valida_a_la_primera():
    modelo, cadena = armar([respuesta_ok()])
    resultado = await process_text(TEXTO, chain=cadena)
    assert isinstance(resultado, EntidadesTecnicas)
    assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert resultado.nivel_de_criticidad is NivelCriticidad.ALTA
    assert modelo.llamadas == 1


# --------------------------------------------------------------------------- #
# Reintentos ante respuestas mal formadas o incompletas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mala",
    [
        respuesta_incompleta(),  # JSON incompleto: falta un campo
        respuesta_con_criticidad_invalida(),  # valor fuera del enum
        respuesta_sin_herramienta(),  # texto libre, sin datos estructurados
        respuesta_ok(tecnologias=[]),  # viola la restricción "lista no vacía"
    ],
    ids=["campo-faltante", "enum-invalido", "sin-herramienta", "lista-vacia"],
)
async def test_se_recupera_tras_una_respuesta_invalida(mala):
    modelo, cadena = armar([mala, respuesta_ok()])
    resultado = await process_text(TEXTO, chain=cadena)
    assert resultado.nivel_de_criticidad is NivelCriticidad.ALTA
    assert modelo.llamadas == 2  # 1 intento fallido + 1 reintento exitoso


async def test_se_recupera_tras_dos_respuestas_invalidas():
    modelo, cadena = armar([respuesta_incompleta(), respuesta_con_criticidad_invalida(), respuesta_ok()])
    await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 3


async def test_agota_los_reintentos_y_falla_en_forma_controlada():
    modelo, cadena = armar([respuesta_incompleta()], max_intentos=3)  # siempre inválida
    with pytest.raises(RespuestaInvalidaError, match="resumen_tecnico"):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 3


async def test_max_intentos_es_configurable():
    modelo, cadena = armar([respuesta_incompleta()], max_intentos=5)
    with pytest.raises(RespuestaInvalidaError):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 5


# --------------------------------------------------------------------------- #
# finish_reason: respuestas cortadas por falta de tokens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("motivo", ["max_tokens", "length", "MAX_TOKENS"])
async def test_respuesta_truncada_se_detecta_y_no_se_reintenta(motivo):
    modelo, cadena = armar([respuesta_truncada(motivo), respuesta_ok()])
    with pytest.raises(RespuestaTruncadaError, match=motivo):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 1  # reintentar con el mismo max_tokens daría lo mismo


# --------------------------------------------------------------------------- #
# Forma válida, contenido vacío: el texto no menciona tecnologías (NO se reintenta)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("marcador", ["SIN_TECNOLOGIAS", "No aplica", "<UNKNOWN>", "N/A"])
async def test_texto_sin_tecnologias_falla_controlado_y_no_se_reintenta(marcador):
    # Pasa la validación de Pydantic (la lista no está vacía), pero no hay nada real que extraer.
    modelo, cadena = armar([respuesta_sin_tecnologias(marcador), respuesta_ok()])
    with pytest.raises(TextoSinEntidadesError, match="ninguna tecnología"):
        await process_text("Mañana hay reunión de presupuesto.", chain=cadena)
    assert modelo.llamadas == 1  # el texto no cambia: reintentar daría lo mismo y se pagaría


def test_texto_sin_entidades_no_es_un_error_reintentable():
    assert not issubclass(TextoSinEntidadesError, ERRORES_REINTENTABLES)


async def test_lista_mixta_descarta_el_relleno_y_se_acepta():
    modelo, cadena = armar([respuesta_ok(tecnologias=["Redis", "No aplica"])])
    resultado = await process_text(TEXTO, chain=cadena)
    assert resultado.tecnologias == ["Redis"]
    assert modelo.llamadas == 1


async def test_texto_sin_tecnologias_se_registra_en_los_logs(caplog):
    _, cadena = armar([respuesta_sin_tecnologias("No aplica")])
    with caplog.at_level(logging.INFO, logger="pipeline"):
        with pytest.raises(TextoSinEntidadesError):
            await process_text("Reunión de presupuesto.", chain=cadena)
    assert "no menciona tecnologías identificables" in caplog.text
    assert "No aplica" in caplog.text
    assert "Validación OK" not in caplog.text  # no se da por buena una respuesta vacía


# --------------------------------------------------------------------------- #
# Errores del proveedor: transitorios sí se reintentan, permanentes no
# --------------------------------------------------------------------------- #
def _error_de_conexion():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))


def _error_de_autenticacion():
    resp = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com"))
    return anthropic.AuthenticationError("invalid x-api-key", response=resp, body=None)


async def test_error_de_red_transitorio_se_reintenta():
    modelo, cadena = armar([_error_de_conexion(), respuesta_ok()])
    resultado = await process_text(TEXTO, chain=cadena)
    assert isinstance(resultado, EntidadesTecnicas)
    assert modelo.llamadas == 2


async def test_api_key_invalida_no_se_reintenta():
    modelo, cadena = armar([_error_de_autenticacion(), respuesta_ok()])
    with pytest.raises(ProveedorError, match="AuthenticationError"):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 1


async def test_red_caida_tras_agotar_reintentos_es_proveedor_error():
    modelo, cadena = armar([_error_de_conexion()], max_intentos=2)
    with pytest.raises(ProveedorError):
        await process_text(TEXTO, chain=cadena)
    assert modelo.llamadas == 2


# --------------------------------------------------------------------------- #
# process_text: entradas y logs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("texto", ["", "   ", "\n\t"])
async def test_texto_vacio_se_rechaza_sin_llamar_al_modelo(texto):
    modelo, cadena = armar([respuesta_ok()])
    with pytest.raises(ValueError):
        await process_text(texto, chain=cadena)
    assert modelo.llamadas == 0


async def test_textos_con_llaves_pasan_por_el_pipeline():
    modelo, cadena = armar([respuesta_ok()])
    await process_text('{"level":"error","msg":"redis timeout"}', chain=cadena)
    assert modelo.llamadas == 1


async def test_los_logs_muestran_validacion_y_reintentos(caplog):
    _, cadena = armar([respuesta_incompleta(), respuesta_ok()])
    with caplog.at_level(logging.INFO, logger="pipeline"):
        await process_text(TEXTO, chain=cadena)
    texto_log = caplog.text
    assert "Validación FALLIDA" in texto_log and "resumen_tecnico" in texto_log
    assert "REINTENTO" in texto_log and "intento 2" in texto_log
    assert "Validación OK" in texto_log
    assert "intentos usados: 2" in texto_log


# --------------------------------------------------------------------------- #
# Conteo de tokens
# --------------------------------------------------------------------------- #
async def test_los_logs_incluyen_los_tokens_de_cada_intento_y_el_total(caplog):
    # Un intento fallido + uno exitoso: se pagan DOS llamadas (2 x 165 = 330 tokens en total).
    _, cadena = armar([respuesta_incompleta(), respuesta_ok()])
    with caplog.at_level(logging.INFO, logger="pipeline"):
        await process_text(TEXTO, chain=cadena)
    assert "Tokens del intento 1: entrada=120, salida=45" in caplog.text
    assert "Tokens del intento 2: entrada=120, salida=45" in caplog.text
    assert "tokens: entrada=240, salida=90, total=330" in caplog.text


async def test_el_error_final_informa_los_tokens_consumidos(caplog):
    _, cadena = armar([respuesta_incompleta()], max_intentos=3)
    with caplog.at_level(logging.INFO, logger="pipeline"):
        with pytest.raises(RespuestaInvalidaError):
            await process_text(TEXTO, chain=cadena)
    assert "tokens consumidos: 495" in caplog.text  # 3 intentos x 165


async def test_el_handler_acumula_tokens_y_tolera_respuestas_sin_uso():
    handler = PipelineLogHandler()
    modelo = ModeloSimulado(respuestas=[AIMessage(content="sin uso", response_metadata={"stop_reason": "end_turn"})])
    await modelo.ainvoke("hola", config={"callbacks": [handler]})  # el mensaje no trae usage_metadata
    assert handler.tokens_totales == 0

    modelo = ModeloSimulado(respuestas=[respuesta_ok()])
    await modelo.ainvoke("hola", config={"callbacks": [handler]})
    assert (handler.tokens_entrada, handler.tokens_salida, handler.tokens_totales) == (120, 45, 165)


# --------------------------------------------------------------------------- #
# Construcción del modelo real (sin llamar a la API)
# --------------------------------------------------------------------------- #
def test_build_model_anthropic_deja_los_reintentos_a_with_retry(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    modelo = build_model("anthropic", max_tokens=256)
    assert isinstance(modelo, ChatAnthropic)
    assert modelo.model == "claude-haiku-4-5"
    assert modelo.max_retries == 0  # la única política de reintentos es .with_retry()
    assert modelo.max_tokens == 256


def test_build_model_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "clave-falsa")
    modelo = build_model("openai")
    assert modelo.model_name == "gpt-4o-mini"
    assert modelo.max_retries == 0


def test_build_model_proveedor_desconocido():
    with pytest.raises(ValueError, match="no soportado"):
        build_model("gemini")


def test_cadena_por_defecto_se_arma_desde_el_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setenv("LLM_MAX_INTENTOS", "4")
    chain_module.get_default_chain.cache_clear()
    try:
        cadena = chain_module.get_default_chain()
        assert cadena.last.max_attempt_number == 4
    finally:
        chain_module.get_default_chain.cache_clear()
