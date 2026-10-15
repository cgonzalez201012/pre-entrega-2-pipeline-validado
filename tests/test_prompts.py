"""Tests del prompt (prompts.py)."""

from langchain_core.prompts import ChatPromptTemplate

from prompts import INSTRUCCIONES_DE_FORMATO_POR_DEFECTO, PROMPT_EXTRACCION
from schemas import SIN_TECNOLOGIAS


def test_es_un_chat_prompt_template():
    assert isinstance(PROMPT_EXTRACCION, ChatPromptTemplate)


def test_la_unica_variable_obligatoria_es_texto():
    # instrucciones_de_formato viene precargada (partial), así que no hay que pasarla.
    assert PROMPT_EXTRACCION.input_variables == ["texto"]


def test_inyecta_el_texto_y_las_instrucciones_por_defecto():
    mensajes = PROMPT_EXTRACCION.invoke({"texto": "Redis se cayó"}).to_messages()
    assert [m.type for m in mensajes] == ["system", "human"]
    assert INSTRUCCIONES_DE_FORMATO_POR_DEFECTO in mensajes[0].content
    assert "Redis se cayó" in mensajes[1].content


def test_se_pueden_reemplazar_las_instrucciones_de_formato():
    mensajes = PROMPT_EXTRACCION.invoke({"texto": "x", "instrucciones_de_formato": "SOLO TECNOLOGIAS"}).to_messages()
    assert "SOLO TECNOLOGIAS" in mensajes[0].content
    assert INSTRUCCIONES_DE_FORMATO_POR_DEFECTO not in mensajes[0].content


def test_textos_con_llaves_no_rompen_el_template():
    # Un log con JSON trae { } que un f-string/format interpretaría como variables.
    log = '{"level":"error","msg":"timeout","service":"orders"}'
    mensajes = PROMPT_EXTRACCION.invoke({"texto": log}).to_messages()
    assert log in mensajes[1].content


def test_el_prompt_ofrece_la_salida_explicita_para_textos_sin_tecnologias():
    # Sin esta salida, "lista no vacía" empuja al modelo a inventar "No aplica" / "<UNKNOWN>".
    # El valor del prompt (texto literal, sin f-strings) debe coincidir con la constante del esquema.
    mensajes = PROMPT_EXTRACCION.invoke({"texto": "x"}).to_messages()
    assert SIN_TECNOLOGIAS in mensajes[0].content
    assert SIN_TECNOLOGIAS in INSTRUCCIONES_DE_FORMATO_POR_DEFECTO
