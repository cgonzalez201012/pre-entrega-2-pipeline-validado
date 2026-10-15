"""Tests del contrato Pydantic (schemas.py): no hace falta ningún LLM."""

import pytest
from pydantic import ValidationError

from schemas import SIN_TECNOLOGIAS, EntidadesTecnicas, NivelCriticidad, es_marcador_de_relleno

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL.",
}


def test_objeto_valido():
    obj = EntidadesTecnicas(**VALIDO)
    assert obj.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA


def test_salida_json_tiene_el_formato_esperado():
    datos = EntidadesTecnicas(**VALIDO).model_dump(mode="json")
    assert datos == VALIDO  # el enum sale como el string "alta"


def test_lista_de_tecnologias_vacia_es_invalida():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "tecnologias": []})


def test_tecnologias_solo_con_espacios_es_invalida():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "tecnologias": ["  ", ""]})


def test_tecnologias_se_limpian_y_sin_duplicados():
    obj = EntidadesTecnicas(**{**VALIDO, "tecnologias": [" Redis ", "redis", "", "PostgreSQL"]})
    assert obj.tecnologias == ["Redis", "PostgreSQL"]


@pytest.mark.parametrize("valor", ["critica", "urgente", "", 3, None])
def test_criticidad_fuera_del_enum_es_invalida(valor):
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "nivel_de_criticidad": valor})


@pytest.mark.parametrize("valor", ["Alta", " ALTA ", "alta"])
def test_criticidad_tolera_mayusculas_y_espacios(valor):
    assert EntidadesTecnicas(**{**VALIDO, "nivel_de_criticidad": valor}).nivel_de_criticidad is NivelCriticidad.ALTA


@pytest.mark.parametrize("campo", ["tecnologias", "nivel_de_criticidad", "resumen_tecnico"])
def test_campo_faltante_es_invalido(campo):
    datos = {k: v for k, v in VALIDO.items() if k != campo}
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**datos)


def test_resumen_demasiado_corto_o_largo_es_invalido():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "resumen_tecnico": "corto"})
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "resumen_tecnico": "x" * 501})


def test_clave_inventada_por_el_llm_es_invalida():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(**{**VALIDO, "confianza": 0.9})


# --------------------------------------------------------------------------- #
# Forma vs. verdad: marcadores de relleno (hallazgo de la prueba de estrés real)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "nombre",
    ["SIN_TECNOLOGIAS", "No aplica", "<UNKNOWN>", "N/A", "n/a", " Ninguna ", "None", "[desconocido]", "-", "Sin tecnologías"],
)
def test_detecta_marcadores_de_relleno(nombre):
    assert es_marcador_de_relleno(nombre)


@pytest.mark.parametrize("nombre", ["Redis", "Go", "R", "C#", "Node.js", "Kubernetes", "NAS", "Nginx", "Unknown Horizons DB"])
def test_las_tecnologias_reales_no_se_confunden_con_relleno(nombre):
    assert not es_marcador_de_relleno(nombre)


def test_el_escape_explicito_es_una_lista_valida_pero_sin_tecnologias_reales():
    obj = EntidadesTecnicas(**{**VALIDO, "tecnologias": [SIN_TECNOLOGIAS]})
    assert obj.tecnologias == [SIN_TECNOLOGIAS]  # cumple la forma: pasa Pydantic
    assert not obj.tiene_tecnologias_reales()  # ...pero el pipeline sabe que no hay nada real


def test_se_descartan_los_rellenos_cuando_hay_tecnologias_reales():
    obj = EntidadesTecnicas(**{**VALIDO, "tecnologias": ["Redis", "No aplica", "<UNKNOWN>", "PostgreSQL"]})
    assert obj.tecnologias == ["Redis", "PostgreSQL"]
    assert obj.tiene_tecnologias_reales()


def test_la_descripcion_del_campo_le_ofrece_al_modelo_la_salida_explicita():
    descripcion = EntidadesTecnicas.model_json_schema()["properties"]["tecnologias"]["description"]
    assert SIN_TECNOLOGIAS in descripcion
