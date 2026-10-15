"""schemas.py - El CONTRATO de datos del pipeline (Pydantic).

Este modelo cumple dos funciones a la vez:
  1. Le dice al LLM que forma debe tener su respuesta (LangChain lo convierte en
     un JSON Schema y se lo manda al proveedor como "herramienta" a invocar).
  2. Valida lo que el LLM devuelve ANTES de que llegue al resto de la aplicacion.

Si el LLM responde algo que no cumple el contrato (falta un campo, la criticidad
no es baja/media/alta, la lista de tecnologias viene vacia...) Pydantic lanza un
ValidationError y el pipeline lo trata como "respuesta invalida" -> reintento.

Ojo con el limite de la validacion: Pydantic comprueba la FORMA, no la VERDAD. Una lista
["No aplica"] cumple "lista no vacia" y es basura. Por eso el contrato incluye una valvula
de escape explicita (SIN_TECNOLOGIAS) y el pipeline detecta los marcadores de relleno.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Valor que se le pide al modelo cuando el texto NO menciona ninguna tecnologia. Sin una
# salida explicita, la regla "la lista no puede ser vacia" lo empuja a inventar relleno.
SIN_TECNOLOGIAS = "SIN_TECNOLOGIAS"

# Marcadores de relleno que un modelo suele inventar para "cumplir" una lista no vacia.
# Es una heuristica (lista corta, en minuscula): la salida explicita es SIN_TECNOLOGIAS.
_MARCADORES_DE_RELLENO = {
    SIN_TECNOLOGIAS.lower(), "sin tecnologias", "sin tecnologías", "no aplica", "n/a", "na",
    "ninguna", "ninguno", "none", "null", "unknown", "desconocido", "no especificado", "-",
}


def es_marcador_de_relleno(nombre: str) -> bool:
    """True si `nombre` no es una tecnologia real sino un relleno ('No aplica', '<UNKNOWN>'...)."""
    limpio = nombre.strip().strip("<>[]()'\".").strip().lower()
    return limpio in _MARCADORES_DE_RELLENO


class NivelCriticidad(str, Enum):
    """Valores permitidos para la criticidad (enum cerrado: no se aceptan otros)."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """Entidades tecnicas extraidas de un texto de arquitectura, log o incidente."""

    # extra="forbid": si el LLM "alucina" una clave que no existe en el contrato,
    # falla la validacion en lugar de ignorarla en silencio.
    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        min_length=1,  # restriccion de negocio: la lista NO puede venir vacia
        description=(
            "Tecnologias, lenguajes, frameworks, bases de datos, servicios o "
            "herramientas mencionados EXPLICITAMENTE en el texto. Sin duplicados. "
            "Si el texto NO menciona ninguna tecnologia, devolver exactamente "
            "['SIN_TECNOLOGIAS'] (no inventar nada)."  # literal (sin f-string); un test lo cruza con la constante
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Criticidad de lo descripto: 'baja', 'media' o 'alta'."
    )
    resumen_tecnico: str = Field(
        min_length=10,
        max_length=500,
        description="Resumen tecnico de 1 o 2 oraciones, en espanol.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Normaliza la lista: sin espacios sobrantes, sin vacios, sin repetidos."""
        vistos: set[str] = set()
        limpias: list[str] = []
        for valor in valores:
            nombre = valor.strip()
            if nombre and nombre.lower() not in vistos:
                vistos.add(nombre.lower())
                limpias.append(nombre)
        if not limpias:
            raise ValueError("tecnologias no puede estar vacia (ni tener solo textos en blanco)")
        # Si hay tecnologias reales se descartan los marcadores de relleno. Si SOLO hay
        # marcadores, se dejan tal cual: el pipeline los detecta y lo informa como
        # "texto sin tecnologias" (un error permanente, que no tiene sentido reintentar).
        reales = [n for n in limpias if not es_marcador_de_relleno(n)]
        return reales or limpias

    def tiene_tecnologias_reales(self) -> bool:
        """False si la lista solo trae marcadores de relleno (el texto no tenia tecnologias)."""
        return any(not es_marcador_de_relleno(n) for n in self.tecnologias)

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def normalizar_criticidad(cls, valor):
        """Acepta 'Alta', ' ALTA ' etc. y los lleva a minuscula antes de validar el enum.

        Es ser tolerante con diferencias cosmeticas (no vale la pena gastar un
        reintento por una mayuscula) pero estricto con el contenido: 'critica' sigue
        siendo invalido.
        """
        if isinstance(valor, str):
            return valor.strip().lower()
        return valor
