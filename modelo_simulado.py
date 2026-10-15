"""modelo_simulado.py - Un "LLM de mentira" con respuestas guionadas.

Para qué sirve:
  * Los tests pueden probar reintentos, JSON mal formado, cortes por max_tokens, etc.
    sin gastar API ni depender de que el modelo real falle "a pedido".
  * `python main.py --simular reintento` muestra el reintento en vivo, de forma reproducible.

Es un BaseChatModel de LangChain de verdad: pasa por el mismo .with_structured_output(),
el mismo parser de Pydantic y el mismo .with_retry() que un modelo real. Lo único falso
es el texto que "responde".
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

NOMBRE_HERRAMIENTA = "EntidadesTecnicas"  # nombre de la clase Pydantic = nombre de la "tool"

# Consumo de tokens que "informa" cada respuesta simulada (como lo haría un proveedor real).
USO_SIMULADO = {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}


class ModeloSimulado(BaseChatModel):
    """Chat model que devuelve, en orden, las respuestas guionadas.

    Cada elemento de `respuestas` es un AIMessage (se devuelve) o una excepción (se lanza,
    para simular fallos de red o de autenticación). Al llegar al final repite la última.
    """

    respuestas: list[Any]
    llamadas: int = 0  # cuántas veces se llamó al modelo (útil para asserts)

    @property
    def _llm_type(self) -> str:
        return "modelo-simulado"

    def bind_tools(self, tools, **kwargs):  # lo usa with_structured_output()
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        item = self.respuestas[min(self.llamadas, len(self.respuestas) - 1)]
        self.llamadas += 1
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


# --------------------------------------------------------------------------- #
# Fábricas de respuestas guionadas
# --------------------------------------------------------------------------- #
def _tool_call(args: dict, stop_reason: str = "tool_use") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": NOMBRE_HERRAMIENTA, "args": args, "id": "call_simulada", "type": "tool_call"}],
        response_metadata={"stop_reason": stop_reason},
        usage_metadata=USO_SIMULADO,
    )


def respuesta_ok(
    tecnologias: list[str] | None = None,
    nivel: str = "alta",
    resumen: str = "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes.",
) -> AIMessage:
    """Respuesta correcta: cumple el contrato."""
    return _tool_call(
        {
            "tecnologias": tecnologias if tecnologias is not None else ["FastAPI", "Redis", "PostgreSQL"],
            "nivel_de_criticidad": nivel,
            "resumen_tecnico": resumen,
        }
    )


def respuesta_incompleta() -> AIMessage:
    """JSON incompleto: falta el campo obligatorio `resumen_tecnico`."""
    return _tool_call({"tecnologias": ["FastAPI"], "nivel_de_criticidad": "alta"})


def respuesta_con_criticidad_invalida() -> AIMessage:
    """Valor fuera del enum: 'critica' no es baja/media/alta."""
    return _tool_call(
        {"tecnologias": ["Redis"], "nivel_de_criticidad": "critica", "resumen_tecnico": "Redis se quedó sin memoria."}
    )


def respuesta_sin_herramienta() -> AIMessage:
    """El modelo contesta con texto libre en vez de usar la herramienta de salida."""
    return AIMessage(
        content="Claro! El texto habla de una API en FastAPI.",
        response_metadata={"stop_reason": "end_turn"},
        usage_metadata=USO_SIMULADO,
    )


def respuesta_sin_tecnologias(marcador: str = "SIN_TECNOLOGIAS") -> AIMessage:
    """Forma VÁLIDA pero contenido vacío: el texto no mencionaba ninguna tecnología.

    Reproduce lo que hizo el modelo real en la prueba de estrés ("No aplica", "<UNKNOWN>").
    """
    return _tool_call(
        {
            "tecnologias": [marcador],
            "nivel_de_criticidad": "baja",
            "resumen_tecnico": "El texto es una convocatoria a una reunión y no describe sistemas.",
        }
    )


def respuesta_truncada(stop_reason: str = "max_tokens") -> AIMessage:
    """El modelo se quedó sin tokens: la salida quedó cortada (JSON incompleto)."""
    return _tool_call({"tecnologias": ["FastAPI"]}, stop_reason=stop_reason)
