"""main.py - Mini-script de prueba del pipeline de extracción de entidades técnicas.

Uso:
    python main.py                          # ejemplo de arquitectura con el modelo real
    python main.py --ejemplo log            # ejemplo de log de error
    python main.py --texto "Tu texto..."    # tu propio texto
    python main.py --estres                 # textos ambiguos: ¿se recupera o falla en forma controlada?
    python main.py --provider openai        # cambia de proveedor (por defecto: LLM_PROVIDER o anthropic)

    # Sin API key ni internet: modelo simulado que falla a propósito
    python main.py --simular reintento      # 2 respuestas inválidas y a la 3ra se recupera
    python main.py --simular sin-recuperacion   # siempre inválido: agota los reintentos
    python main.py --simular truncado       # corte por max_tokens: NO reintenta
    python main.py --simular sin-entidades  # el texto no menciona tecnologías: NO reintenta
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from chain import PipelineError, build_chain, build_model, process_text
from modelo_simulado import (
    ModeloSimulado,
    respuesta_con_criticidad_invalida,
    respuesta_incompleta,
    respuesta_ok,
    respuesta_sin_tecnologias,
    respuesta_truncada,
)

EJEMPLOS = {
    "arquitectura": (
        "Nuestro backend es una API en FastAPI que consulta PostgreSQL y usa Redis como caché de "
        "sesiones. En el pico de ventas del Black Friday aparecieron timeouts porque el pool de "
        "conexiones a PostgreSQL se agota cuando hay muchas peticiones concurrentes."
    ),
    "log": (
        "2026-10-14 14:02:11 ERROR [orders-service] psycopg2.OperationalError: FATAL: remaining "
        "connection slots are reserved for non-replication superuser connections. "
        "2026-10-14 14:02:12 WARN [orders-service] Redis cache miss rate 98%. "
        "Healthcheck /health -> 503 (3 réplicas caídas en Kubernetes)."
    ),
}

# Textos pensados para tensionar el contrato (ver README, sección "Prueba de estrés").
TEXTOS_DE_ESTRES = {
    "ambiguo": "Che, se cayó otra vez lo de siempre. Creo que es la base o la red, no sé, mirá y avisame.",
    "sin tecnologías": "Mañana a las 10 hay reunión de equipo para revisar el presupuesto del trimestre.",
    "casi vacío": "ok",
}

SIMULACIONES = {
    "reintento": [respuesta_incompleta(), respuesta_con_criticidad_invalida(), respuesta_ok()],
    "sin-recuperacion": [respuesta_incompleta()],  # repite la última: siempre inválida
    "truncado": [respuesta_truncada(), respuesta_ok()],
    "sin-entidades": [respuesta_sin_tecnologias(), respuesta_ok()],  # texto sin tecnologías: NO reintenta
}


def configurar_logs() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # LangChain registra su propio traceback cuando ve un max_tokens; nuestro pipeline ya
    # lo explica con un mensaje claro, así que silenciamos ese duplicado.
    logging.getLogger("langchain_core.output_parsers").setLevel(logging.CRITICAL)


async def procesar_e_imprimir(titulo: str, texto: str, chain) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
    print(f"Texto: {texto}\n")
    try:
        resultado = await process_text(texto, chain=chain)
    except (PipelineError, ValueError) as exc:
        # El pipeline NO se cae: devuelve un error controlado que la app puede manejar.
        print(f"\n❌ Falla controlada ({type(exc).__name__}): {exc}")
        return
    print("\n✅ Resultado validado:")
    print(resultado.model_dump_json(indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de extracción de entidades técnicas")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=os.getenv("LLM_PROVIDER", "anthropic"))
    parser.add_argument("--ejemplo", choices=list(EJEMPLOS), default="arquitectura")
    parser.add_argument("--texto", help="Texto propio a analizar")
    parser.add_argument("--estres", action="store_true", help="Prueba de estrés con textos ambiguos")
    parser.add_argument("--simular", choices=list(SIMULACIONES), help="Usa un modelo simulado (sin API)")
    args = parser.parse_args()

    load_dotenv()
    configurar_logs()

    if args.simular:
        modelo = ModeloSimulado(respuestas=SIMULACIONES[args.simular])
        # esperar_entre_intentos=False: en la simulación no hace falta el backoff.
        chain = build_chain(modelo, esperar_entre_intentos=False)
        print(f"[Modo simulado: '{args.simular}' - sin llamar a ninguna API]")
    else:
        variable_de_clave = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[args.provider]
        if not os.getenv(variable_de_clave):
            raise SystemExit(
                f"Falta {variable_de_clave}. Copiá .env.example como .env y completá tu clave "
                "(o probá sin API con: python main.py --simular reintento)."
            )
        modelo = build_model(
            provider=args.provider,
            model=os.getenv("LLM_MODEL") or None,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )
        chain = build_chain(modelo, max_intentos=int(os.getenv("LLM_MAX_INTENTOS", "3")))
        print(f"[Proveedor: {args.provider} - modelo: {getattr(modelo, 'model', None) or getattr(modelo, 'model_name', '?')}]")

    if args.estres:
        for nombre, texto in TEXTOS_DE_ESTRES.items():
            await procesar_e_imprimir(f"PRUEBA DE ESTRÉS: {nombre}", texto, chain)
    elif args.texto:
        await procesar_e_imprimir("TEXTO PROPIO", args.texto, chain)
    elif args.simular == "sin-entidades":
        await procesar_e_imprimir("TEXTO SIN TECNOLOGÍAS", TEXTOS_DE_ESTRES["sin tecnologías"], chain)
    else:
        texto = EJEMPLOS[args.ejemplo] if not args.simular else EJEMPLOS["arquitectura"]
        await procesar_e_imprimir(f"EJEMPLO: {args.ejemplo}", texto, chain)

    if args.simular:
        print(f"\n(El modelo simulado fue llamado {modelo.llamadas} vez/veces)")


if __name__ == "__main__":
    asyncio.run(main())
