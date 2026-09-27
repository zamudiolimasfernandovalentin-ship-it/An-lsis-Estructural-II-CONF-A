from __future__ import annotations

from typing import Any, Dict, List, Tuple


def ejercicio_1() -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Se devuelve un ejemplo válido de prueba mientras no esté disponible el PDF oficial.

    Este valor no sustituye los datos del Ejercicio N.º 1 del curso y debe reemplazarse
    por los datos reales del PDF cuando se incorporen.
    """
    nodes = [
        {"Nodo": 1, "X": 0.0, "Y": 0.0},
        {"Nodo": 2, "X": 4.0, "Y": 0.0},
        {"Nodo": 3, "X": 2.0, "Y": 3.0},
    ]
    elements = [
        {"Barra": 1, "Nodo_i": 1, "Nodo_j": 2, "E": 2.10e11, "A": 5.0e-3},
        {"Barra": 2, "Nodo_i": 2, "Nodo_j": 3, "E": 2.10e11, "A": 5.0e-3},
        {"Barra": 3, "Nodo_i": 1, "Nodo_j": 3, "E": 2.10e11, "A": 5.0e-3},
    ]
    supports = [
        {"Nodo": 1, "Ux": 0, "Uy": 0},
        {"Nodo": 2, "Ux": 1, "Uy": 0},
    ]
    loads = [
        {"Nodo": 3, "Fx": 0.0, "Fy": -50_000.0},
    ]
    return nodes, elements, supports, loads


def validate_exercise_1() -> Dict[str, Any]:
    """Función de validación del ejercicio N.º 1.

    En ausencia del PDF oficial, devuelve un estado explícito de pendiente y no afirma
    coincidencia alguna.
    """
    return {
        "status": "PENDING",
        "message": "No se puede validar numéricamente contra el PDF del curso porque el archivo adjunto no está presente en el workspace.",
        "missing_data": [
            "Valores exactos del ejercicio N.º 1 del PDF",
            "Desplazamientos esperados",
            "Reacciones esperadas",
            "Fuerzas axiales esperadas",
            "Matriz de referencia del curso",
        ],
        "tolerance": {"rtol": 1e-5, "atol": 1e-8},
    }
