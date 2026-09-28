from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def parse_numeric_expression(value: Any) -> float:
    """Parse a numeric value or a safe arithmetic expression used in model inputs."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if np.isnan(number):
            return number
        if not np.isfinite(number):
            raise ValueError("El valor debe ser un número finito.")
        return number

    expression = str(value).strip()
    if not expression:
        return float("nan")
    if "," in expression and "." not in expression:
        expression = expression.replace(",", ".")
    expression = expression.replace("×", "*").replace("√", "sqrt")
    expression = re.sub(r"raiz\s*\(", "sqrt(", expression, flags=re.IGNORECASE)
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=[+-]?(?:\d|\())", "*", expression)
    expression = re.sub(r"(?<=\d)\s*(?=sqrt\s*\()", "*", expression, flags=re.IGNORECASE)
    expression = expression.replace("^", "**")
    if len(expression) > 256:
        raise ValueError("La expresión numérica es demasiado larga.")

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("La expresión numérica es demasiado compleja.")

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
                left = evaluate(node.left)
                right = evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                    raise ValueError("El exponente debe estar entre -1000 y 1000.")
                return float(_BINARY_OPERATORS[type(node.op)](left, right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
                return float(_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.lower() == "sqrt"
                and len(node.args) == 1
                and not node.keywords
            ):
                return math.sqrt(evaluate(node.args[0]))
            raise ValueError("La expresión solo admite +, -, *, /, potencias y sqrt()/raiz().")

        result = evaluate(tree)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Expresión numérica no válida: {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError("El resultado de la expresión debe ser finito.")
    return float(result)


def format_number(value: Any, significant_digits: int = 4, latex: bool = False, signed: bool = False) -> str:
    """Format a value with significant digits and compact scientific notation at extremes."""
    number = float(value)
    if not np.isfinite(number):
        return "—" if np.isnan(number) else ("∞" if number > 0 else "−∞")
    if number == 0:
        return "+0" if signed else "0"

    sign_prefix = "+" if signed and number > 0 else ""
    def scientific(value_to_format: float) -> str:
        mantissa, exponent = f"{value_to_format:.{significant_digits - 1}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        mantissa = f"{sign_prefix}{mantissa}"
        if latex:
            return f"{mantissa} \\times 10^{{{int(exponent)}}}"
        return f"{mantissa} × 10^{int(exponent)}"

    magnitude = abs(number)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return scientific(number)

    order = math.floor(math.log10(magnitude))
    decimal_places = significant_digits - order - 1
    rounded = round(number, decimal_places)
    if 0 < abs(rounded) < 1e-3 or abs(rounded) >= 1e5:
        return scientific(rounded)
    text = f"{rounded:.{max(decimal_places, 0)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign_prefix}{text}"


@dataclass
class Node:
    id: int
    x: float
    y: float


@dataclass
class Element:
    id: int
    node_i: int
    node_j: int
    E: float
    A: float


@dataclass
class Support:
    node: int
    ux: float = 0.0
    uy: float = 0.0


@dataclass
class Load:
    node: int
    fx: float = 0.0
    fy: float = 0.0


@dataclass
class DistributedLoad:
    element_id: int
    q_i: float
    q_j: float
    direction: str = "Axial"


class Truss2D:
    """Solver basado en el método de rigidez directa para armaduras planas 2D."""

    def __init__(self, nodes: Iterable[Node] | List[dict], elements: Iterable[Element] | List[dict], supports: Optional[Iterable[Support] | List[dict]] = None, loads: Optional[Iterable[Load] | List[dict]] = None, distributed_loads: Optional[Iterable[DistributedLoad] | List[dict]] = None):
        self.nodes = self._coerce_nodes(nodes)
        self.elements = self._coerce_elements(elements)
        self.supports = self._coerce_supports(supports or [])
        self.loads = self._coerce_loads(loads or [])
        self.distributed_loads = self._coerce_distributed_loads(distributed_loads or [])
        self._validate_inputs()
        self.displacements = None
        self.reactions = None
        self.axial_forces = None
        self.stresses = None
        self.strains = None
        self.global_stiffness = self.assemble_global_matrix()
        self.global_force_vector = self.global_load_vector()
        self.free_dofs, self.restrained_dofs, self.prescribed_displacements = self.classify_dofs()
        self.partition = self.partition_matrix()

    def _coerce_nodes(self, nodes: Iterable[Node] | List[dict]) -> List[Node]:
        coerced: List[Node] = []
        for item in nodes:
            if isinstance(item, Node):
                coerced.append(item)
            else:
                coerced.append(Node(
                    id=int(parse_numeric_expression(item["Nodo"] if "Nodo" in item else item["node"])),
                    x=parse_numeric_expression(item["X"] if "X" in item else item["x"]),
                    y=parse_numeric_expression(item["Y"] if "Y" in item else item["y"]),
                ))
        return coerced

    def _coerce_elements(self, elements: Iterable[Element] | List[dict]) -> List[Element]:
        coerced: List[Element] = []
        for item in elements:
            if isinstance(item, Element):
                coerced.append(item)
            else:
                node_i = int(parse_numeric_expression(item["Nodo_i"] if "Nodo_i" in item else item["node_i"]))
                node_j = int(parse_numeric_expression(item["Nodo_j"] if "Nodo_j" in item else item["node_j"]))
                coerced.append(Element(
                    id=int(parse_numeric_expression(item["Barra"] if "Barra" in item else item["id"])),
                    node_i=node_i, node_j=node_j,
                    E=parse_numeric_expression(item["E"]), A=parse_numeric_expression(item["A"]),
                ))
        return coerced

    def _coerce_supports(self, supports: Iterable[Support] | List[dict]) -> List[Support]:
        coerced: List[Support] = []
        for item in supports:
            if isinstance(item, Support):
                coerced.append(item)
            else:
                node = int(parse_numeric_expression(item["Nodo"] if "Nodo" in item else item["node"]))
                ux = parse_numeric_expression(item.get("Ux", item.get("ux", 0.0)))
                uy = parse_numeric_expression(item.get("Uy", item.get("uy", 0.0)))
                coerced.append(Support(node=node, ux=ux, uy=uy))
        return coerced

    def _coerce_loads(self, loads: Iterable[Load] | List[dict]) -> List[Load]:
        coerced: List[Load] = []
        for item in loads:
            if isinstance(item, Load):
                coerced.append(item)
            else:
                node = int(parse_numeric_expression(item["Nodo"] if "Nodo" in item else item["node"]))
                coerced.append(Load(
                    node=node,
                    fx=parse_numeric_expression(item.get("Fx", item.get("fx", 0.0))),
                    fy=parse_numeric_expression(item.get("Fy", item.get("fy", 0.0))),
                ))
        return coerced

    def _coerce_distributed_loads(self, loads: Iterable[DistributedLoad] | List[dict]) -> List[DistributedLoad]:
        coerced: List[DistributedLoad] = []
        for item in loads:
            if isinstance(item, DistributedLoad):
                coerced.append(item)
            else:
                coerced.append(DistributedLoad(
                    element_id=int(parse_numeric_expression(item.get("Barra", item.get("element_id")))),
                    q_i=parse_numeric_expression(item.get("q_i", item.get("q_inicio", 0.0))),
                    q_j=parse_numeric_expression(item.get("q_j", item.get("q_fin", 0.0))),
                    direction=str(item.get("Tipo", item.get("direction", "Axial"))),
                ))
        return coerced

    def _node_lookup(self) -> Dict[int, Node]:
        return {node.id: node for node in self.nodes}

    def _validate_inputs(self) -> None:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Hay nodos duplicados en la estructura.")
        for node in self.nodes:
            if not np.isfinite(node.x) or not np.isfinite(node.y):
                raise ValueError(f"Las coordenadas del nodo {node.id} deben ser números finitos.")
        for element in self.elements:
            if element.node_i not in node_ids or element.node_j not in node_ids:
                raise ValueError(f"La barra {element.id} conecta nodos inexistentes.")
            if not np.isfinite(element.A) or element.A <= 0:
                raise ValueError(f"La barra {element.id} tiene área A <= 0.")
            if not np.isfinite(element.E) or element.E <= 0:
                raise ValueError(f"La barra {element.id} tiene módulo E <= 0.")
            if self.element_length(element) <= 0:
                raise ValueError(f"La barra {element.id} presenta longitud cero o nula.")
        for support in self.supports:
            if support.node not in node_ids:
                raise ValueError(f"El apoyo en el nodo {support.node} no existe.")
            if not np.isfinite(support.ux) or not np.isfinite(support.uy):
                raise ValueError(f"Los desplazamientos prescritos del apoyo en el nodo {support.node} deben ser finitos.")
        for load in self.loads:
            if load.node not in node_ids:
                raise ValueError(f"La carga en el nodo {load.node} no existe.")
            if not np.isfinite(load.fx) or not np.isfinite(load.fy):
                raise ValueError(f"Las cargas del nodo {load.node} deben ser números finitos.")
        element_ids = {element.id for element in self.elements}
        for load in self.distributed_loads:
            if load.element_id not in element_ids:
                raise ValueError(f"La carga distribuida referencia la barra inexistente {load.element_id}.")
            if load.direction.strip().lower() != "axial":
                raise ValueError("El modelo de armadura 2D solo admite cargas distribuidas axiales; las transversales requieren elementos de pórtico con rigidez flexional.")
            if not np.isfinite(load.q_i) or not np.isfinite(load.q_j):
                raise ValueError(f"La carga distribuida en la barra {load.element_id} debe tener intensidades numéricas finitas.")

    def node_dof_indices(self, node_id: int) -> Tuple[int, int]:
        node = self._node_lookup()[node_id]
        return (node.id - 1) * 2, (node.id - 1) * 2 + 1

    def element_length(self, element: Element) -> float:
        node_i = self._node_lookup()[element.node_i]
        node_j = self._node_lookup()[element.node_j]
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        return float(np.hypot(dx, dy))

    def element_direction_cosines(self, element: Element) -> Tuple[float, float]:
        node_i = self._node_lookup()[element.node_i]
        node_j = self._node_lookup()[element.node_j]
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        L = self.element_length(element)
        if L == 0:
            raise ValueError(f"La barra {element.id} tiene longitud cero.")
        c = dx / L
        s = dy / L
        return c, s

    def element_transformation_matrix(self, element: Element) -> np.ndarray:
        c, s = self.element_direction_cosines(element)
        return np.array(
            [[c, s, 0.0, 0.0], [-s, c, 0.0, 0.0],
             [0.0, 0.0, c, s], [0.0, 0.0, -s, c]],
            dtype=float,
        )

    def element_stiffness_local(self, element: Element) -> np.ndarray:
        L = self.element_length(element)
        k = (element.E * element.A / L) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        return k

    def element_stiffness_local_4x4(self, element: Element) -> np.ndarray:
        factor = element.E * element.A / self.element_length(element)
        return factor * np.array(
            [[1.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [-1.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )

    def element_distributed_load_local(self, load: DistributedLoad) -> np.ndarray:
        element = next(element for element in self.elements if element.id == load.element_id)
        length = self.element_length(element)
        force_i = length * (2.0 * load.q_i + load.q_j) / 6.0
        force_j = length * (load.q_i + 2.0 * load.q_j) / 6.0
        return np.array([force_i, 0.0, force_j, 0.0], dtype=float)

    def element_distributed_load_global(self, load: DistributedLoad) -> np.ndarray:
        element = next(element for element in self.elements if element.id == load.element_id)
        c, s = self.element_direction_cosines(element)
        local = self.element_distributed_load_local(load)
        return np.array([c * local[0], s * local[0], c * local[2], s * local[2]], dtype=float)

    def element_stiffness_global(self, element: Element) -> np.ndarray:
        transform = self.element_transformation_matrix(element)
        local_stiffness = self.element_stiffness_local_4x4(element)
        Ke = transform.T @ local_stiffness @ transform
        if not np.allclose(Ke, Ke.T, atol=1e-10, rtol=1e-8):
            raise ValueError(f"La matriz de rigidez local de la barra {element.id} no es simétrica.")
        return Ke

    def global_load_vector(self) -> np.ndarray:
        n_dof = len(self.nodes) * 2
        F = np.zeros(n_dof, dtype=float)
        for load in self.loads:
            dofx, dofy = self.node_dof_indices(load.node)
            F[dofx] += float(load.fx)
            F[dofy] += float(load.fy)
        for load in self.distributed_loads:
            element = next(element for element in self.elements if element.id == load.element_id)
            dofs = [*self.node_dof_indices(element.node_i), *self.node_dof_indices(element.node_j)]
            F[dofs] += self.element_distributed_load_global(load)
        return F

    def assemble_global_matrix(self) -> np.ndarray:
        n_dof = len(self.nodes) * 2
        K = np.zeros((n_dof, n_dof), dtype=float)
        for element in self.elements:
            Ke = self.element_stiffness_global(element)
            node_i = element.node_i
            node_j = element.node_j
            dof_i = [2 * (node_i - 1), 2 * (node_i - 1) + 1, 2 * (node_j - 1), 2 * (node_j - 1) + 1]
            for i in range(4):
                for j in range(4):
                    K[dof_i[i], dof_i[j]] += Ke[i, j]
        return K

    def classify_dofs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_dofs = np.arange(len(self.nodes) * 2, dtype=int)
        fixed_dofs = []
        free_dofs = []
        prescribed = np.zeros(len(self.nodes) * 2, dtype=float)

        for support in self.supports:
            dofx, dofy = self.node_dof_indices(support.node)
            if support.ux == 0:
                fixed_dofs.append(dofx)
                prescribed[dofx] = 0.0
            elif support.ux == 1:
                free_dofs.append(dofx)
            else:
                fixed_dofs.append(dofx)
                prescribed[dofx] = float(support.ux)

            if support.uy == 0:
                fixed_dofs.append(dofy)
                prescribed[dofy] = 0.0
            elif support.uy == 1:
                free_dofs.append(dofy)
            else:
                fixed_dofs.append(dofy)
                prescribed[dofy] = float(support.uy)

        fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
        free_dofs = np.array(sorted(set(np.setdiff1d(total_dofs, fixed_dofs))), dtype=int)
        if free_dofs.size == 0:
            raise ValueError("La estructura no tiene grados de libertad libres; no puede resolverse.")
        if fixed_dofs.size == 0:
            raise ValueError("La estructura no tiene apoyos suficientes; no hay grados de libertad restringidos.")
        return free_dofs, fixed_dofs, prescribed

    def partition_matrix(self):
        K = self.global_stiffness
        F = self.global_load_vector()
        L = self.free_dofs
        R = self.restrained_dofs
        prescribed = self.prescribed_displacements

        K_LL = K[np.ix_(L, L)]
        K_LR = K[np.ix_(L, R)]
        K_RL = K[np.ix_(R, L)]
        K_RR = K[np.ix_(R, R)]
        F_L = F[L]
        F_R = F[R]
        D_R = prescribed[R]
        return {
            "K_LL": K_LL,
            "K_LR": K_LR,
            "K_RL": K_RL,
            "K_RR": K_RR,
            "F_L": F_L,
            "F_R": F_R,
            "D_R": D_R,
            "D_L": None,
            "F_RR": None,
        }

    def solve_displacements(self) -> np.ndarray:
        K_LL = self.partition["K_LL"]
        K_LR = self.partition["K_LR"]
        F_L = self.partition["F_L"]
        D_R = self.partition["D_R"]

        if np.linalg.matrix_rank(K_LL) < K_LL.shape[0]:
            raise ValueError("La estructura es singular o inestable; no se puede resolver el sistema.")

        D_L = np.linalg.solve(K_LL, F_L - K_LR @ D_R)
        D_total = np.zeros(len(self.nodes) * 2, dtype=float)
        D_total[self.free_dofs] = D_L
        D_total[self.restrained_dofs] = D_R
        self.displacements = D_total
        self.partition["D_L"] = D_L
        return D_total

    def calculate_reactions(self) -> np.ndarray:
        if self.displacements is None:
            self.solve_displacements()
        residual = self.global_stiffness @ self.displacements - self.global_load_vector()
        self.reactions = residual[self.restrained_dofs]
        self.partition["F_RR"] = self.reactions
        return self.reactions

    def calculate_axial_forces(self) -> List[dict]:
        if self.displacements is None:
            self.solve_displacements()
        results: List[dict] = []
        for element in self.elements:
            c, s = self.element_direction_cosines(element)
            phi = float(np.arctan2(s, c))
            dof_i = [2 * (element.node_i - 1), 2 * (element.node_i - 1) + 1, 2 * (element.node_j - 1), 2 * (element.node_j - 1) + 1]
            D_e = self.displacements[dof_i]
            transform = self.element_transformation_matrix(element)
            local_stiffness = self.element_stiffness_local_4x4(element)
            global_stiffness = transform.T @ local_stiffness @ transform
            local_displacements = transform @ D_e
            delta = local_displacements[2] - local_displacements[0]
            local_end_forces = local_stiffness @ local_displacements
            member_loads = [load for load in self.distributed_loads if load.element_id == element.id]
            equivalent_local_load = np.zeros(4, dtype=float)
            for member_load in member_loads:
                equivalent_local_load += self.element_distributed_load_local(member_load)
            internal_end_forces = local_end_forces - equivalent_local_load
            force_i = -internal_end_forces[0]
            force_j = internal_end_forces[2]
            N = (force_i + force_j) / 2.0
            if force_i > 1e-9 and force_j > 1e-9:
                estado = "TRACCIÓN"
            elif force_i < -1e-9 and force_j < -1e-9:
                estado = "COMPRESIÓN"
            elif abs(force_i) > 1e-9 or abs(force_j) > 1e-9:
                estado = "ESFUERZO VARIABLE"
            else:
                estado = "FUERZA NULA"
            results.append({
                "Barra": element.id,
                "Longitud": self.element_length(element),
                "phi": phi,
                "c": c,
                "s": s,
                "De": D_e,
                "Tg": transform,
                "KL": local_stiffness,
                "Kg": global_stiffness,
                "delta": delta,
                "d_local": local_displacements,
                "f_L": local_end_forces,
                "carga_equivalente_local": equivalent_local_load,
                "fuerzas_internas_locales": internal_end_forces,
                "N": N,
                "N_i": force_i,
                "N_j": force_j,
                "estado": estado,
            })
        self.axial_forces = results
        return results

    def calculate_strains(self) -> List[float]:
        if self.axial_forces is None:
            self.calculate_axial_forces()
        strains = []
        for element in self.elements:
            c, s = self.element_direction_cosines(element)
            dof_i = [2 * (element.node_i - 1), 2 * (element.node_i - 1) + 1, 2 * (element.node_j - 1), 2 * (element.node_j - 1) + 1]
            D_e = self.displacements[dof_i]
            delta = np.array([-c, -s, c, s], dtype=float) @ D_e
            strains.append(delta / self.element_length(element))
        self.strains = strains
        return strains

    def calculate_stresses(self) -> List[float]:
        if self.strains is None:
            self.calculate_strains()
        stresses = []
        for element, strain in zip(self.elements, self.strains):
            stresses.append(element.E * strain)
        self.stresses = stresses
        return stresses


def parse_supports_from_df(df) -> List[Support]:
    supports = []
    for _, row in df.iterrows():
        node = int(parse_numeric_expression(row["Nodo"]))
        ux_val = row.get("Ux", 0.0)
        uy_val = row.get("Uy", 0.0)
        supports.append(Support(node=node, ux=parse_numeric_expression(ux_val), uy=parse_numeric_expression(uy_val)))
    return supports


def parse_loads_from_df(df) -> List[Load]:
    loads = []
    for _, row in df.iterrows():
        node = int(parse_numeric_expression(row["Nodo"]))
        loads.append(Load(node=node, fx=parse_numeric_expression(row.get("Fx", 0.0)), fy=parse_numeric_expression(row.get("Fy", 0.0))))
    return loads


def parse_distributed_loads_from_df(df) -> List[DistributedLoad]:
    loads = []
    for _, row in df.iterrows():
        loads.append(DistributedLoad(
            element_id=int(parse_numeric_expression(row["Barra"])),
            q_i=parse_numeric_expression(row.get("q_i", 0.0)),
            q_j=parse_numeric_expression(row.get("q_j", 0.0)),
            direction=str(row.get("Tipo", "Axial")),
        ))
    return loads


def parse_nodes_from_df(df) -> List[Node]:
    nodes = []
    for _, row in df.iterrows():
        nodes.append(Node(
            id=int(parse_numeric_expression(row["Nodo"])),
            x=parse_numeric_expression(row["X"]),
            y=parse_numeric_expression(row["Y"]),
        ))
    return nodes


def parse_elements_from_df(df) -> List[Element]:
    elements = []
    for _, row in df.iterrows():
        elements.append(Element(
            id=int(parse_numeric_expression(row["Barra"])),
            node_i=int(parse_numeric_expression(row["Nodo_i"])),
            node_j=int(parse_numeric_expression(row["Nodo_j"])),
            E=parse_numeric_expression(row["E"]),
            A=parse_numeric_expression(row["A"]),
        ))
    return elements


def validate_exercise_1() -> Dict[str, Any]:
    """Placeholder validation: the official PDF is not in the workspace, so the comparison must be explicit and not claimed as successful."""
    return {
        "status": "PENDING",
        "message": "La validación definitiva contra el PDF del curso no puede ejecutarse porque el archivo adjunto no está disponible en el workspace.",
        "missing_data": [
            "Valores numéricos exactos del Ejercicio N.º 1 del PDF",
            "Desplazamientos esperados publicados",
            "Reacciones esperadas publicadas",
            "Fuerzas axiales esperadas publicadas",
            "Matrices de referencia del documento",
        ],
        "tolerance": {"rtol": 1e-5, "atol": 1e-8},
    }
