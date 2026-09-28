from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from examples import ejercicio_1
from reports import export_analysis_excel, generate_pdf_report
from structural_solver import Truss2D, format_number, parse_distributed_loads_from_df, parse_elements_from_df, parse_loads_from_df, parse_nodes_from_df, parse_numeric_expression, parse_supports_from_df, validate_exercise_1


st.set_page_config(page_title="Análisis Estructural II – Método de Rigidez", layout="wide")

FORCE_TO_NEWTONS = {"tonf": 9806.65, "kgf": 9.80665, "kN": 1000.0, "N": 1.0}
LENGTH_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
EXPRESSION_HELP = "Admite expresiones como 2*sqrt(3), 2raiz(3), 10^6, 2x10^6 y fracciones."
EDITOR_KEYS = ("tabla_nodos", "tabla_barras", "tabla_apoyos", "tabla_cargas", "tabla_cargas_distribuidas")
structural_canvas = components.declare_component(
    "structural_canvas", path=str(Path(__file__).parent / "canvas_component")
)


def as_dataframe(rows: Any, columns: List[str]) -> pd.DataFrame:
    """Normaliza estructuras como listas de diccionarios o tuplas antes de construir el DataFrame."""
    if rows is None:
        return pd.DataFrame(columns=columns)
    if isinstance(rows, pd.DataFrame):
        out = rows.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        return out[columns]
    if isinstance(rows, dict):
        rows = [rows]
    if isinstance(rows, (list, tuple)):
        if not rows:
            return pd.DataFrame(columns=columns)
        if all(isinstance(item, dict) for item in rows):
            return pd.DataFrame(rows, columns=columns)
        if all(isinstance(item, (list, tuple)) for item in rows):
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame([rows], columns=columns)
    return pd.DataFrame([rows], columns=columns)


def parse_numeric_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    parsed = frame.copy()
    for column in columns:
        if column in parsed:
            parsed[column] = parsed[column].map(
                lambda value: np.nan if pd.isna(value) else parse_numeric_expression(value)
            )
    return parsed


def parse_numeric_for_display(value: Any) -> float:
    try:
        return parse_numeric_expression(value)
    except (TypeError, ValueError):
        return float("nan")


def numeric_columns_for_display(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    parsed = frame.copy()
    for column in columns:
        if column in parsed:
            parsed[column] = parsed[column].map(parse_numeric_for_display)
    return parsed


def expression_editor_frame(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    editable = frame.copy()
    for column in columns:
        if column in editable:
            editable[column] = editable[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            ).astype(object)
    return editable


def format_dataframe(frame: pd.DataFrame, signed_columns: List[str] | None = None) -> pd.io.formats.style.Styler:
    signed_columns = signed_columns or []
    formatters = {
        column: (lambda value, signed=column in signed_columns: format_number(value, signed=signed))
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    return frame.style.format(formatters, na_rep="—", precision=4)


def initialize_state() -> None:
    if "exercise_data" not in st.session_state:
        st.session_state.exercise_data = ejercicio_1()
    if "nodes_df" not in st.session_state:
        st.session_state.nodes_df = as_dataframe(st.session_state.exercise_data[0], ["Nodo", "X", "Y"])
    if "elements_df" not in st.session_state:
        st.session_state.elements_df = as_dataframe(st.session_state.exercise_data[1], ["Barra", "Nodo_i", "Nodo_j", "E", "A"])
    if "supports_df" not in st.session_state:
        st.session_state.supports_df = as_dataframe(st.session_state.exercise_data[2], ["Nodo", "Ux", "Uy"])
    if "loads_df" not in st.session_state:
        st.session_state.loads_df = as_dataframe(st.session_state.exercise_data[3], ["Nodo", "Fx", "Fy"])
    if "distributed_loads_df" not in st.session_state:
        st.session_state.distributed_loads_df = pd.DataFrame(columns=["Barra", "Tipo", "q_i", "q_j"])
    if "solution" not in st.session_state:
        st.session_state.solution = None
    if "validation" not in st.session_state:
        st.session_state.validation = validate_exercise_1()
    if "structure_name_input" not in st.session_state:
        st.session_state.structure_name_input = "Ejercicio N.º 1"
    if "exercise_choice" not in st.session_state:
        st.session_state.exercise_choice = "Ejercicio N.º 1"
    if "force_unit" not in st.session_state:
        st.session_state.force_unit = "N"
    if "length_unit" not in st.session_state:
        st.session_state.length_unit = "m"
    if "current_force_unit" not in st.session_state:
        st.session_state.current_force_unit = st.session_state.force_unit
    if "current_length_unit" not in st.session_state:
        st.session_state.current_length_unit = st.session_state.length_unit
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Integrada"
    if "solution_error" not in st.session_state:
        st.session_state.solution_error = None


def clear_editor_widget_state() -> None:
    for key in (*EDITOR_KEYS, "structure_plot"):
        st.session_state.pop(key, None)


def apply_data_editor_state(frame: pd.DataFrame, editor_state: Any) -> pd.DataFrame:
    if isinstance(editor_state, pd.DataFrame):
        return editor_state.copy()
    if not isinstance(editor_state, dict):
        return frame.copy()

    updated = frame.copy().reset_index(drop=True)
    for row_key, changes in editor_state.get("edited_rows", {}).items():
        row_index = int(row_key)
        if not 0 <= row_index < len(updated):
            continue
        for column, value in changes.items():
            if column in updated.columns:
                updated.at[row_index, column] = value

    deleted_rows = sorted({int(index) for index in editor_state.get("deleted_rows", [])}, reverse=True)
    for row_index in deleted_rows:
        if 0 <= row_index < len(updated):
            updated = updated.drop(index=row_index)
    updated = updated.reset_index(drop=True)

    added_rows = editor_state.get("added_rows", [])
    if added_rows:
        additions = pd.DataFrame(added_rows).reindex(columns=updated.columns)
        updated = pd.concat([updated, additions], ignore_index=True)
    return updated


def load_selected_exercise() -> None:
    if st.session_state.exercise_choice == "Ejercicio N.º 1":
        nodes, elements, supports, loads = (
            as_dataframe(rows, columns)
            for rows, columns in zip(
                ejercicio_1(),
                (["Nodo", "X", "Y"], ["Barra", "Nodo_i", "Nodo_j", "E", "A"], ["Nodo", "Ux", "Uy"], ["Nodo", "Fx", "Fy"]),
            )
        )
        force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
        length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
        nodes[["X", "Y"]] = nodes[["X", "Y"]] / length_factor
        elements["E"] = elements["E"] / (force_factor / length_factor**2)
        elements["A"] = elements["A"] / length_factor**2
        loads[["Fx", "Fy"]] = loads[["Fx", "Fy"]] / force_factor
        data = (nodes, elements, supports, loads)
        name = "Ejercicio N.º 1"
    else:
        data = ([], [], [], [])
        name = "Estructura personalizada"

    st.session_state.exercise_data = data
    st.session_state.nodes_df = as_dataframe(data[0], ["Nodo", "X", "Y"])
    st.session_state.elements_df = as_dataframe(data[1], ["Barra", "Nodo_i", "Nodo_j", "E", "A"])
    st.session_state.supports_df = as_dataframe(data[2], ["Nodo", "Ux", "Uy"])
    st.session_state.loads_df = as_dataframe(data[3], ["Nodo", "Fx", "Fy"])
    st.session_state.distributed_loads_df = pd.DataFrame(columns=["Barra", "Tipo", "q_i", "q_j"])
    st.session_state.structure_name_input = name
    st.session_state.solution = None
    st.session_state.solution_error = None
    st.session_state.validation = validate_exercise_1()
    st.session_state.pop("selected_graph_node", None)
    clear_editor_widget_state()


def reset_structure() -> None:
    load_selected_exercise()


def change_units() -> None:
    old_force = FORCE_TO_NEWTONS[st.session_state.current_force_unit]
    new_force = FORCE_TO_NEWTONS[st.session_state.force_unit]
    old_length = LENGTH_TO_METERS[st.session_state.current_length_unit]
    new_length = LENGTH_TO_METERS[st.session_state.length_unit]

    try:
        loads = parse_numeric_columns(st.session_state.loads_df, ["Fx", "Fy"])
        nodes = parse_numeric_columns(st.session_state.nodes_df, ["X", "Y"])
        elements = parse_numeric_columns(st.session_state.elements_df, ["E", "A"])
        distributed_loads = parse_numeric_columns(st.session_state.distributed_loads_df, ["q_i", "q_j"])
    except ValueError as exc:
        st.session_state.force_unit = st.session_state.current_force_unit
        st.session_state.length_unit = st.session_state.current_length_unit
        st.session_state.solution_error = str(exc)
        return

    for column in ("Fx", "Fy"):
        if column in loads:
            loads[column] = loads[column] * old_force / new_force
    for column in ("X", "Y"):
        if column in nodes:
            nodes[column] = nodes[column] * old_length / new_length
    if "A" in elements:
        elements["A"] = elements["A"] * old_length**2 / new_length**2
    if "E" in elements:
        elements["E"] = elements["E"] * (old_force / old_length**2) / (new_force / new_length**2)
    for column in ("q_i", "q_j"):
        if column in distributed_loads:
            distributed_loads[column] = distributed_loads[column] * (old_force / old_length) / (new_force / new_length)

    st.session_state.loads_df = loads
    st.session_state.nodes_df = nodes
    st.session_state.elements_df = elements
    st.session_state.distributed_loads_df = distributed_loads
    st.session_state.current_force_unit = st.session_state.force_unit
    st.session_state.current_length_unit = st.session_state.length_unit
    st.session_state.solution = None
    st.session_state.solution_error = None
    clear_editor_widget_state()


def current_inputs_in_si() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    nodes = parse_numeric_columns(st.session_state.nodes_df, ["X", "Y"])
    elements = parse_numeric_columns(st.session_state.elements_df, ["E", "A"])
    supports = st.session_state.supports_df.copy()
    loads = parse_numeric_columns(st.session_state.loads_df, ["Fx", "Fy"])
    distributed_loads = parse_numeric_columns(st.session_state.distributed_loads_df, ["q_i", "q_j"])

    for column in ("X", "Y"):
        nodes[column] = nodes[column] * length_factor
    elements["E"] = elements["E"] * force_factor / length_factor**2
    elements["A"] = elements["A"] * length_factor**2
    for column in ("Fx", "Fy"):
        loads[column] = loads[column] * force_factor
    for column in ("q_i", "q_j"):
        distributed_loads[column] = distributed_loads[column] * force_factor / length_factor
    return nodes, elements, supports, loads, distributed_loads


def solve_structure() -> None:
    try:
        nodes, elements, supports, loads, distributed_loads = current_inputs_in_si()
        solver = Truss2D(
            nodes=parse_nodes_from_df(nodes),
            elements=parse_elements_from_df(elements),
            supports=parse_supports_from_df(supports),
            loads=parse_loads_from_df(loads),
            distributed_loads=parse_distributed_loads_from_df(distributed_loads),
        )
        solver.solve_displacements()
        solver.calculate_reactions()
        solver.calculate_axial_forces()
        solver.calculate_strains()
        solver.calculate_stresses()
        st.session_state.solution = solver
        st.session_state.solution_error = None
    except Exception as exc:
        st.session_state.solution = None
        st.session_state.solution_error = str(exc)


def update_graph_node_coordinates() -> None:
    node_id = st.session_state.get("selected_graph_node")
    if node_id is None or "Nodo" not in st.session_state.nodes_df:
        return
    node_rows = st.session_state.nodes_df.index[st.session_state.nodes_df["Nodo"] == node_id]
    if len(node_rows) == 0:
        return
    row_index = node_rows[0]
    st.session_state.nodes_df.loc[row_index, "X"] = st.session_state.graph_node_x
    st.session_state.nodes_df.loc[row_index, "Y"] = st.session_state.graph_node_y
    st.session_state.solution = None
    clear_editor_widget_state()


def mark_structure_changed() -> None:
    st.session_state.solution = None


def sync_editor_dataframe(editor_key: str, dataframe_key: str) -> None:
    edited_frame = st.session_state.get(editor_key)
    current_frame = st.session_state.get(dataframe_key)
    if isinstance(current_frame, pd.DataFrame):
        st.session_state[dataframe_key] = apply_data_editor_state(current_frame, edited_frame)
    st.session_state.pop(editor_key, None)
    st.session_state.solution = None


def render_structure_canvas() -> None:
    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    existing_elements = st.session_state.elements_df
    valid_elements = existing_elements.dropna(subset=["E", "A"]) if not existing_elements.empty else existing_elements
    default_e = parse_numeric_for_display(valid_elements.iloc[0]["E"]) if not valid_elements.empty else np.nan
    default_a = parse_numeric_for_display(valid_elements.iloc[0]["A"]) if not valid_elements.empty else np.nan
    if not np.isfinite(default_e) or default_e <= 0:
        default_e = 210e9 / (force_factor / length_factor**2)
    if not np.isfinite(default_a) or default_a <= 0:
        default_a = 5e-3 / length_factor**2
    canvas_nodes = numeric_columns_for_display(st.session_state.nodes_df, ["Nodo", "X", "Y"])
    canvas_elements = numeric_columns_for_display(st.session_state.elements_df, ["Barra", "Nodo_i", "Nodo_j", "E", "A"])
    canvas_supports = numeric_columns_for_display(st.session_state.supports_df, ["Nodo", "Ux", "Uy"])
    canvas_loads = numeric_columns_for_display(st.session_state.loads_df, ["Nodo", "Fx", "Fy"])
    event = structural_canvas(
        nodes=json.loads(canvas_nodes.to_json(orient="records")),
        elements=json.loads(canvas_elements.to_json(orient="records")),
        supports=json.loads(canvas_supports.to_json(orient="records")),
        loads=json.loads(canvas_loads.to_json(orient="records")),
        default_e=default_e,
        default_a=default_a,
        length_unit=st.session_state.length_unit,
        force_unit=st.session_state.force_unit,
        key="structure_canvas",
        default=None,
    )
    if not isinstance(event, dict) or event.get("event_id") == st.session_state.get("canvas_last_event_id"):
        return

    st.session_state.canvas_last_event_id = event.get("event_id")
    updated_nodes = as_dataframe(event.get("nodes"), ["Nodo", "X", "Y"])
    for column in ("X", "Y"):
        updated_nodes[column] = pd.to_numeric(updated_nodes[column], errors="coerce").astype(float)
    st.session_state.nodes_df = updated_nodes
    st.session_state.elements_df = as_dataframe(event.get("elements"), ["Barra", "Nodo_i", "Nodo_j", "E", "A"])
    st.session_state.supports_df = as_dataframe(event.get("supports"), ["Nodo", "Ux", "Uy"])
    st.session_state.loads_df = as_dataframe(event.get("loads"), ["Nodo", "Fx", "Fy"])
    st.session_state.solution = None
    st.session_state.solution_error = None
    clear_editor_widget_state()
    st.rerun()


def render_data_editors() -> None:
    st.subheader("Nodos")
    st.session_state.nodes_df = st.data_editor(
        expression_editor_frame(st.session_state.nodes_df, ["X", "Y"]), key="tabla_nodos", num_rows="dynamic", width="stretch",
        column_config={
            "X": st.column_config.TextColumn(f"X ({st.session_state.length_unit})", help=EXPRESSION_HELP),
            "Y": st.column_config.TextColumn(f"Y ({st.session_state.length_unit})", help=EXPRESSION_HELP),
        },
        on_change=sync_editor_dataframe, args=("tabla_nodos", "nodes_df"),
    )
    st.subheader("Barras y propiedades")
    st.session_state.elements_df = st.data_editor(
        expression_editor_frame(st.session_state.elements_df, ["E", "A"]), key="tabla_barras", num_rows="dynamic", width="stretch",
        column_config={
            "E": st.column_config.TextColumn(f"E ({st.session_state.force_unit}/{st.session_state.length_unit}²)", help=EXPRESSION_HELP),
            "A": st.column_config.TextColumn(f"A ({st.session_state.length_unit}²)", help=EXPRESSION_HELP),
        },
        on_change=sync_editor_dataframe, args=("tabla_barras", "elements_df"),
    )
    st.subheader("Apoyos")
    st.session_state.supports_df = st.data_editor(
        expression_editor_frame(st.session_state.supports_df, ["Ux", "Uy"]), key="tabla_apoyos", num_rows="dynamic", width="stretch",
        column_config={
            "Ux": st.column_config.TextColumn("Ux (0=restringido, 1=libre)", help=EXPRESSION_HELP),
            "Uy": st.column_config.TextColumn("Uy (0=restringido, 1=libre)", help=EXPRESSION_HELP),
        },
        on_change=sync_editor_dataframe, args=("tabla_apoyos", "supports_df"),
    )
    st.subheader("Cargas")
    st.session_state.loads_df = st.data_editor(
        expression_editor_frame(st.session_state.loads_df, ["Fx", "Fy"]), key="tabla_cargas", num_rows="dynamic", width="stretch",
        column_config={
            "Fx": st.column_config.TextColumn(f"Fx ({st.session_state.force_unit})", help=EXPRESSION_HELP),
            "Fy": st.column_config.TextColumn(f"Fy ({st.session_state.force_unit})", help=EXPRESSION_HELP),
        },
        on_change=sync_editor_dataframe, args=("tabla_cargas", "loads_df"),
    )
    st.subheader("Cargas distribuidas axiales")
    st.caption(
        f"Intensidad q en {st.session_state.force_unit}/{st.session_state.length_unit}. "
        "El modelo de armadura no admite cargas transversales ni momentos de extremo."
    )
    st.session_state.distributed_loads_df = st.data_editor(
        expression_editor_frame(st.session_state.distributed_loads_df, ["q_i", "q_j"]),
        key="tabla_cargas_distribuidas", num_rows="dynamic", width="stretch",
        column_config={
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Axial"], required=True),
            "q_i": st.column_config.TextColumn(f"q_i ({st.session_state.force_unit}/{st.session_state.length_unit})", help=EXPRESSION_HELP),
            "q_j": st.column_config.TextColumn(f"q_j ({st.session_state.force_unit}/{st.session_state.length_unit})", help=EXPRESSION_HELP),
        },
        on_change=sync_editor_dataframe, args=("tabla_cargas_distribuidas", "distributed_loads_df"),
    )


def render_structure_plot() -> None:
    fig = build_structure_figure()
    chart_state = st.plotly_chart(
        fig, key="structure_plot", on_select="rerun", selection_mode="points", width="stretch",
    )
    selected_node = None
    for point in chart_state.selection.points:
        curve_number = point.get("curve_number")
        if curve_number is None or fig.data[curve_number].name != "Nodos":
            continue
        customdata = point.get("customdata")
        if customdata is not None:
            selected_node = int(customdata[0] if isinstance(customdata, (list, tuple)) else customdata)
            break

    node_ids = set(numeric_columns_for_display(st.session_state.nodes_df, ["Nodo"]) ["Nodo"].dropna().astype(int))
    if selected_node in node_ids:
        st.session_state.selected_graph_node = selected_node
    elif st.session_state.get("selected_graph_node") not in node_ids:
        st.session_state.selected_graph_node = next(iter(node_ids), None)

    selected_node = st.session_state.get("selected_graph_node")
    if selected_node is None:
        st.caption("Agrega nodos para habilitar la edición desde el diagrama.")
        return

    if st.session_state.get("graph_editor_node") != selected_node:
        row = st.session_state.nodes_df.loc[st.session_state.nodes_df["Nodo"] == selected_node].iloc[0]
        st.session_state.graph_node_x = float(row["X"])
        st.session_state.graph_node_y = float(row["Y"])
        st.session_state.graph_editor_node = selected_node
    st.caption(f"Nodo seleccionado: {selected_node}. Edita sus coordenadas aquí o en la tabla.")
    coord_x, coord_y = st.columns(2)
    with coord_x:
        st.number_input(
            f"X ({st.session_state.length_unit})", key="graph_node_x", on_change=update_graph_node_coordinates,
        )
    with coord_y:
        st.number_input(
            f"Y ({st.session_state.length_unit})", key="graph_node_y", on_change=update_graph_node_coordinates,
        )


def solution_plot_base(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", title=title,
        xaxis_title=f"X ({st.session_state.length_unit})",
        yaxis_title=f"Y ({st.session_state.length_unit})",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=True, zeroline=True)
    fig.update_xaxes(showgrid=True, zeroline=True)
    return fig


def solver_node_coordinates(solver: Truss2D) -> dict[int, tuple[float, float]]:
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    return {node.id: (node.x / length_factor, node.y / length_factor) for node in solver.nodes}


def build_element_detail_figure(solver: Truss2D, element: Any, result: dict[str, Any]) -> go.Figure:
    coordinates = solver_node_coordinates(solver)
    x_i, y_i = coordinates[element.node_i]
    x_j, y_j = coordinates[element.node_j]
    span = max(abs(x_j - x_i), abs(y_j - y_i), result["Longitud"] / LENGTH_TO_METERS[st.session_state.length_unit], 1.0)
    arrow_size = span * 0.12
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_i, x_j], y=[y_i, y_j], mode="lines+markers+text",
        line=dict(color="#263746", width=5), marker=dict(size=12, color="#263746"),
        text=[f"N{element.node_i} (I)", f"N{element.node_j} (J)"],
        textposition="bottom center", textfont=dict(size=12, color="#263746"),
        hoverinfo="skip", showlegend=False,
    ))
    for node_id, (x, y) in ((element.node_i, (x_i, y_i)), (element.node_j, (x_j, y_j))):
        dof_x, dof_y = solver.node_dof_indices(node_id)
        for dx, dy, label in (
            (arrow_size, 0, f"g{dof_x + 1} · +X"),
            (0, arrow_size, f"g{dof_y + 1} · +Y"),
        ):
            fig.add_annotation(
                x=x + dx, y=y + dy, ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y",
                text=label, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor="#52705e", font=dict(size=9, color="#40564a"), bgcolor="white",
            )
    midpoint_x, midpoint_y = (x_i + x_j) / 2, (y_i + y_j) / 2
    c, s = result["c"], result["s"]
    local_arrow_length = span * 0.18
    fig.add_annotation(
        x=midpoint_x + c * local_arrow_length, y=midpoint_y + s * local_arrow_length,
        ax=midpoint_x, ay=midpoint_y, xref="x", yref="y", axref="x", ayref="y",
        text="+x_l", showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2,
        arrowcolor="#bd5b2a", font=dict(size=10, color="#9c4823"), bgcolor="white",
    )
    padding = span * 0.38
    fig.update_layout(
        title=f"Esquema de la barra {element.id} · φ = {np.degrees(result['phi']):.2f}°",
        template="plotly_white", height=250, showlegend=False,
        margin=dict(l=35, r=35, t=45, b=45),
        xaxis=dict(visible=False, range=[min(x_i, x_j) - padding, max(x_i, x_j) + padding]),
        yaxis=dict(visible=False, range=[min(y_i, y_j) - padding, max(y_i, y_j) + padding], scaleanchor="x", scaleratio=1),
    )
    return fig


def add_structure_members(fig: go.Figure, solver: Truss2D, coordinates: dict[int, tuple[float, float]], **style: Any) -> None:
    for element in solver.elements:
        xi, yi = coordinates[element.node_i]
        xj, yj = coordinates[element.node_j]
        fig.add_trace(go.Scatter(
            x=[xi, xj], y=[yi, yj], mode="lines",
            hoverinfo="skip", showlegend=False, **style,
        ))


def build_reactions_figure(solver: Truss2D) -> go.Figure:
    fig = solution_plot_base("Reacciones nodales")
    coordinates = solver_node_coordinates(solver)
    add_structure_members(fig, solver, coordinates, line=dict(color="#64748b", width=2))
    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    reaction_by_dof = dict(zip(solver.restrained_dofs, solver.reactions))
    max_reaction = max((abs(value) for value in solver.reactions), default=0.0)
    span = max(
        max(x for x, _ in coordinates.values()) - min(x for x, _ in coordinates.values()),
        max(y for _, y in coordinates.values()) - min(y for _, y in coordinates.values()),
        1.0,
    )
    arrow_scale = span * 0.14 / max(max_reaction, 1.0)
    for node_id, (x, y) in coordinates.items():
        dof_x, dof_y = solver.node_dof_indices(node_id)
        rx = float(reaction_by_dof.get(dof_x, 0.0))
        ry = float(reaction_by_dof.get(dof_y, 0.0))
        magnitude = float(np.hypot(rx, ry))
        if magnitude <= 1e-10:
            continue
        dx, dy = rx * arrow_scale, ry * arrow_scale
        fig.add_annotation(
            x=x + dx, y=y + dy, ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5,
            arrowcolor="#dc2626",
        )
        fig.add_annotation(
            x=x + dx * 1.2, y=y + dy * 1.2,
            text=f"R{node_id} = {format_number(magnitude / force_factor)} {st.session_state.force_unit}",
            showarrow=False, font=dict(color="#b91c1c", size=11), bgcolor="white",
        )
    fig.add_trace(go.Scatter(
        x=[point[0] for point in coordinates.values()],
        y=[point[1] for point in coordinates.values()], mode="markers+text",
        text=[str(node_id) for node_id in coordinates], textposition="top center",
        marker=dict(size=9, color="#334155"), name="Nodos",
    ))
    return fig


def build_axial_forces_figure(solver: Truss2D) -> go.Figure:
    fig = solution_plot_base("Esfuerzo axial por barra")
    coordinates = solver_node_coordinates(solver)
    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    results = solver.calculate_axial_forces()
    for result in results:
        element = next(item for item in solver.elements if item.id == result["Barra"])
        xi, yi = coordinates[element.node_i]
        xj, yj = coordinates[element.node_j]
        force = float(result["N"])
        color = "#2563eb" if force > 1e-9 else "#dc2626" if force < -1e-9 else "#64748b"
        fig.add_trace(go.Scatter(
            x=[xi, xj], y=[yi, yj], mode="lines",
            line=dict(color=color, width=5), name=f"Barra {element.id}: {result['estado']}",
            hovertemplate=f"Barra {element.id}<br>N={format_number(force / force_factor, signed=True)} {st.session_state.force_unit}<extra></extra>",
        ))
        fig.add_annotation(
            x=(xi + xj) / 2, y=(yi + yj) / 2,
            text=f"{format_number(force / force_factor, signed=True)} {st.session_state.force_unit}",
            showarrow=False, font=dict(color=color, size=11), bgcolor="white",
        )
    fig.add_trace(go.Scatter(
        x=[point[0] for point in coordinates.values()],
        y=[point[1] for point in coordinates.values()], mode="markers+text",
        text=[str(node_id) for node_id in coordinates], textposition="top center",
        marker=dict(size=8, color="#334155"), name="Nodos",
    ))
    return fig


def build_deformed_figure(solver: Truss2D, scale: int) -> go.Figure:
    fig = solution_plot_base(f"Deformada · escala {scale}×")
    original = solver_node_coordinates(solver)
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    deformed = {}
    for node in solver.nodes:
        dof_x, dof_y = solver.node_dof_indices(node.id)
        deformed[node.id] = (
            node.x / length_factor + solver.displacements[dof_x] / length_factor * scale,
            node.y / length_factor + solver.displacements[dof_y] / length_factor * scale,
        )
    add_structure_members(fig, solver, original, line=dict(color="#94a3b8", width=2, dash="dot"))
    add_structure_members(fig, solver, deformed, line=dict(color="#0f766e", width=3))
    for coordinates, name, color in ((original, "Original", "#94a3b8"), (deformed, "Deformada", "#0f766e")):
        fig.add_trace(go.Scatter(
            x=[point[0] for point in coordinates.values()],
            y=[point[1] for point in coordinates.values()], mode="markers+text",
            text=[str(node_id) for node_id in coordinates], textposition="top center",
            marker=dict(size=8, color=color), name=name,
        ))
    return fig


def styled_matrix(
    matrix: Any,
    row_labels: list[str],
    column_labels: list[str],
) -> pd.io.formats.style.Styler:
    frame = pd.DataFrame(matrix, index=row_labels, columns=column_labels)
    styler = frame.style.format(format_number, na_rep="—")
    return styler.set_table_styles([
        {"selector": "th", "props": [("font-weight", "bold"), ("border", "1px solid #333333")]},
        {"selector": "td", "props": [("border", "1px solid #333333"), ("text-align", "right")]},
    ])


def report_payload() -> dict[str, Any]:
    solver = st.session_state.solution
    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    reactions_by_dof = dict(zip(solver.restrained_dofs, solver.reactions))
    reactions = []
    for node in solver.nodes:
        dof_x, dof_y = solver.node_dof_indices(node.id)
        reactions.append({
            "Nodo": node.id,
            "Rx": float(reactions_by_dof.get(dof_x, 0.0)),
            "Ry": float(reactions_by_dof.get(dof_y, 0.0)),
        })
    return {
        "name": st.session_state.structure_name_input,
        "force_unit": st.session_state.force_unit,
        "length_unit": st.session_state.length_unit,
        "force_factor": force_factor,
        "length_factor": length_factor,
        "nodes": st.session_state.nodes_df.to_dict("records"),
        "elements": st.session_state.elements_df.to_dict("records"),
        "supports": st.session_state.supports_df.to_dict("records"),
        "loads": st.session_state.loads_df.to_dict("records"),
        "distributed_loads": st.session_state.distributed_loads_df.to_dict("records"),
        "K_global": solver.global_stiffness,
        "F_global": solver.global_load_vector(),
        "displacements": solver.displacements,
        "reactions": reactions,
        "axial_forces": solver.calculate_axial_forces(),
        "partition": solver.partition,
        "free_dofs": solver.free_dofs,
        "restrained_dofs": solver.restrained_dofs,
        "validation": st.session_state.validation,
    }


def render_solution() -> None:
    solver = st.session_state.solution
    if solver is None:
        st.info("Sin resultados. Revisa los datos y selecciona «Resolver estructura».")
        return

    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    st.caption(f"Desplazamientos en {st.session_state.length_unit}; fuerzas en {st.session_state.force_unit}; matrices y solución interna en SI.")
    axial_results = solver.calculate_axial_forces()
    st.markdown("### 5. Desplazamientos y reacciones")
    displacement_rows = []
    reaction_rows = []
    for node in solver.nodes:
        dof_x, dof_y = solver.node_dof_indices(node.id)
        displacement_rows.append({
            "Nodo": node.id,
            f"Dx ({st.session_state.length_unit})": solver.displacements[dof_x] / length_factor,
            f"Dy ({st.session_state.length_unit})": solver.displacements[dof_y] / length_factor,
        })
        reaction_rows.append({
            "Nodo": node.id,
            f"Rx ({st.session_state.force_unit})": next((value for dof, value in zip(solver.restrained_dofs, solver.reactions) if dof == dof_x), 0.0) / force_factor,
            f"Ry ({st.session_state.force_unit})": next((value for dof, value in zip(solver.restrained_dofs, solver.reactions) if dof == dof_y), 0.0) / force_factor,
        })
    st.write("Desplazamientos nodales")
    st.dataframe(format_dataframe(pd.DataFrame(displacement_rows)), width="stretch")
    extreme_displacements = []
    for node, row in zip(solver.nodes, displacement_rows):
        for axis, column in (("x", f"Dx ({st.session_state.length_unit})"), ("y", f"Dy ({st.session_state.length_unit})")):
            value = float(row[column])
            if 0 < abs(value) < 1e-3 or abs(value) >= 1e5:
                extreme_displacements.append((node.id, axis, value))
    for node_id, axis, value in extreme_displacements:
        st.latex(
            rf"D_{{N{node_id},{axis}}} = {format_number(value, latex=True)}\;"
            rf"\mathrm{{{st.session_state.length_unit}}}"
        )
    st.write("Reacciones por apoyo")
    st.dataframe(format_dataframe(pd.DataFrame(reaction_rows)), width="stretch")

    total_rx = sum(row[f"Rx ({st.session_state.force_unit})"] for row in reaction_rows)
    total_ry = sum(row[f"Ry ({st.session_state.force_unit})"] for row in reaction_rows)
    external_loads = solver.global_load_vector()
    total_fx = float(np.sum(external_loads[0::2])) / force_factor
    total_fy = float(np.sum(external_loads[1::2])) / force_factor
    residual = float(np.hypot(total_rx + total_fx, total_ry + total_fy))
    st.info(
        f"Reacción total: ΣRx = {format_number(total_rx)} {st.session_state.force_unit}; "
        f"ΣRy = {format_number(total_ry)} {st.session_state.force_unit}; "
        f"resultante = {format_number(np.hypot(total_rx, total_ry))} {st.session_state.force_unit}."
    )
    if residual <= max(1e-6, 1e-6 * float(np.hypot(total_fx, total_fy))):
        st.success("Equilibrio global verificado: la suma de reacciones y cargas es prácticamente cero.")
    else:
        st.warning(f"Desequilibrio global residual: {format_number(residual)} {st.session_state.force_unit}.")

    st.markdown("### 6. Fuerzas axiales internas por barra")
    element_by_id = {int(row["Barra"]): row for row in st.session_state.elements_df.to_dict("records") if pd.notna(row.get("Barra"))}
    axial_table = []
    for result in axial_results:
        element = element_by_id[result["Barra"]]
        axial_table.append({
            "Barra": result["Barra"],
            "Nodos": f"N{element['Nodo_i']}–N{element['Nodo_j']}",
            f"N ({st.session_state.force_unit})": result["N"] / force_factor,
            "Clasificación": result["estado"],
        })
    st.dataframe(format_dataframe(pd.DataFrame(axial_table), [f"N ({st.session_state.force_unit})"]), width="stretch")

    reaction_column, axial_column = st.columns(2)
    with reaction_column:
        st.plotly_chart(build_reactions_figure(solver), width="stretch")
    with axial_column:
        st.plotly_chart(build_axial_forces_figure(solver), width="stretch")
    deformation_scale = st.slider(
        "Factor de escala de deformación", min_value=1, max_value=1000, value=100,
        key="deformation_scale",
    )
    st.plotly_chart(build_deformed_figure(solver, deformation_scale), width="stretch")


def render_detailed_development() -> None:
    solver = st.session_state.solution
    if solver is None:
        st.info("Resuelve la estructura para generar el desarrollo paso a paso.")
        return

    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    dof_labels = [f"g{index + 1}" for index in range(len(solver.displacements))]
    free_dofs = set(solver.free_dofs.tolist())
    axial_results = solver.calculate_axial_forces()
    axial_results_by_id = {result["Barra"]: result for result in axial_results}

    st.subheader("Desarrollo Detallado Paso a Paso")
    st.caption("Geometría y propiedades en las unidades de entrada; matrices y vector global en SI.")
    st.markdown("### Paso 1: Geometría y grados de libertad")
    dof_rows = []
    for node in solver.nodes:
        dof_x, dof_y = solver.node_dof_indices(node.id)
        dof_rows.append({
            "Nodo": node.id,
            f"X ({st.session_state.length_unit})": node.x / length_factor,
            f"Y ({st.session_state.length_unit})": node.y / length_factor,
            "GDL X": dof_labels[dof_x],
            "Estado X": "Libre" if dof_x in free_dofs else "Restringido",
            "GDL Y": dof_labels[dof_y],
            "Estado Y": "Libre" if dof_y in free_dofs else "Restringido",
        })
    st.dataframe(format_dataframe(pd.DataFrame(dof_rows)), width="stretch")

    st.markdown("### Paso 2: Análisis barra por barra")
    ui_elements = st.session_state.elements_df.set_index("Barra", drop=False)
    distributed_loads = st.session_state.distributed_loads_df
    for element in solver.elements:
        c, s = solver.element_direction_cosines(element)
        length = solver.element_length(element)
        bar_result = axial_results_by_id[element.id]
        phi = bar_result["phi"]
        input_row = ui_elements.loc[element.id]
        if isinstance(input_row, pd.DataFrame):
            input_row = input_row.iloc[0]
        with st.expander(f"Barra {element.id} · N{element.node_i} → N{element.node_j} · φ = {np.degrees(phi):.2f}°", expanded=False):
            member_length = length / length_factor
            member_area = parse_numeric_expression(input_row["A"])
            member_modulus = parse_numeric_expression(input_row["E"])
            axial_rigidity = element.E * element.A / length / (force_factor / length_factor)
            property_columns = st.columns(5)
            for column, label, value in zip(
                property_columns,
                ("Longitud", "Ángulo φ", "Área A", "Módulo E", "Rigidez AE/L"),
                (
                    f"{format_number(member_length)} {st.session_state.length_unit}",
                    f"{format_number(np.degrees(phi))}°",
                    f"{format_number(member_area)} {st.session_state.length_unit}²",
                    f"{format_number(member_modulus)} {st.session_state.force_unit}/{st.session_state.length_unit}²",
                    f"{format_number(axial_rigidity)} {st.session_state.force_unit}/{st.session_state.length_unit}",
                ),
            ):
                with column:
                    st.metric(label, value)
            if abs(member_modulus) < 1e-3 or abs(member_modulus) >= 1e5:
                st.latex(
                    rf"E = {format_number(member_modulus, latex=True)}\;"
                    rf"\mathrm{{{st.session_state.force_unit}/{st.session_state.length_unit}^2}}"
                )
            st.caption(f"cos φ = {format_number(c)} · sen φ = {format_number(s)}")
            st.plotly_chart(
                build_element_detail_figure(solver, element, bar_result),
                width="stretch", config={"displayModeBar": False},
                key=f"member_geometry_{element.id}",
            )

            local_dofs = [f"u{element.node_i}'", f"v{element.node_i}'", f"u{element.node_j}'", f"v{element.node_j}'"]
            global_dofs = [f"g{index + 1}" for index in (
                *solver.node_dof_indices(element.node_i), *solver.node_dof_indices(element.node_j),
            )]
            st.caption("Secuencia matricial: [T]ᵀ × [k_l] × [T] = [K_g]")
            st.latex(r"[K_g]=[T]^T[k_l][T]")
            matrix_columns = st.columns(4)
            matrix_items = (
                ("[T]ᵀ", bar_result["Tg"].T, global_dofs, local_dofs),
                ("[k_l] (N/m)", bar_result["KL"], local_dofs, local_dofs),
                ("[T]", bar_result["Tg"], local_dofs, global_dofs),
                ("[K_g] (N/m)", bar_result["Kg"], global_dofs, global_dofs),
            )
            for column, (label, matrix, row_labels, column_labels) in zip(matrix_columns, matrix_items):
                with column:
                    st.caption(label)
                    st.dataframe(styled_matrix(matrix, row_labels, column_labels), width="stretch")

            member_load_rows = distributed_loads[distributed_loads["Barra"] == element.id]
            member_loads = [load for load in solver.distributed_loads if load.element_id == element.id]
            for ((_, load_row), member_load) in zip(member_load_rows.iterrows(), member_loads):
                st.caption(
                    f"Carga axial distribuida: q_i={format_number(load_row['q_i'])}, "
                    f"q_j={format_number(load_row['q_j'])} {st.session_state.force_unit}/{st.session_state.length_unit}"
                )
                equivalent_local = solver.element_distributed_load_local(member_load) / force_factor
                equivalent_global = solver.element_distributed_load_global(member_load) / force_factor
                st.write("Cargas nodales equivalentes locales")
                st.dataframe(format_dataframe(pd.DataFrame({"GDL local": local_dofs, f"Carga ({st.session_state.force_unit})": equivalent_local})), width="stretch")
                st.caption("Para q uniforme, la carga equivalente axial por extremo es q·L/2.")
                st.write("Cargas nodales equivalentes globales")
                st.dataframe(format_dataframe(pd.DataFrame({"GDL global": global_dofs, f"Carga ({st.session_state.force_unit})": equivalent_global})), width="stretch")

    st.markdown("### Paso 3: Ensamble y partición de la matriz global")
    st.caption("Matriz ensamblada en el orden global de grados de libertad g1…gN; unidades SI.")
    st.dataframe(styled_matrix(solver.global_stiffness, dof_labels, dof_labels), width="stretch")
    st.write("Bloques según GDL libres (L) y restringidos (R)")
    block_columns = st.columns(2)
    for column, key, row_ids, col_ids in (
        (block_columns[0], "K_LL", solver.free_dofs, solver.free_dofs),
        (block_columns[1], "K_LR", solver.free_dofs, solver.restrained_dofs),
        (block_columns[0], "K_RL", solver.restrained_dofs, solver.free_dofs),
        (block_columns[1], "K_RR", solver.restrained_dofs, solver.restrained_dofs),
    ):
        with column:
            st.write(f"Submatriz [{key}]")
            row_labels = [dof_labels[index] for index in row_ids]
            column_labels = [dof_labels[index] for index in col_ids]
            st.dataframe(styled_matrix(solver.partition[key], row_labels, column_labels), width="stretch")
    st.write("Inversa de la submatriz libre-libre [K_LL]⁻¹ [m/N]")
    inverse_k_ll = np.linalg.inv(solver.partition["K_LL"])
    free_labels = [dof_labels[index] for index in solver.free_dofs]
    st.dataframe(styled_matrix(inverse_k_ll, free_labels, free_labels), width="stretch")

    st.markdown("### Paso 4: Vector P y solución")
    load_vector = solver.global_load_vector()
    st.write("P_global: cargas nodales más fuerzas equivalentes distribuidas [N]")
    st.dataframe(format_dataframe(pd.DataFrame({"GDL": dof_labels, "P [N]": load_vector})), width="stretch")
    st.latex(r"K_{LL}D_L=P_L-K_{LR}D_R")
    st.latex(r"D_L=K_{LL}^{-1}(P_L-K_{LR}D_R);\quad D_R=0\Rightarrow D_L=K_{LL}^{-1}P_L")
    free_solution = pd.DataFrame({
        "GDL libre": [dof_labels[index] for index in solver.free_dofs],
        "P_L (N)": solver.partition["F_L"],
        "D_L (m)": solver.displacements[solver.free_dofs],
    })
    st.dataframe(format_dataframe(free_solution), width="stretch")
    restrained_solution = pd.DataFrame({
        "GDL restringido": [dof_labels[index] for index in solver.restrained_dofs],
        "P_R (N)": solver.partition["F_R"],
        "D_R (m)": solver.displacements[solver.restrained_dofs],
    })
    st.dataframe(format_dataframe(restrained_solution), width="stretch")
    st.write("Reacciones en GDL restringidos")
    st.latex(r"F_{RR}=K_{RL}D_L+K_{RR}D_R-P_R;\quad D_R=0,\ P_R=0\Rightarrow F_{RR}=K_{RL}D_L")
    st.dataframe(format_dataframe(pd.DataFrame({
        "GDL": [dof_labels[index] for index in solver.restrained_dofs],
        f"Reacción ({st.session_state.force_unit})": solver.reactions / force_factor,
    })), width="stretch")

    st.markdown("### Paso 5: Cálculo de fuerzas internas por barra")
    st.dataframe(format_dataframe(pd.DataFrame([{
        "Barra": result["Barra"],
        f"N_i ({st.session_state.force_unit})": result["N_i"] / force_factor,
        f"N_j ({st.session_state.force_unit})": result["N_j"] / force_factor,
        "Clasificación": result["estado"],
    } for result in axial_results]), [
        f"N_i ({st.session_state.force_unit})", f"N_j ({st.session_state.force_unit})",
    ]), width="stretch")
    for element in solver.elements:
        bar_result = axial_results_by_id[element.id]
        local_dofs = [f"u{element.node_i}'", f"v{element.node_i}'", f"u{element.node_j}'", f"v{element.node_j}'"]
        global_indices = [*solver.node_dof_indices(element.node_i), *solver.node_dof_indices(element.node_j)]
        global_dofs = [dof_labels[index] for index in global_indices]
        with st.expander(
            f"Barra {element.id} · N{element.node_i} → N{element.node_j} · {bar_result['estado']}",
            expanded=element.id == solver.elements[0].id,
        ):
            st.plotly_chart(
                build_element_detail_figure(solver, element, bar_result),
                width="stretch", config={"displayModeBar": False},
                key=f"member_forces_{element.id}",
            )
            st.latex(r"\{d_l\}=[T]\{d_g\};\quad \{f_L\}=[k_l]\{d_l\}")
            st.caption("Desplazamientos globales del elemento {d_g} (m)")
            st.dataframe(format_dataframe(pd.DataFrame({
                "GDL global": global_dofs,
                "Desplazamiento global d_g (m)": bar_result["De"],
            })), width="stretch")
            st.caption("Matriz de transformación [T]")
            st.dataframe(styled_matrix(bar_result["Tg"], local_dofs, global_dofs), width="stretch")
            st.caption("Desplazamientos locales {d_l} = [T]{d_g} (m)")
            st.dataframe(format_dataframe(pd.DataFrame({
                "GDL local": local_dofs,
                "Desplazamiento local d_l (m)": bar_result["d_local"],
            })), width="stretch")
            st.caption("Matriz de rigidez local [k_l] (N/m)")
            st.dataframe(styled_matrix(bar_result["KL"], local_dofs, local_dofs), width="stretch")
            st.caption("Vector de fuerzas elásticas {f_L} = [k_l]{d_l} (N)")
            st.dataframe(format_dataframe(pd.DataFrame({
                "GDL local": local_dofs,
                "Fuerza elástica f_L (N)": bar_result["f_L"],
            }), ["Fuerza elástica f_L (N)"]), width="stretch")
            force_table = pd.DataFrame({
                "GDL local": local_dofs,
                "Desplazamiento local d_l (m)": bar_result["d_local"],
                "Fuerza elástica f_L (N)": bar_result["f_L"],
                "Fuerza interna (N)": bar_result["fuerzas_internas_locales"],
            })
            st.dataframe(format_dataframe(force_table, ["Fuerza elástica f_L (N)", "Fuerza interna (N)"]), width="stretch")
            axial_color = "#1d4ed8" if bar_result["N"] >= 0 else "#b91c1c"
            st.markdown(
                f"Fuerza axial **N = {format_number(bar_result['N'] / force_factor, signed=True)} {st.session_state.force_unit}** "
                f"· <span style='color:{axial_color};font-weight:700'>{bar_result['estado']}</span>",
                unsafe_allow_html=True,
            )


def build_structure_figure() -> go.Figure:
    fig = go.Figure()
    length_unit = st.session_state.length_unit
    force_unit = st.session_state.force_unit
    fig.update_layout(
        template="plotly_white",
        title=st.session_state.structure_name_input,
        xaxis_title=f"X ({length_unit})",
        yaxis_title=f"Y ({length_unit})",
        showlegend=True,
        clickmode="event+select",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    nodes_df = st.session_state.get("nodes_df", pd.DataFrame(columns=["Nodo", "X", "Y"]))
    elements_df = st.session_state.get("elements_df", pd.DataFrame(columns=["Barra", "Nodo_i", "Nodo_j", "E", "A"]))
    supports_df = st.session_state.get("supports_df", pd.DataFrame(columns=["Nodo", "Ux", "Uy"]))
    loads_df = st.session_state.get("loads_df", pd.DataFrame(columns=["Nodo", "Fx", "Fy"]))
    distributed_loads_df = st.session_state.get("distributed_loads_df", pd.DataFrame(columns=["Barra", "Tipo", "q_i", "q_j"]))

    if nodes_df.empty or not {"Nodo", "X", "Y"}.issubset(nodes_df.columns):
        fig.update_xaxes(range=[-1, 1], showgrid=True)
        fig.update_yaxes(range=[-1, 1], showgrid=True)
        return fig

    nodes_df = nodes_df.copy()
    nodes_df = numeric_columns_for_display(nodes_df, ["Nodo", "X", "Y"])
    nodes_df = nodes_df.dropna(subset=["Nodo", "X", "Y"]).copy()

    if nodes_df.empty:
        fig.update_xaxes(range=[-1, 1], showgrid=True)
        fig.update_yaxes(range=[-1, 1], showgrid=True)
        return fig

    node_map = {int(row["Nodo"]): (float(row["X"]), float(row["Y"])) for _, row in nodes_df.iterrows()}
    x_values = [coords[0] for coords in node_map.values()]
    y_values = [coords[1] for coords in node_map.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    span_x = x_max - x_min
    span_y = y_max - y_min
    pad_x = span_x * 0.3 if span_x else 1.0
    pad_y = span_y * 0.3 if span_y else 1.0
    fig.update_xaxes(range=[min(x_min - pad_x, x_min - 1), max(x_max + pad_x, x_max + 1)], showgrid=True)
    fig.update_yaxes(range=[min(y_min - pad_y, y_min - 1), max(y_max + pad_y, y_max + 1)], showgrid=True, scaleanchor="x", scaleratio=1)
    arrow_x = max(span_x * 0.1, 0.2)
    arrow_y = max(span_y * 0.1, 0.2)

    if not elements_df.empty and {"Nodo_i", "Nodo_j"}.issubset(elements_df.columns):
        for _, row in elements_df.iterrows():
            try:
                i = int(row["Nodo_i"])
                j = int(row["Nodo_j"])
                if i in node_map and j in node_map:
                    xi, yi = node_map[i]
                    xj, yj = node_map[j]
                    fig.add_trace(go.Scatter(x=[xi, xj], y=[yi, yj], mode="lines", line=dict(color="#2563eb", width=2), name=f"Barra {row.get('Barra', '')}"))
                    fig.add_annotation(
                        x=(xi + xj) / 2, y=(yi + yj) / 2,
                        text=f"B{format_number(parse_numeric_for_display(row.get('Barra', '')))}",
                        showarrow=False, font=dict(color="#1e3a5f", size=11), bgcolor="white",
                    )
            except Exception:
                continue

    if not distributed_loads_df.empty and {"Barra", "q_i", "q_j"}.issubset(distributed_loads_df.columns):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#dc2626", width=2), name="Carga distribuida axial",
        ))
        element_lookup = {int(row["Barra"]): row for _, row in elements_df.iterrows() if pd.notna(row.get("Barra"))}
        for _, load_row in distributed_loads_df.iterrows():
            try:
                element_row = element_lookup[int(load_row["Barra"])]
                if str(load_row.get("Tipo", "Axial")).strip().lower() != "axial":
                    continue
                node_i, node_j = int(element_row["Nodo_i"]), int(element_row["Nodo_j"])
                if node_i not in node_map or node_j not in node_map:
                    continue
                xi, yi = node_map[node_i]
                xj, yj = node_map[node_j]
                dx, dy = xj - xi, yj - yi
                length = float(np.hypot(dx, dy))
                if length == 0:
                    continue
                ux, uy = dx / length, dy / length
                nx, ny = -uy, ux
                q_i = float(load_row["q_i"])
                q_j = float(load_row["q_j"])
                q_peak = max(abs(q_i), abs(q_j))
                arrow_length = min(length * 0.12, max(length * 0.07, 0.25))
                offset = max(length * 0.055, 0.16)
                for fraction in (0.18, 0.34, 0.50, 0.66, 0.82):
                    q_value = q_i + (q_j - q_i) * fraction
                    if abs(q_value) < 1e-12:
                        continue
                    sign = 1.0 if q_value > 0 else -1.0
                    x_base = xi + dx * fraction + nx * offset * sign
                    y_base = yi + dy * fraction + ny * offset * sign
                    fig.add_annotation(
                        x=x_base + ux * arrow_length * sign,
                        y=y_base + uy * arrow_length * sign,
                        ax=x_base, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                        arrowcolor="#dc2626",
                    )
                fig.add_annotation(
                    x=xi + dx * 0.5 + nx * offset * (1 if q_i + q_j >= 0 else -1) * 2,
                    y=yi + dy * 0.5 + ny * offset * (1 if q_i + q_j >= 0 else -1) * 2,
                    text=f"q={format_number(q_peak)} {force_unit}/{length_unit}",
                    showarrow=False, font=dict(color="#b91c1c", size=10), bgcolor="white",
                )
            except Exception:
                continue

    fig.add_trace(go.Scatter(
        x=[coords[0] for coords in node_map.values()],
        y=[coords[1] for coords in node_map.values()],
        mode="markers+text",
        text=[str(int(row["Nodo"])) for _, row in nodes_df.iterrows()],
        textposition="top center",
        marker=dict(size=10, color="#ef4444"),
        name="Nodos",
        customdata=[int(row["Nodo"]) for _, row in nodes_df.iterrows()],
        hovertemplate="Nodo %{customdata}<br>X %{x:g}<br>Y %{y:g}<extra></extra>",
    ))

    for row in nodes_df.itertuples(index=False):
        node_id = int(row.Nodo)
        x0, y0 = node_map[node_id]
        dof_x = 2 * (node_id - 1) + 1
        dof_y = dof_x + 1
        for x_head, y_head, label, color in (
            (x0 + arrow_x, y0, f"g{dof_x} · +X", "#2563eb"),
            (x0, y0 + arrow_y, f"g{dof_y} · +Y", "#059669"),
        ):
            fig.add_annotation(
                x=x_head, y=y_head, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                text=label, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor=color, font=dict(color=color, size=11), bgcolor="white", borderpad=2,
            )

    support_legend = set()
    if not supports_df.empty and {"Nodo", "Ux", "Uy"}.issubset(supports_df.columns):
        for _, row in supports_df.iterrows():
            try:
                node_id = int(row["Nodo"])
                if node_id not in node_map:
                    continue
                x0, y0 = node_map[node_id]
                ux = int(row.get("Ux", 0))
                uy = int(row.get("Uy", 0))
                support_type = "Apoyo fijo" if ux == 0 and uy == 0 else "Apoyo móvil/liso"
                support_symbol = "triangle-down" if support_type == "Apoyo fijo" else "circle-open"
                support_color = "#111827" if support_type == "Apoyo fijo" else "#0f766e"
                fig.add_trace(go.Scatter(
                    x=[x0], y=[y0 - arrow_y * 0.3], mode="markers",
                    marker=dict(symbol=support_symbol, size=13, color=support_color),
                    name=support_type, showlegend=support_type not in support_legend,
                    hovertemplate=f"{support_type}<br>Nodo {node_id}<br>Ux={ux}, Uy={uy}<extra></extra>",
                ))
                support_legend.add(support_type)
            except Exception:
                continue

    if not loads_df.empty and {"Nodo", "Fx", "Fy"}.issubset(loads_df.columns):
        for _, row in loads_df.iterrows():
            try:
                node_id = int(row["Nodo"])
                if node_id not in node_map:
                    continue
                x0, y0 = node_map[node_id]
                fx = float(row.get("Fx", 0.0))
                fy = float(row.get("Fy", 0.0))
                if abs(fx) < 1e-9 and abs(fy) < 1e-9:
                    continue
                arrow_scale_x = arrow_x * 1.2
                arrow_scale_y = arrow_y * 1.2
                scale = max(abs(fx), abs(fy), 1.0)
                dx = fx * arrow_scale_x / scale
                dy = fy * arrow_scale_y / scale
                magnitude = float(np.hypot(fx, fy))
                fig.add_annotation(
                    x=x0 + dx, y=y0 + dy, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                    text=f"Fx={fx:g}, Fy={fy:g}; |P|={magnitude:g} {force_unit}", showarrow=True, arrowhead=3,
                    arrowsize=1, arrowwidth=2, arrowcolor="#ea580c", font=dict(color="#c2410c", size=10),
                )
            except Exception:
                continue

    return fig


initialize_state()

st.title("Análisis Estructural II | Método de Rigidez")
header_name, header_exercise, header_actions = st.columns([2, 2, 2])
with header_name:
    st.text_input("Nombre de la estructura", key="structure_name_input")
with header_exercise:
    st.selectbox(
        "Estructura", ["Ejercicio N.º 1", "Estructura personalizada"],
        key="exercise_choice", on_change=load_selected_exercise,
    )
with header_actions:
    action_solve, action_reset = st.columns(2)
    with action_solve:
        st.button("Resolver estructura", on_click=solve_structure, type="primary", width="stretch")
    with action_reset:
        st.button("Restablecer", on_click=reset_structure, width="stretch")

with st.sidebar:
    st.subheader("Unidades de entrada")
    st.selectbox("Fuerza", list(FORCE_TO_NEWTONS), key="force_unit", on_change=change_units)
    st.selectbox("Longitud / coordenadas", list(LENGTH_TO_METERS), key="length_unit", on_change=change_units)
    st.caption(
        f"Coordenadas: {st.session_state.length_unit}; cargas: {st.session_state.force_unit}; "
        f"E: {st.session_state.force_unit}/{st.session_state.length_unit}²; "
        f"A: {st.session_state.length_unit}²; q: {st.session_state.force_unit}/{st.session_state.length_unit}. El solver convierte a SI."
    )
    st.divider()
    if st.button("Descargar Excel", width="stretch"):
        if st.session_state.solution is not None:
            results = report_payload()
            path = export_analysis_excel(results, "Memoria_Analisis_Estructural.xlsx")
            with open(path, "rb") as file:
                st.download_button("Descargar archivo Excel", file.read(), file_name="Memoria_Analisis_Estructural.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Primero resuelve la estructura para generar el archivo Excel.")

    if st.button("Preparar memoria PDF", width="stretch"):
        if st.session_state.solution is not None:
            results = report_payload()
            path = generate_pdf_report(results, "Memoria_Analisis_Estructural.pdf")
            with open(path, "rb") as file:
                st.download_button("Descargar PDF", file.read(), file_name="Memoria_Analisis_Estructural.pdf", mime="application/pdf")
        else:
            st.warning("Primero resuelve la estructura para generar el PDF.")

st.radio("Vista", ["Integrada", "Por secciones"], horizontal=True, key="view_mode", label_visibility="collapsed")
st.caption(f"{st.session_state.structure_name_input} · {len(st.session_state.nodes_df)} nodos · {len(st.session_state.elements_df)} barras")
if st.session_state.solution_error:
    st.error(f"No se pudo resolver la estructura: {st.session_state.solution_error}")

if st.session_state.view_mode == "Integrada":
    data_column, diagram_column = st.columns([1.2, 1], gap="large")
    with data_column:
        render_data_editors()
    with diagram_column:
        st.subheader("Canvas de edición")
        render_structure_canvas()
        with st.expander("Diagrama de resultados", expanded=False):
            render_structure_plot()
    with st.expander("Resultados del análisis", expanded=st.session_state.solution is not None):
        render_solution()
    with st.expander("Desarrollo detallado paso a paso"):
        render_detailed_development()
else:
    tabs = st.tabs(["Nodos", "Barras", "Apoyos", "Cargas", "Solución", "Desarrollo detallado", "Diagrama", "Validación"])
    text_input_columns = {
        "nodes_df": {
            "X": st.column_config.TextColumn("X", help=EXPRESSION_HELP),
            "Y": st.column_config.TextColumn("Y", help=EXPRESSION_HELP),
        },
        "elements_df": {
            "E": st.column_config.TextColumn("E", help=EXPRESSION_HELP),
            "A": st.column_config.TextColumn("A", help=EXPRESSION_HELP),
        },
        "supports_df": {
            "Ux": st.column_config.TextColumn("Ux", help=EXPRESSION_HELP),
            "Uy": st.column_config.TextColumn("Uy", help=EXPRESSION_HELP),
        },
        "loads_df": {
            "Fx": st.column_config.TextColumn("Fx", help=EXPRESSION_HELP),
            "Fy": st.column_config.TextColumn("Fy", help=EXPRESSION_HELP),
        },
        "distributed_loads_df": {
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Axial"], required=True),
            "q_i": st.column_config.TextColumn("q_i", help=EXPRESSION_HELP),
            "q_j": st.column_config.TextColumn("q_j", help=EXPRESSION_HELP),
        },
    }
    for tab, title, state_key, editor_key in zip(
        tabs[:5],
        ("Tabla de nodos", "Tabla de barras", "Tabla de apoyos", "Tabla de cargas nodales", "Tabla de cargas distribuidas"),
        ("nodes_df", "elements_df", "supports_df", "loads_df", "distributed_loads_df"),
        EDITOR_KEYS,
    ):
        with tab:
            st.subheader(title)
            if state_key == "distributed_loads_df":
                st.caption(f"q en {st.session_state.force_unit}/{st.session_state.length_unit}; solo se admite Tipo = Axial.")
            st.session_state[state_key] = st.data_editor(
                expression_editor_frame(st.session_state[state_key], list(text_input_columns[state_key])), key=editor_key, num_rows="dynamic", width="stretch",
                column_config=text_input_columns[state_key],
                on_change=sync_editor_dataframe, args=(editor_key, state_key),
            )
    with tabs[4]:
        render_solution()
    with tabs[5]:
        render_detailed_development()
    with tabs[6]:
        st.subheader("Canvas de edición")
        render_structure_canvas()
        st.subheader("Diagrama de resultados")
        render_structure_plot()
    with tabs[7]:
        validation = st.session_state.validation
        st.subheader("Validación del ejercicio")
        message = validation.get("message", "Sin mensaje de validación.")
        if validation.get("status") == "PASS":
            st.success(message)
        else:
            st.info(message)
        missing_data = validation.get("missing_data", [])
        if missing_data:
            st.markdown("**Datos pendientes para cotejo:**")
            for item in missing_data:
                st.markdown(f"- {item}")
