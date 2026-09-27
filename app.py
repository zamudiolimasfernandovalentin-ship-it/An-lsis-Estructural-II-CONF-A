from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from examples import ejercicio_1
from reports import export_analysis_excel, generate_pdf_report
from structural_solver import Truss2D, parse_distributed_loads_from_df, parse_elements_from_df, parse_loads_from_df, parse_nodes_from_df, parse_supports_from_df, validate_exercise_1


st.set_page_config(page_title="Análisis Estructural II – Método de Rigidez", layout="wide")

FORCE_TO_NEWTONS = {"tonf": 9806.65, "kgf": 9.80665, "kN": 1000.0, "N": 1.0}
LENGTH_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
EDITOR_KEYS = ("nodes_editor", "elements_editor", "supports_editor", "loads_editor", "distributed_loads_editor")


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

    for column in ("Fx", "Fy"):
        if column in st.session_state.loads_df:
            st.session_state.loads_df[column] = pd.to_numeric(st.session_state.loads_df[column], errors="coerce") * old_force / new_force
    for column in ("X", "Y"):
        if column in st.session_state.nodes_df:
            st.session_state.nodes_df[column] = pd.to_numeric(st.session_state.nodes_df[column], errors="coerce") * old_length / new_length
    if "A" in st.session_state.elements_df:
        st.session_state.elements_df["A"] = pd.to_numeric(st.session_state.elements_df["A"], errors="coerce") * old_length**2 / new_length**2
    if "E" in st.session_state.elements_df:
        st.session_state.elements_df["E"] = (
            pd.to_numeric(st.session_state.elements_df["E"], errors="coerce")
            * (old_force / old_length**2)
            / (new_force / new_length**2)
        )
    for column in ("q_i", "q_j"):
        if column in st.session_state.distributed_loads_df:
            st.session_state.distributed_loads_df[column] = (
                pd.to_numeric(st.session_state.distributed_loads_df[column], errors="coerce")
                * (old_force / old_length) / (new_force / new_length)
            )

    st.session_state.current_force_unit = st.session_state.force_unit
    st.session_state.current_length_unit = st.session_state.length_unit
    st.session_state.solution = None
    st.session_state.solution_error = None
    clear_editor_widget_state()


def current_inputs_in_si() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    nodes = st.session_state.nodes_df.copy()
    elements = st.session_state.elements_df.copy()
    supports = st.session_state.supports_df.copy()
    loads = st.session_state.loads_df.copy()
    distributed_loads = st.session_state.distributed_loads_df.copy()

    for column in ("X", "Y"):
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce") * length_factor
    elements["E"] = pd.to_numeric(elements["E"], errors="coerce") * force_factor / length_factor**2
    elements["A"] = pd.to_numeric(elements["A"], errors="coerce") * length_factor**2
    for column in ("Fx", "Fy"):
        loads[column] = pd.to_numeric(loads[column], errors="coerce") * force_factor
    for column in ("q_i", "q_j"):
        distributed_loads[column] = pd.to_numeric(distributed_loads[column], errors="coerce") * force_factor / length_factor
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


def render_data_editors() -> None:
    st.subheader("Nodos")
    st.session_state.nodes_df = st.data_editor(
        st.session_state.nodes_df, key="nodes_editor", num_rows="dynamic", width="stretch",
        on_change=mark_structure_changed,
    )
    st.subheader("Barras y propiedades")
    st.session_state.elements_df = st.data_editor(
        st.session_state.elements_df, key="elements_editor", num_rows="dynamic", width="stretch",
        on_change=mark_structure_changed,
    )
    st.subheader("Apoyos")
    st.session_state.supports_df = st.data_editor(
        st.session_state.supports_df, key="supports_editor", num_rows="dynamic", width="stretch",
        on_change=mark_structure_changed,
    )
    st.subheader("Cargas")
    st.session_state.loads_df = st.data_editor(
        st.session_state.loads_df, key="loads_editor", num_rows="dynamic", width="stretch",
        on_change=mark_structure_changed,
    )
    st.subheader("Cargas distribuidas axiales")
    st.caption(
        f"Intensidad q en {st.session_state.force_unit}/{st.session_state.length_unit}. "
        "El modelo de armadura no admite cargas transversales ni momentos de extremo."
    )
    st.session_state.distributed_loads_df = st.data_editor(
        st.session_state.distributed_loads_df,
        key="distributed_loads_editor", num_rows="dynamic", width="stretch",
        column_config={
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Axial"], required=True),
        },
        on_change=mark_structure_changed,
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

    node_ids = set(pd.to_numeric(st.session_state.nodes_df["Nodo"], errors="coerce").dropna().astype(int))
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


def render_solution() -> None:
    solver = st.session_state.solution
    if solver is None:
        st.info("Sin resultados. Revisa los datos y selecciona «Resolver estructura».")
        return

    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    st.caption(f"Desplazamientos en {st.session_state.length_unit}; fuerzas en {st.session_state.force_unit}; matrices y solución interna en SI.")
    st.write("Desplazamientos")
    st.dataframe(pd.DataFrame({
        "GDL": np.arange(1, len(solver.displacements) + 1),
        f"Desplazamiento ({st.session_state.length_unit})": solver.displacements / length_factor,
    }), width="stretch")
    st.write("Reacciones")
    st.dataframe(pd.DataFrame({
        "GDL restringido": solver.restrained_dofs + 1,
        f"Reacción ({st.session_state.force_unit})": solver.reactions / force_factor,
    }), width="stretch")
    st.write("Fuerzas axiales")
    axial_results = solver.calculate_axial_forces()
    st.dataframe(pd.DataFrame([{
        "Barra": result["Barra"],
        f"Longitud ({st.session_state.length_unit})": result["Longitud"] / length_factor,
        f"Fuerza axial ({st.session_state.force_unit})": result["N"] / force_factor,
        f"N_i ({st.session_state.force_unit})": result["N_i"] / force_factor,
        f"N_j ({st.session_state.force_unit})": result["N_j"] / force_factor,
        "Estado": result["estado"],
    } for result in axial_results]), width="stretch")


def render_detailed_development() -> None:
    solver = st.session_state.solution
    if solver is None:
        st.info("Resuelve la estructura para generar el desarrollo paso a paso.")
        return

    force_factor = FORCE_TO_NEWTONS[st.session_state.force_unit]
    length_factor = LENGTH_TO_METERS[st.session_state.length_unit]
    dof_labels = [f"d{index + 1}" for index in range(len(solver.displacements))]
    free_dofs = set(solver.free_dofs.tolist())

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
    st.dataframe(pd.DataFrame(dof_rows), width="stretch")

    st.markdown("### Paso 2: Análisis barra por barra")
    ui_elements = st.session_state.elements_df.set_index("Barra", drop=False)
    distributed_loads = st.session_state.distributed_loads_df
    for element in solver.elements:
        node_i = next(node for node in solver.nodes if node.id == element.node_i)
        node_j = next(node for node in solver.nodes if node.id == element.node_j)
        c, s = solver.element_direction_cosines(element)
        length = solver.element_length(element)
        theta = float(np.degrees(np.arctan2(s, c)))
        input_row = ui_elements.loc[element.id]
        if isinstance(input_row, pd.DataFrame):
            input_row = input_row.iloc[0]
        st.markdown(f"**Barra {element.id}: nodo {element.node_i} → nodo {element.node_j}**")
        st.write({
            f"(x1, y1) [{st.session_state.length_unit}]": (node_i.x / length_factor, node_i.y / length_factor),
            f"(x2, y2) [{st.session_state.length_unit}]": (node_j.x / length_factor, node_j.y / length_factor),
            f"L [{st.session_state.length_unit}]": length / length_factor,
            f"A [{st.session_state.length_unit}²]": float(input_row["A"]),
            f"E [{st.session_state.force_unit}/{st.session_state.length_unit}²]": float(input_row["E"]),
            "θ [°]": theta,
            "cos θ": c,
            "sin θ": s,
        })
        st.write("Matriz de rigidez local 4 × 4 (coordenadas de la barra)")
        st.dataframe(pd.DataFrame(
            solver.element_stiffness_local_4x4(element),
            index=[f"d{element.node_i}x", f"d{element.node_i}y", f"d{element.node_j}x", f"d{element.node_j}y"],
            columns=[f"d{element.node_i}x", f"d{element.node_i}y", f"d{element.node_j}x", f"d{element.node_j}y"],
        ), width="stretch")
        st.write("Matriz de rigidez global 4 × 4 (coordenadas de la estructura)")
        st.dataframe(pd.DataFrame(
            solver.element_stiffness_global(element),
            index=[f"d{element.node_i}x", f"d{element.node_i}y", f"d{element.node_j}x", f"d{element.node_j}y"],
            columns=[f"d{element.node_i}x", f"d{element.node_i}y", f"d{element.node_j}x", f"d{element.node_j}y"],
        ), width="stretch")
        member_load_rows = distributed_loads[distributed_loads["Barra"] == element.id]
        member_loads = [load for load in solver.distributed_loads if load.element_id == element.id]
        for ((_, load_row), member_load) in zip(member_load_rows.iterrows(), member_loads):
            st.write(
                f"Carga distribuida axial: q_i={float(load_row['q_i']):g}, "
                f"q_j={float(load_row['q_j']):g} {st.session_state.force_unit}/{st.session_state.length_unit}"
            )
            equivalent_local = solver.element_distributed_load_local(member_load) / force_factor
            equivalent_global = solver.element_distributed_load_global(member_load) / force_factor
            st.write("Fuerzas nodales equivalentes locales, en fuerza de entrada:")
            st.write(equivalent_local)
            st.write("Fuerzas nodales equivalentes globales, en fuerza de entrada:")
            st.write(equivalent_global)

    st.markdown("### Paso 3: Ensamble y partición de la matriz global")
    st.write("Aportes de cada barra: K_global[I_i, I_j] += K_elemento[i, j]")
    assembly_rows = []
    for element in solver.elements:
        element_dofs = [*solver.node_dof_indices(element.node_i), *solver.node_dof_indices(element.node_j)]
        element_matrix = solver.element_stiffness_global(element)
        for local_i, global_i in enumerate(element_dofs):
            for local_j, global_j in enumerate(element_dofs):
                assembly_rows.append({
                    "Barra": element.id,
                    "Fila global": dof_labels[global_i],
                    "Columna global": dof_labels[global_j],
                    "Aporte [N/m]": element_matrix[local_i, local_j],
                })
    st.dataframe(pd.DataFrame(assembly_rows), width="stretch")
    st.write("K_global [N/m]")
    st.dataframe(pd.DataFrame(solver.global_stiffness, index=dof_labels, columns=dof_labels), width="stretch")
    st.write("Bloques según GDL libres (L) y restringidos (R)")
    block_columns = st.columns(2)
    for column, key, row_ids, col_ids in (
        (block_columns[0], "K_LL", solver.free_dofs, solver.free_dofs),
        (block_columns[1], "K_LR", solver.free_dofs, solver.restrained_dofs),
        (block_columns[0], "K_RL", solver.restrained_dofs, solver.free_dofs),
        (block_columns[1], "K_RR", solver.restrained_dofs, solver.restrained_dofs),
    ):
        with column:
            st.write(key)
            st.dataframe(pd.DataFrame(
                solver.partition[key],
                index=[dof_labels[index] for index in row_ids],
                columns=[dof_labels[index] for index in col_ids],
            ), width="stretch")

    st.markdown("### Paso 4: Vector P y solución")
    load_vector = solver.global_load_vector()
    st.write("P_global: cargas nodales más fuerzas equivalentes distribuidas [N]")
    st.dataframe(pd.DataFrame({"GDL": dof_labels, "P [N]": load_vector}), width="stretch")
    st.latex(r"K_{LL}D_L=P_L-K_{LR}D_R")
    st.write({
        "GDL libres": [dof_labels[index] for index in solver.free_dofs],
        "P_L [N]": solver.partition["F_L"],
        "D_L [m]": solver.displacements[solver.free_dofs],
        "GDL restringidos": [dof_labels[index] for index in solver.restrained_dofs],
        "D_R [m]": solver.displacements[solver.restrained_dofs],
    })
    st.write("Fuerzas axiales de extremo por barra")
    st.dataframe(pd.DataFrame([{
        "Barra": result["Barra"],
        f"N_i ({st.session_state.force_unit})": result["N_i"] / force_factor,
        f"N_j ({st.session_state.force_unit})": result["N_j"] / force_factor,
        "Estado": result["estado"],
    } for result in solver.calculate_axial_forces()]), width="stretch")
    st.write("Reacciones en GDL restringidos")
    st.dataframe(pd.DataFrame({
        "GDL": [dof_labels[index] for index in solver.restrained_dofs],
        f"Reacción ({st.session_state.force_unit})": solver.reactions / force_factor,
    }), width="stretch")


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
    nodes_df["Nodo"] = pd.to_numeric(nodes_df["Nodo"], errors="coerce")
    nodes_df["X"] = pd.to_numeric(nodes_df["X"], errors="coerce")
    nodes_df["Y"] = pd.to_numeric(nodes_df["Y"], errors="coerce")
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
                    text=f"q={q_peak:g} {force_unit}/{length_unit}",
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
        dof_x = (node_id - 1) * 2 + 1
        dof_y = dof_x + 1
        for x_head, y_head, label, color in (
            (x0 + arrow_x, y0, f"d{dof_x}", "#2563eb"),
            (x0, y0 + arrow_y, f"d{dof_y}", "#059669"),
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
            solver = st.session_state.solution
            results = {
                "nodes": st.session_state.nodes_df.to_dict("records"),
                "elements": st.session_state.elements_df.to_dict("records"),
                "supports": st.session_state.supports_df.to_dict("records"),
                "loads": st.session_state.loads_df.to_dict("records"),
                "distributed_loads": st.session_state.distributed_loads_df.to_dict("records"),
                "K_global": solver.global_stiffness,
                "F_global": solver.global_load_vector(),
                "validation": st.session_state.validation,
            }
            path = export_analysis_excel(results, "Memoria_Analisis_Estructural.xlsx")
            with open(path, "rb") as file:
                st.download_button("Descargar archivo Excel", file.read(), file_name="Memoria_Analisis_Estructural.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Primero resuelve la estructura para generar el archivo Excel.")

    if st.button("Preparar memoria PDF", width="stretch"):
        if st.session_state.solution is not None:
            solver = st.session_state.solution
            results = {
                "nodes": st.session_state.nodes_df.to_dict("records"),
                "elements": st.session_state.elements_df.to_dict("records"),
                "supports": st.session_state.supports_df.to_dict("records"),
                "loads": st.session_state.loads_df.to_dict("records"),
                "distributed_loads": st.session_state.distributed_loads_df.to_dict("records"),
                "K_global": solver.global_stiffness,
                "F_global": solver.global_load_vector(),
                "validation": st.session_state.validation,
            }
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
        st.subheader("Diagrama interactivo")
        render_structure_plot()
    with st.expander("Resultados del análisis", expanded=st.session_state.solution is not None):
        render_solution()
    with st.expander("Desarrollo detallado paso a paso"):
        render_detailed_development()
else:
    tabs = st.tabs(["Nodos", "Barras", "Apoyos", "Cargas", "Solución", "Desarrollo detallado", "Diagrama", "Validación"])
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
                st.session_state[state_key], key=editor_key, num_rows="dynamic", width="stretch",
                column_config={"Tipo": st.column_config.SelectboxColumn("Tipo", options=["Axial"], required=True)} if state_key == "distributed_loads_df" else None,
                on_change=mark_structure_changed,
            )
    with tabs[4]:
        render_solution()
    with tabs[5]:
        render_detailed_development()
    with tabs[6]:
        st.subheader("Diagrama interactivo")
        render_structure_plot()
    with tabs[7]:
        st.write("Validación del ejercicio:")
        st.write(st.session_state.validation)
