from __future__ import annotations

import io
from typing import Any, Dict

from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def export_analysis_excel(results: Dict[str, Any], output_path: str = "Memoria_Analisis_Estructural.xlsx") -> str:
    wb = Workbook()
    ws = wb.active
    ws.title = "Datos_Nodos"
    ws.append(["Nodo", "X", "Y"])
    for node in results.get("nodes", []):
        ws.append([node["Nodo"], node["X"], node["Y"]])

    ws_barras = wb.create_sheet("Datos_Barras")
    ws_barras.append(["Barra", "Nodo_i", "Nodo_j", "E", "A"])
    for element in results.get("elements", []):
        ws_barras.append([element["Barra"], element["Nodo_i"], element["Nodo_j"], element["E"], element["A"]])

    ws_apoyos = wb.create_sheet("Apoyos")
    ws_apoyos.append(["Nodo", "Ux", "Uy"])
    for support in results.get("supports", []):
        ws_apoyos.append([support["Nodo"], support["Ux"], support["Uy"]])

    ws_cargas = wb.create_sheet("Cargas")
    ws_cargas.append(["Nodo", "Fx", "Fy"])
    for load in results.get("loads", []):
        ws_cargas.append([load["Nodo"], load["Fx"], load["Fy"]])

    ws_global = wb.create_sheet("K_Global")
    if "K_global" in results:
        for row in results["K_global"].tolist():
            ws_global.append([float(v) for v in row])

    ws_vector = wb.create_sheet("Vector_F")
    if "F_global" in results:
        ws_vector.append(["F"])
        for value in results["F_global"]:
            ws_vector.append([float(value)])

    ws_validation = wb.create_sheet("Validacion")
    ws_validation.append(["Estado", results.get("validation", {}).get("status", "PENDING")])
    ws_validation.append(["Mensaje", results.get("validation", {}).get("message", "")])

    wb.save(output_path)
    return output_path


def generate_pdf_report(results: Dict[str, Any], output_path: str = "Memoria_Analisis_Estructural.pdf") -> str:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, leading=22, alignment=1, spaceAfter=18)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, leading=14, spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=11)

    story = []
    story.append(Paragraph("ANÁLISIS ESTRUCTURAL II", title_style))
    story.append(Paragraph("MÉTODO MATRICIAL DE RIGIDEZ", title_style))
    story.append(Paragraph("Ejercicio N.º 1", styles["Heading1"]))
    story.append(Spacer(1, 18))

    sections = [
        "1. Datos",
        "2. Geometría",
        "3. Propiedades",
        "4. Condiciones de apoyo",
        "5. Cargas",
        "6. Grados de libertad",
        "7. Matrices de rigidez de elementos",
        "8. Ensamblaje de matriz global",
        "9. Partición de matrices",
        "10. Desplazamientos",
        "11. Reacciones",
        "12. Fuerzas axiales",
        "13. Validación",
    ]
    for section in sections:
        story.append(Paragraph(section, section_style))
        story.append(Paragraph("Sección generada por la aplicación. La validación final contra el PDF del curso queda pendiente porque el archivo adjunto no se encuentra en el workspace.", body_style))

    if "K_global" in results:
        story.append(Paragraph("Matriz K global", section_style))
        matrix = results["K_global"]
        data = [[f"{float(value):.4g}" for value in row] for row in matrix]
        table = Table(data)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
        ]))
        story.append(table)

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    doc.build(story)
    return output_path


def export_excel_bytes(results: Dict[str, Any]) -> bytes:
    bio = io.BytesIO()
    export_analysis_excel(results, output_path="temp.xlsx")
    from openpyxl import load_workbook
    wb = load_workbook("temp.xlsx")
    wb.save(bio)
    bio.seek(0)
    return bio.read()
