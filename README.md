# Análisis Estructural II – Método de Rigidez

Este proyecto implementa una aplicación web en Streamlit para resolver armaduras planas 2D mediante el método de rigidez directa.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

## Método utilizado

Se emplea la formulación matricial de rigidez directa para elementos de barra en 2D. Cada barra se modela con:

- longitud L
- cosenos directores c y s
- matriz de rigidez global
- ensamblaje en la matriz global K
- partición de grados de libertad libres y restringidos
- solución del sistema K_LL * D_L = F_L - K_LR * D_R

## Convenciones de signos

- Las cargas positivas se toman en el sentido positivo de los ejes.
- La fuerza axial se reporta como:
  - TRACCIÓN cuando N > 0
  - COMPRESIÓN cuando N < 0
  - FUERZA NULA cuando |N| ≈ 0
- Los apoyos se codifican con:
  - 0 = desplazamiento restringido
  - 1 = desplazamiento libre
  - valores numéricos distintos de 0/1 se interpretan como desplazamientos prescritos

## Grados de libertad

Cada nodo tiene dos grados de libertad:

- Ux = 2*i
- Uy = 2*i + 1

## Unidades

La configuración base se usa en sistema SI:

- Fuerzas: N
- Longitudes: m
- Esfuerzos: Pa

Se permite la elección de unidades para la representación visual, pero la resolución interna se mantiene coherente en un sistema SI.

## Exportación

La aplicación permite exportar:

- Excel con la memoria de cálculo
- PDF con la memoria de cálculo

## Validación

La validación definitiva contra el PDF oficial del curso queda pendiente hasta disponer del archivo adjunto completo. El proyecto incluye una estructura preparada para recibir los datos reales del Ejercicio N.º 1 y una rutina de validación explícita que reporta exactamente qué información falta.
