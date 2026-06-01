# Apéndice A — Plan de Proyecto Software

## A.1. Introducción

En este apéndice se documenta la planificación temporal del proyecto, la
metodología de gestión utilizada y el estudio de viabilidad económica y legal
que sustenta la realización del trabajo.

El proyecto **EEG ADHD Classifier** se ha desarrollado como Trabajo de Fin de
Grado del Grado en Ingeniería Informática de la Universidad de Burgos durante
el curso 2025-2026. Su objetivo es construir un sistema de apoyo académico
para la clasificación binaria de señales EEG (ADHD vs Control) mediante una
aplicación web que integra modelos de Machine Learning clásico y Deep
Learning, con énfasis en el rigor metodológico de la evaluación
*cross-subject*.

El TFG está dirigido por los siguientes docentes:

- **Tutor**: por confirmar (Departamento de Ingeniería Informática, UBU).

El producto entregado consta de:

- Una **API REST** desarrollada con FastAPI.
- Una **SPA** desarrollada con React 19 + Vite.
- Una **base de datos PostgreSQL 16** para el histórico de experimentos.
- Un **pipeline de investigación reproducible** en `scripts/` para entrenar y
  exportar los modelos servidos por la API.
- Tests automatizados, CI con GitHub Actions y análisis estático con
  SonarCloud.

### A.1.1. Metodología

Se ha utilizado un enfoque **Scrum adaptado** al contexto de un TFG
individual, soportado por **ZenHub** como gestor de tareas integrado con
GitHub Issues:

- **Sprints de duración variable** (entre 1 y 3 semanas), ajustados al ritmo
  académico y a la disponibilidad real del autor. Esta flexibilidad se ha
  considerado preferible a forzar sprints fijos que no reflejarían el
  trabajo real, compatibilizado con otras asignaturas del último curso.
- **Cada sprint cierra con un incremento funcional comprometible** en
  `main`, verificable a través del repositorio Git público.
- **Backlog estructurado en Epics** gestionados desde ZenHub. Cada Epic
  agrupa las *issues* (historias de usuario y tareas técnicas) necesarias
  para alcanzar un objetivo concreto del proyecto.
- **Métricas Scrum**: ZenHub proporciona burndown y velocity por sprint,
  que sirven como evidencia de cumplimiento y ritmo. Las capturas
  correspondientes se incluyen en la sección A.3.
- **Trazabilidad bidireccional commit ↔ issue**: los commits relevantes
  enlazan con las *issues* de GitHub a las que pertenecen, manteniendo
  el rastro entre código entregado y tarea planificada.
- **Revisiones con el tutor** en las reuniones programadas a lo largo del
  curso, ajustando el contenido de los siguientes sprints en función del
  feedback recibido.

> **Enlace al tablero ZenHub**: <https://app.zenhub.com/workspaces/tfg-adhd-class-69bc5516af887f0036c68928/board>.
> Al tratarse de un workspace asociado a un repositorio público y a la
> cuenta personal del autor, el acceso completo al tablero requiere
> autenticación e invitación. Por este motivo, las gráficas relevantes
> (lista de Epics, tableros por sprint, burndown, velocity) se reproducen
> en este apéndice como figuras A.1 a A.11, conforme exige la guía oficial
> del TFG de la UBU. Adicionalmente, se invitará al usuario `ubutfgm` y
> a los miembros del tribunal para que dispongan de acceso directo.

### A.1.2. Recursos del proyecto

| Tipo | Recurso                                                                                         |
|------|-------------------------------------------------------------------------------------------------|
| Repositorio | <https://github.com/ajg1005/eeg-adhd-clasificacion>                                       |
| Gestión de tareas | ZenHub (workspace `tfg-adhd-class`, integrado con GitHub Issues) — <https://app.zenhub.com/workspaces/tfg-adhd-class-69bc5516af887f0036c68928/board>                    |
| CI | GitHub Actions (`.github/workflows/ci.yml`)                                                       |
| Calidad | SonarCloud (proyecto `ajg1005_eeg-adhd-clasificacion`)                                      |
| Licencia | MIT (ver `LICENSE` en la raíz del repositorio)                                             |
| Dataset | EEG ADHD de la Universidad Shahed (datos públicos, 121 sujetos, 19 canales, 128 Hz)         |

---

## A.2. Estructura de Sprints

El proyecto se ha estructurado en **ocho sprints temáticos** seguidos de un
sprint final dedicado a documentación y entrega. Cada sprint corresponde a un
incremento funcional verificable en el repositorio mediante commits firmados.

### A.2.1. Resumen de sprints

| Sprint | Fechas              | Tema                        | Resultado verificable                                        |
|--------|---------------------|-----------------------------|--------------------------------------------------------------|
| 0      | 26 feb – 9 mar      | Arranque del proyecto       | Repositorio creado, dataset descargado, artículos enlazados  |
| 1      | 18 mar – 27 mar     | Pipeline base de datos      | Carga, preprocesado, epochs, features básicas, CV inicial    |
| 2      | 6 abr – 15 abr      | Features avanzadas          | Features espectrales, combinadas, CV solo en train           |
| 3      | 19 abr – 21 abr     | Modelos avanzados           | XGBoost añadido; modelos DL (CNN, CNN-LSTM, EEGNet)          |
| 4      | 30 abr – 3 may      | Aplicación básica           | Exportación del mejor modelo, Streamlit inicial, pestañas    |
| 5      | 10 may – 18 may     | Migración a React + FastAPI | SPA con React 19 + Vite, Docker Compose, CI, tests, SonarCloud |
| 6      | 18 may – 27 may     | Calidad y robustez          | Refactor central, robustez de la API, LICENSE, BD inicial    |
| 7      | 27 may – 30 may     | Histórico de experimentos   | Esquema BD (Alembic), pestaña "Experimentos", endurecimiento |
| 8      | 1 jun – 6 jul       | Memoria, anexos y entrega   | Documentación completa, vídeos, release final, defensa       |

### A.2.2. Detalle por sprint

#### Sprint 0 — Arranque del proyecto (26 feb – 9 mar)

- **Objetivos**: localizar dataset adecuado, definir alcance académico,
  preparar repositorio y bibliografía de partida.
- **Tareas realizadas**:
  - Búsqueda y selección del dataset EEG ADHD de la Universidad Shahed
    (publicado en IEEE).
  - Creación del repositorio GitHub.
  - Recopilación de artículos de referencia sobre clasificación EEG-ADHD.
  - Página inicial de enlaces (`docs/enlaces.md`).
- **Evidencia en git**: commits `5064679`, `df7e709`, `1f2eedf`, `5f89fc0`,
  `fb273a3`, `449b6bc`.

#### Sprint 1 — Pipeline base de datos (18 mar – 27 mar)

- **Objetivos**: construir un pipeline reproducible desde el CSV crudo hasta
  un primer baseline de Machine Learning con validación cross-subject.
- **Tareas realizadas**:
  - Carga y preprocesado del CSV: limpieza de filas corruptas, normalización
    de la columna de clase.
  - Segmentación manual de la señal en *epochs* parametrizables (tamaño de
    ventana y solape).
  - Separación de los pacientes (`GroupShuffleSplit`) para evitar fuga de
    información.
  - Extracción de features temporales básicas (media, std, min, max,
    rango, energía).
  - Primer baseline con `RandomForest`, `SVM RBF`, `LogisticRegression` y
    `GaussianNB`.
  - Validación cruzada por grupos.
- **Evidencia en git**: commits `7960240`, `d47ff70`, `7600c53`, `de3a372`.

#### Sprint 2 — Features avanzadas (6 abr – 15 abr)

- **Objetivos**: enriquecer la representación de la señal con información
  del dominio de la frecuencia y aislar correctamente la evaluación.
- **Tareas realizadas**:
  - Extracción de features espectrales mediante `welch`: potencia absoluta
    y relativa por banda (delta, theta, alpha, beta, gamma), entropía
    espectral, ratio theta/beta.
  - Combinación de features temporales y espectrales.
  - Refactor del CV para garantizar que únicamente se realiza sobre el
    conjunto de entrenamiento, reservando el test final para una sola
    evaluación.
- **Evidencia en git**: commits `a587b58`, `5706426`, `55c0197`.

#### Sprint 3 — Modelos avanzados (19 abr – 21 abr)

- **Objetivos**: ampliar el catálogo de modelos clásicos y comenzar la
  exploración de Deep Learning.
- **Tareas realizadas**:
  - Incorporación de XGBoost al catálogo ML.
  - Implementación de modelos DL: CNN-1D, CNN-LSTM y EEGNet.
  - Pipeline DL paralelo al ML, con filtrado pasabanda + z-score por
    sujeto.
- **Evidencia en git**: commits `65a98d8`, `580e0db`, `e2011f0`.

#### Sprint 4 — Aplicación básica (30 abr – 3 may)

- **Objetivos**: trasladar los resultados de investigación a una primera
  aplicación funcional accesible al usuario.
- **Tareas realizadas**:
  - Exportación del mejor modelo ML.
  - Primera versión de la aplicación con Streamlit.
  - Pestañas iniciales con información del modelo y métricas.
  - Tests básicos.
- **Evidencia en git**: commits `5ddfc79`, `548d9cd`, `5b496c6`.

#### Sprint 5 — Migración a React + FastAPI (10 may – 18 may)

- **Objetivos**: profesionalizar la arquitectura desacoplando frontend y
  backend, e introducir prácticas modernas de ingeniería del software.
- **Tareas realizadas**:
  - Reescritura del backend con FastAPI.
  - Reescritura del frontend con React 19 + Vite.
  - Reorganización de la interfaz en cuatro pestañas (Modelo, Dataset,
    Entrenamiento, Predicción).
  - Contenerización con Docker Compose.
  - Configuración de CI con GitHub Actions.
  - Tests con pytest y SonarCloud.
  - Linter `ruff` para Python y ESLint para JavaScript.
- **Evidencia en git**: commits `141b4ff`, `924d1cb`, `d8bf6f8`, `4530174`,
  `ab05277`, `b198d50`, `9933042`, `d24949d`, `447a734`, `aceb7bf`,
  `f1eb637`, `24e3f79`, `23e0294`, `4535196`.

#### Sprint 6 — Calidad y robustez (18 may – 27 may)

- **Objetivos**: consolidar la calidad del código, centralizar puntos de
  decisión y endurecer la API.
- **Tareas realizadas**:
  - Refactor para centralizar paths, registros de modelos y constantes en
    un único punto.
  - Refactor del pipeline de features para reutilización.
  - Mejora de la robustez de los endpoints y de la cobertura de los tests.
  - Alineación de metadatos de entrenamiento y defaults DL.
  - Validación de props en componentes React.
  - Adición del fichero `LICENSE` (MIT) al repositorio.
- **Evidencia en git**: commits `60a3a25`, `45ac9bf`, `2950957`, `28ad71e`,
  `4b86413`, `8f934e0`, `759f1f5`, `f8aaebc`, `4d0d1db`.

#### Sprint 7 — Histórico de experimentos (27 may – 30 may)

- **Objetivos**: dotar al sistema de persistencia para que los entrenamientos
  ejecutados desde la interfaz queden registrados y sean comparables.
- **Tareas realizadas**:
  - Diseño del esquema de tres tablas (`datasets`, `experiments`,
    `experiment_folds`).
  - Implementación con SQLAlchemy 2.0 y migración inicial con Alembic.
  - Servicio PostgreSQL 16 añadido al Docker Compose.
  - Nueva pestaña "Experimentos" en el frontend con listado, filtros y
    detalle.
  - Endurecimiento de seguridad de la BD: usuario sin privilegios en el
    contenedor backend y credenciales por variables de entorno.
- **Evidencia en git**: commits `c9ca85a`, `0afa2bd`.

#### Sprint 8 — Memoria, anexos y entrega (1 jun – 6 jul)

- **Objetivos**: documentar el trabajo realizado, preparar el material de
  defensa y formalizar la entrega.
- **Tareas planificadas**:
  - Redacción de la memoria (~40 páginas) siguiendo la plantilla oficial UBU.
  - Redacción de los anexos A, B, C, D, E (y F si aplica).
  - Vídeo demostrativo del funcionamiento de la aplicación.
  - Release final en GitHub (`v1.0.0`).
  - Encuadernación de la copia física de la memoria.
  - Preparación de la defensa.

---

## A.3. Planificación temporal

### A.3.1. Cronograma global

El proyecto se ha desarrollado de forma continua entre el **26 de febrero de
2026** y la **fecha prevista de defensa: 6 de julio de 2026**, lo que arroja
una duración total de aproximadamente **19 semanas** (4,5 meses).

```
Feb  ─── Mar  ─── Abr  ─── May  ─── Jun  ─── Jul
[S0]    [   S1   ][      S2 + S3       ]
                                       [S4][      S5       ]
                                                      [S6 + S7]
                                                              [        S8        ]
```

### A.3.2. Hitos principales

| Hito                                          | Fecha        | Evidencia                                          |
|-----------------------------------------------|--------------|----------------------------------------------------|
| Inicio del proyecto (commit inicial)          | 26 feb 2026  | Commit `5064679`                                   |
| Primer pipeline ML completo con CV            | 27 mar 2026  | Commit `de3a372` "Cross-validation"                |
| Features espectrales integradas               | 6 abr 2026   | Commit `a587b58`                                   |
| Modelos DL operativos                         | 21 abr 2026  | Commit `e2011f0`                                   |
| Primera versión de la app (Streamlit)         | 30 abr 2026  | Commit `5ddfc79`                                   |
| Migración a React + FastAPI completada        | 14 may 2026  | Commit `924d1cb`                                   |
| CI verde y tests automatizados                | 18 may 2026  | Commit `f1eb637` "CI y tests fixed"                |
| Licencia MIT añadida                          | 25 may 2026  | Commit `759f1f5`                                   |
| BD de experimentos en producción              | 27 may 2026  | Commit `c9ca85a`                                   |
| Pestaña "Experimentos" en frontend            | 30 may 2026  | Commit `0afa2bd`                                   |
| Entrega de memoria y anexos                   | 6 jul 2026   | Por confirmar (depende de plazo oficial)           |
| Defensa pública                               | Julio 2026   | Convocatoria a confirmar                           |

### A.3.3. Evidencia del seguimiento

> **Nota metodológica.** El seguimiento del proyecto se ha realizado en
> **ZenHub** vinculado al repositorio GitHub, con un flujo Kanban estándar
> de ocho carriles (*New Issues → Icebox → Product Backlog → Sprint
> Backlog → In Progress → Review/QA → Done → Closed*) y *sprints*
> periódicos. El primer sprint del proyecto se formalizó adicionalmente
> como GitHub Milestone (*"Sprint 1: 24 feb – 10 mar 2026"*) para dejar
> constancia pública del cierre de las primeras tareas. La trazabilidad de
> cada incremento es verificable a través del historial de commits
> (sección A.2.2). El tablero ZenHub puede consultarse en:
> <https://app.zenhub.com/workspaces/tfg-adhd-class-69bc5516af887f0036c68928/board>
> (requiere autenticación con cuenta GitHub invitada al workspace).

![Figura A.1 — Estado del tablero ZenHub al cierre del Sprint 7. Se observan los ocho carriles del flujo Kanban, las dos issues en curso (Sprint 8) y la distribución de las tareas en Review/QA, Done y Closed.](figuras/anexo_a/zenhub_board.png)

El backlog del proyecto está estructurado en **34 issues** alojadas en el
gestor de incidencias de GitHub (consultables públicamente en
<https://github.com/ajg1005/eeg-adhd-clasificacion/issues?q=is%3Aissue>).
A continuación se presenta la **tabla del backlog completo agrupado por
sprint**, equivalente a la vista del tablero ZenHub y verificable contra
el repositorio público.

#### Sprint 0 — Arranque (Milestone "Sprint 1: 24 feb – 10 mar")

| Issue | Título                                                           | Estado |
|-------|------------------------------------------------------------------|--------|
| #1    | Buscar 4 artículos científicos machine learning bci             | Cerrada |
| #2    | 3 dataset publicos de BCI                                       | Cerrada |
| #3    | Crear markdown con enlaces comentados                           | Cerrada |
| #4    | Revisar datasets                                                | Abierta (milestone "Plantear contenido") |

#### Sprint 1 — Pipeline base de datos (18 – 27 mar)

| Issue | Título                                                           | Estado    |
|-------|------------------------------------------------------------------|-----------|
| #5    | Elegir y cargar dataset en python                               | Completada* |
| #6    | Crear pipeline                                                  | Completada* |
| #7    | Seleccionar modelos a utilizar en ML                            | Completada* |
| #8    | Preprocesamiento inicial del dataset                            | Cerrada    |
| #9    | Revisar herramientas y librerías a utilizar                     | Completada* |
| #11   | Implementar cross-validation por sujetos                        | Completada* |

#### Sprint 2 — Features avanzadas (6 – 15 abr)

| Issue | Título                                                           | Estado     |
|-------|------------------------------------------------------------------|------------|
| #10   | Añadir features espectrales                                     | Completada* |
| #12   | Comparar estadísticas vs espectrales vs combinadas              | Completada* |
| #16   | Terminar parte de ML                                            | Completada* |

#### Sprint 3 — Modelos avanzados (19 – 21 abr)

| Issue | Título                                                           | Estado     |
|-------|------------------------------------------------------------------|------------|
| #13   | Añadir deep learning                                            | Completada* |
| #17   | Preparar parte de deep learning                                 | Completada* |
| #18   | Implementar EEGNet y/o CNN-LSTM                                 | Completada* |

#### Sprint 4 — Aplicación básica (30 abr – 3 may)

| Issue | Título                                                           | Estado     |
|-------|------------------------------------------------------------------|------------|
| #14   | Librerías para la app                                           | Completada* |
| #15   | Lista de tareas de la app                                       | Completada* |
| #19   | Interfaz gráfica                                                | Completada* |
| #20   | Entrenamiento (exportar modelo)                                 | Completada* |

#### Sprint 5 — Migración React + FastAPI (10 – 18 may)

| Issue | Título                                                           | Estado     |
|-------|------------------------------------------------------------------|------------|
| #21   | Crear rama React + FastAPI                                      | Completada* |
| #22   | Crear backend FastAPI                                           | Completada* |
| #23   | Crear interfaz gráfica como en la demo de streamlit             | Completada* |
| #24   | Añadir visualización de gráficas del modelo elegido             | Completada* |
| #25   | Añadir modelo de deep learning                                  | Completada* |
| #26   | Docker part                                                     | Completada* |
| #27   | Añadir vista de carga de datos en React                         | Completada* |
| #28   | Añadir opciones de modelos e hiperparámetros                    | Completada* |
| #29   | Añadir entrenamiento de ML en la web                            | Completada* |
| #30   | Añadir entrenamiento de DL en la app                            | Completada* |

#### Sprint 6 — Calidad y robustez (18 – 27 may)

| Issue | Título                                                           | Estado     |
|-------|------------------------------------------------------------------|------------|
| #31   | Mejorar output de resultados por inferencia con un único paciente | Completada* |
| #32   | Mejorar tests y añadir CI                                       | Completada* |

#### Sprint 7 — Histórico de experimentos (27 – 30 may)

| Issue | Título                                                           | Estado     |
|-------|------------------------------------------------------------------|------------|
| #33   | Añadir database simple de experimentos y parámetros del dataset  | Completada* |
| #34   | Crear pestaña que muestre los experimentos y el database utilizado | Completada* |

> *\* "Completada" indica que el trabajo asociado a la *issue* se completó y
> consolidó en `main`, verificable a través de los commits enumerados en
> la sección A.2.2. Algunas *issues* permanecen abiertas en GitHub al no
> haberse cerrado formalmente; este cierre se realizará durante el sprint
> final de documentación. Se considera que el estado real es "Done" en el
> tablero ZenHub.*

### A.3.4. Métricas globales del seguimiento

| Métrica                                        | Valor   |
|------------------------------------------------|---------|
| Total de *issues* en el backlog                | 34      |
| Sprints planificados                           | 8 + 1 documentación |
| Duración total del proyecto                    | ≈ 19 semanas |
| *Commits* en `main`                            | 39      |
| Milestones formalizados en GitHub              | 2       |
| Issues por sprint (media)                      | ≈ 4,5   |

### A.3.5. Estimación de dedicación

La dedicación al TFG ha sido **parcial**, compatibilizándola con otras
asignaturas del último curso del Grado. Se estima una dedicación media de
**25-30 horas semanales** durante las 19 semanas activas:

| Concepto              | Estimación                |
|-----------------------|---------------------------|
| Semanas activas       | 19                        |
| Horas semanales medias | 27,5                     |
| **Total estimado**    | **≈ 525 horas**           |

Esta estimación incluye el trabajo de investigación, desarrollo de software,
redacción de documentación y reuniones con el tutor.

---

## A.4. Estudio de viabilidad

### A.4.1. Viabilidad económica

A continuación se desglosan los costes asociados al desarrollo del proyecto.
Conforme a las recomendaciones de la guía oficial de TFG de la UBU, se
imputan **costes de personal con la parte impositiva correspondiente a la
empresa** y se aplica **amortización** sobre el hardware (no se imputa el
coste total de adquisición).

#### Coste de personal

Se considera el perfil de un **Ingeniero Informático Junior** desarrollando
las labores típicas de un TFG (investigación, desarrollo, documentación,
reuniones de seguimiento).

| Concepto                                  | Valor              |
|-------------------------------------------|--------------------|
| Salario bruto anual de referencia (junior)| 27 000 €           |
| Salario mensual bruto (14 pagas)          | 1 928,57 €         |
| Coste empresa adicional (~32 %)*          | 617,14 €           |
| **Coste mensual total para la empresa**   | **2 545,71 €**     |
| Horas laborables por mes (jornada completa) | 160               |
| **Coste por hora (jornada completa)**     | **≈ 15,90 €/h**    |

\* La parte impositiva a cargo de la empresa incluye:
- **Contingencias comunes** (23,60 %).
- **Desempleo** (5,50 %, tipo general).
- **FOGASA** (0,20 %).
- **Formación Profesional** (0,60 %).
- **Mecanismo de Equidad Intergeneracional (MEI)** (0,67 %).
- **AT y EP** (variable, estimación ~1,5 %).

Total aproximado: **~32 % sobre el salario bruto**, conforme a las tablas
publicadas por la Seguridad Social para 2025-2026.

**Imputación al proyecto:**

| Concepto         | Cálculo                  | Importe         |
|------------------|--------------------------|-----------------|
| Horas dedicadas  | 525 h                    |                 |
| Coste por hora   | 15,90 €                  |                 |
| **Coste total personal** | 525 × 15,90 €     | **≈ 8 348 €**   |

#### Coste de hardware (amortización)

Se aplica amortización lineal del hardware durante su vida útil, imputando
al proyecto únicamente la parte proporcional al tiempo de uso (4,5 meses).

| Equipo                          | Precio | Vida útil | Amortización mensual | Meses imputados | Coste imputado |
|---------------------------------|--------|-----------|----------------------|-----------------|----------------|
| Ordenador portátil de desarrollo| 1 500 €| 5 años    | 25,00 €              | 4,5             | 112,50 €       |
| Monitor externo                 | 220 €  | 5 años    | 3,67 €               | 4,5             | 16,50 €        |
| **Total hardware**              |        |           |                      |                 | **129 €**      |

#### Coste de software

Todas las herramientas y bibliotecas utilizadas son **software libre**
(licencias MIT, BSD, Apache 2.0 y LGPL — ver `docs/licencias.md`). No se ha
adquirido ninguna licencia comercial.

| Concepto                       | Coste |
|--------------------------------|-------|
| Sistema operativo (Windows preinstalado en el equipo) | 0 € |
| IDE (Visual Studio Code)       | 0 €   |
| Bibliotecas de Python y npm    | 0 €   |
| GitHub (cuenta gratuita)       | 0 €   |
| SonarCloud (plan gratuito open source) | 0 € |
| **Total software**             | **0 €** |

#### Coste de infraestructura

El proyecto se ha desarrollado y probado en local, sin coste de hosting o
servicios en la nube de pago. La eventual fase de despliegue público no se
incluye en este presupuesto al no estar todavía contratada.

| Concepto         | Coste |
|------------------|-------|
| Hosting cloud    | 0 €   |
| Dominio          | 0 €   |
| **Total infraestructura** | **0 €** |

#### Resumen de costes

| Categoría     | Importe        |
|---------------|----------------|
| Personal      | 8 348 €        |
| Hardware      | 129 €          |
| Software      | 0 €            |
| Infraestructura | 0 €          |
| **TOTAL**     | **≈ 8 477 €**  |

> **Nota.** Los importes son una estimación académica orientativa elaborada
> conforme a la guía oficial. No representan un presupuesto real para un
> proyecto comercial. Las cifras del coste de personal usan valores típicos
> del mercado español 2025-2026 para un perfil junior y los tipos
> impositivos vigentes publicados por la Seguridad Social.

### A.4.2. Viabilidad legal

#### Licencia del software desarrollado

El proyecto se distribuye bajo **MIT License** (ver fichero `LICENSE` en la
raíz del repositorio). MIT es una licencia *permisiva de software*, no de
contenido, conforme al requisito de la guía oficial UBU, que excluye
explícitamente las licencias Creative Commons para el código.

La decisión de licencia se justifica por:

1. Es una licencia ampliamente reconocida en la comunidad open source.
2. Facilita la reutilización académica y profesional del código.
3. Es compatible con todas las dependencias seleccionadas (ver siguiente
   sección).

#### Análisis de licencias de dependencias

Se ha realizado un análisis sistemático de la licencia de cada una de las
**30 dependencias directas** del proyecto. El detalle completo se encuentra
en `docs/licencias.md`. Resumen ejecutivo:

| Familia de licencia   | Nº de dependencias | Compatibilidad con MIT  |
|-----------------------|--------------------|-------------------------|
| MIT                   | 19                 | Total                   |
| BSD-3-Clause          | 7                  | Total                   |
| Apache 2.0            | 3                  | Total                   |
| LGPL-3.0-or-later     | 1 (psycopg)        | Total (uso como librería) |
| **TOTAL**             | **30**             | **100 % compatibles**   |

Ninguna dependencia es copyleft fuerte (GPL o AGPL), por lo que el código
propio del TFG puede mantenerse bajo MIT sin restricciones adicionales.

#### Tratamiento de datos personales

El dataset utilizado (EEG ADHD de la Universidad Shahed) es de **acceso
público en IEEE DataPort** y se distribuye con fines de investigación. Los
sujetos son anónimos: cada uno se identifica únicamente con un identificador
numérico (`ID`) sin información personal asociada.

La aplicación desarrollada:

- **No persiste datos clínicos** crudos del paciente. Los CSV cargados por
  el usuario se procesan en memoria y solo se almacenan en la base de datos
  los metadatos estadísticos del dataset (hash, número de filas y columnas,
  número de sujetos, distribución de clases, canales detectados).
- **No identifica al usuario** de la aplicación (sistema sin autenticación,
  uso académico).

Por tanto, no se procesan datos personales en el sentido del **Reglamento
General de Protección de Datos (RGPD, UE 2016/679)** ni de la **LOPDGDD
3/2018**. No se requiere una evaluación de impacto adicional.

#### Carácter no clínico

La aplicación tiene **carácter académico y demostrativo**. No constituye un
producto sanitario en el sentido del Reglamento (UE) 2017/745 y no debe
utilizarse como herramienta de diagnóstico clínico. Esta limitación se hace
explícita en el README y en la memoria del proyecto.

### A.4.3. Viabilidad técnica

El proyecto se considera técnicamente viable por los siguientes motivos:

1. **Dataset disponible y validado**: el conjunto de la Universidad Shahed
   es ampliamente utilizado en la literatura científica para evaluación de
   clasificadores EEG-ADHD, lo que permite contrastar resultados.
2. **Stack maduro**: todas las tecnologías utilizadas (FastAPI, React,
   scikit-learn, TensorFlow, PostgreSQL, Docker) son herramientas estables,
   documentadas y con amplia comunidad.
3. **Hardware suficiente**: el equipo de desarrollo del autor permite
   entrenar tanto los modelos ML como los modelos DL en tiempos razonables
   (no se requiere GPU dedicada, aunque los modelos DL se benefician de
   ella).
4. **Reproducibilidad garantizada**: los pipelines de investigación están
   versionados en `scripts/`, los modelos exportados en `models/`, los
   resultados de CV en `results/`, y las versiones de todas las dependencias
   están fijadas en `requirements.txt`.
5. **Calidad del software**: la integración continua con GitHub Actions y
   el análisis estático con SonarCloud aseguran que cualquier cambio
   mantiene los estándares de calidad establecidos.
