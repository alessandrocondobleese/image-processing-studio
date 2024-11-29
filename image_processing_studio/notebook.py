import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium", app_title="Image Processing Studio")


@app.cell
def __(mo):
    _file_label = "Arrastra archivos aquí o haz clic para explorar."
    _file_filetypes = (".jpg", ".png", ".pfd")
    file = mo.ui.file(
        label=_file_label, filetypes=_file_filetypes, kind="area", multiple=False
    )
    return (file,)


@app.cell
def __(file):
    file_buffer = file.contents()
    return (file_buffer,)


@app.cell
def __(file, file_buffer, mo):
    _callout_kind = "neutral"
    _callout_label = mo.md("**Sube una imagen para empezar**").center()
    _callout = mo.md(
        f"""
            { _callout_label }
            { file }
        """
    ).callout(kind=_callout_kind)
    _callout if file_buffer is None else None
    return


@app.cell
def __(file_buffer, mo):
    stop_condition = file_buffer is None
    mo.stop(stop_condition)
    return (stop_condition,)


@app.cell
def __(chartlib, file_buffer, imagelib, mo, stop_condition):
    mo.stop(
        stop_condition,
    )
    preview_image = imagelib.image_from_buffer(file_buffer)
    preview_image_histograms = imagelib.image_channels_histograms(preview_image)
    preview_image_histogram_chart, preview_histograms = chartlib.histogram_chart(
        preview_image_histograms
    )
    return (
        preview_histograms,
        preview_image,
        preview_image_histogram_chart,
        preview_image_histograms,
    )


@app.cell
def __(mo, preview_image_histogram_chart):
    preview_image_histogram = mo.ui.altair_chart(preview_image_histogram_chart)
    return (preview_image_histogram,)


@app.cell
def __(imagelib, preview_histograms, preview_image_histogram):
    preview_image_histogram_stats = imagelib.image_channel_histogram_stats(
        preview_image_histogram.apply_selection(preview_histograms)
    )
    return (preview_image_histogram_stats,)


@app.cell
def __(preview_image_histogram_stats):
    preview_image_stats = [
        (preview_image_histogram_stats[stat_key], stat_label)
        for (stat_key, stat_label) in [
            ("mean", "Promedio"),
            ("standard_deviation", "Desviación estándar"),
            ("skewness", "Asimetría"),
        ]
    ]
    return (preview_image_stats,)


@app.cell
def __(
    dashboardlib,
    preview_image,
    preview_image_histogram,
    preview_image_stats,
):
    preview_dashboard = dashboardlib.dashboard(
        preview_image, preview_image_histogram, preview_image_stats
    )
    return (preview_dashboard,)


@app.cell
def __(mo):
    get_processors, set_processors = mo.state([])
    return get_processors, set_processors


@app.cell
def __(get_processors, mo, preview_image):
    @mo.cache
    def process_image(image, processors):
        processed_image = image.copy()
        for _processor, _processor_params in processors:
            processed_image = _processor.execute(
                processed_image, _processor_params
            )
        return processed_image


    processed_image = process_image(preview_image, get_processors())
    return process_image, processed_image


@app.cell
def __(chartlib, imagelib, processed_image):
    processed_image_histogram = imagelib.image_channels_histograms(processed_image)
    processed_image_histogram_chart, processed_histograms = (
        chartlib.histogram_chart(processed_image_histogram)
    )
    return (
        processed_histograms,
        processed_image_histogram,
        processed_image_histogram_chart,
    )


@app.cell
def __(mo, processed_image_histogram_chart):
    processed_dashboard_histogram = mo.ui.altair_chart(
        processed_image_histogram_chart
    )
    return (processed_dashboard_histogram,)


@app.cell
def __(imagelib, processed_dashboard_histogram, processed_histograms):
    processed_image_histogram_stats = imagelib.image_channel_histogram_stats(
        processed_dashboard_histogram.apply_selection(processed_histograms)
    )
    return (processed_image_histogram_stats,)


@app.cell
def __(processed_image_histogram_stats):
    processed_image_stats = [
        (processed_image_histogram_stats[stat_key], stat_label)
        for (stat_key, stat_label) in [
            ("mean", "Promedio"),
            ("standard_deviation", "Desviación estándar"),
            ("skewness", "Asimetría"),
        ]
    ]
    return (processed_image_stats,)


@app.cell
def __(
    dashboardlib,
    processed_dashboard_histogram,
    processed_image,
    processed_image_stats,
):
    processed_dashboard = dashboardlib.dashboard(
        processed_image, processed_dashboard_histogram, processed_image_stats
    )
    return (processed_dashboard,)


@app.cell
def __(equalizationlib, filterlib, mo, noiselib, thresholdlib):
    GROUPS = {
        "noise": {
            "label": "Ruido",
            "processors": {
                "salt": noiselib.Salt(),
                "pepper": noiselib.Pepper(),
            },
        },
        "threshold": {
            "label": "Umbralización",
            "processors": {
                "invert": thresholdlib.Invert(),
                "binary": thresholdlib.Binary(),
                "otsu": thresholdlib.Otsu(),
                "mean": thresholdlib.Mean(),
            },
        },
        "filter": {
            "label": "Filtro",
            "processors": {
                "gaussian": filterlib.Gaussian(),
                "average": filterlib.Average(),
                "median": filterlib.Median(),
                "mode": filterlib.Mode(),
                "minimum": filterlib.Minimum(),
                "maximum": filterlib.Maximum(),
                "kirsch": filterlib.Kirsch(),
            },
        },
        "equalization": {
            "label": "Ecualización",
            "processors": {
                "rayleigh": equalizationlib.Rayleigh(),
                "uniform": equalizationlib.Uniform(),
            },
        },
    }


    PROCESSOR_PARAMS = mo.ui.dictionary(
        {
            "salt": mo.md("{amount}").batch(
                amount=mo.ui.number(
                    start=0.01,
                    step=0.01,
                    stop=1,
                    label="Cantidad de Sal",
                    full_width=True,
                )
            ),
            "pepper": mo.md("{amount}").batch(
                amount=mo.ui.number(
                    start=0.01,
                    step=0.01,
                    stop=1,
                    label="Cantidad de Pimienta",
                    full_width=True,
                )
            ),
            "binary": mo.md("{threshold}").batch(
                threshold=mo.ui.number(
                    start=0,
                    step=1,
                    stop=255,
                    label="Umbral",
                    full_width=True,
                )
            ),
            "invert": mo.md("").batch(),
            "otsu": mo.md("").batch(),
            "mean": mo.md("").batch(),
            "uniform": mo.md("").batch(),
            "rayleigh": mo.md("{alpha}").batch(
                alpha=mo.ui.number(
                    start=0,
                    value=200,
                    step=10,
                    stop=500,
                    label="Alpha",
                    full_width=True,
                )
            ),
            "gaussian": mo.md(
                """
                {size}
                {sigma}
                """
            ).batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                ),
                sigma=mo.ui.number(
                    start=0,
                    label="Sigma",
                    full_width=True,
                ),
            ),
            "average": mo.md("{size}").batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                )
            ),
            "median": mo.md("{size}").batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                )
            ),
            "mode": mo.md("{size}").batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                )
            ),
            "minimum": mo.md("{size}").batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                )
            ),
            "maximum": mo.md("{size}").batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                )
            ),
            "kirsch": mo.md("{size}").batch(
                size=mo.ui.number(
                    start=3,
                    step=2,
                    stop=10,
                    label="Tamaño de kernel",
                    full_width=True,
                    value=None,
                )
            ),
        }
    )
    return GROUPS, PROCESSOR_PARAMS


@app.cell
def __(GROUPS, mo):
    group_tabs = mo.ui.tabs(
        tabs={
            group_key: group_data["label"]
            for group_key, group_data in GROUPS.items()
        },
        value="noise",
    )
    return (group_tabs,)


@app.cell
def __(GROUPS, mo):
    tab_contents = []
    for group_key, group_data in GROUPS.items():
        group_selector = mo.ui.radio(
            options=group_data["processors"].keys(),
            value=list(group_data["processors"].keys())[0],
            label=f"Selecciona un {group_data['label']}",
        )
        tab_contents.append((group_key, group_selector))
    return group_data, group_key, group_selector, tab_contents


@app.cell
def __(GROUPS, mo):
    group_selectors = {
        group_key: mo.ui.radio(
            options=group_data["processors"].keys(),
            value=list(group_data["processors"].keys())[0],
            label=f"Selecciona un {group_data['label']}",
        )
        for group_key, group_data in GROUPS.items()
    }
    return (group_selectors,)


@app.cell
def __(GROUPS, group_tabs, tab_contents):
    active_tab = group_tabs.value  # Pestaña activa seleccionada
    selected_processor_key = dict(tab_contents)[active_tab].value
    selected_processor = GROUPS[active_tab]["processors"][selected_processor_key]
    return active_tab, selected_processor, selected_processor_key


@app.cell
def __(GROUPS, active_tab, mo):
    processor_radio = mo.ui.radio(
        GROUPS[active_tab]["processors"].keys(),
        value=list(GROUPS[active_tab]["processors"].keys())[0],
    )
    return (processor_radio,)


@app.cell
def __(PROCESSOR_PARAMS, add_processor, processor_radio):
    processor_params_form = PROCESSOR_PARAMS[processor_radio.value].form(
        submit_button_label="Agregar",
        clear_on_submit=True,
        on_change=lambda values: add_processor(values),
    )
    return (processor_params_form,)


@app.cell
def __(group_tabs, mo, processor_params_form, processor_radio):
    processors_form = mo.hstack(
        [
            mo.vstack(
                [
                    group_tabs,
                    processor_radio,
                ]
            ),
            processor_params_form,
        ],
        widths=[1, 2],
        justify="start",
    )
    return (processors_form,)


@app.cell
def __(labels_to_proccessors, processor_radio, set_processors):
    def add_processor(values):
        if values is not None:
            set_processors(
                lambda prev: prev
                + [
                    (
                        labels_to_proccessors[processor_radio.value],
                        values,
                    )
                ]
            )
    return (add_processor,)


@app.cell
def __(file, mo, preview_dashboard, processed_dashboard, processors_form):
    mo.vstack(
        [
            file,
            mo.vstack(
                [
                    preview_dashboard,
                    processors_form,
                    mo.lazy(processed_dashboard),
                ],
                gap=1.5,
            ),
        ],
        gap=6.0,
    )
    return


@app.cell
def __(get_processors, mo, set_processors):
    undo_button = mo.ui.button(
        on_click=lambda x: set_processors(lambda prev: prev[:-1]),
        disabled=len(get_processors()) == 0,
        label="Deshacer transformación",
    )

    delete_button = mo.ui.button(
        on_click=lambda x: set_processors([]),
        disabled=len(get_processors()) == 0,
        label="Eliminar transformaciones",
    )
    return delete_button, undo_button


@app.cell
def __(delete_button, mo, opencv, processed_image, undo_button):
    mo.hstack(
        [
            mo.hstack(
                [
                    undo_button,
                    delete_button,
                ]
            ),
            mo.download(
                data=opencv.imencode(".jpg", processed_image)[1].tobytes(),
                filename="processed-image",
                label="Descargar imagen",
            ),
        ],
        gap=0.75,
    )
    return


@app.cell
def __(get_processors, labels_to_proccessors, mo, pandas, stop_condition):
    mo.stop(stop_condition)

    pandas.DataFrame(
        [
            {
                "processor": list(labels_to_proccessors.keys())[
                    list(labels_to_proccessors.values()).index(processor)
                ],
                "parameters": processor_params,
            }
            for (processor, processor_params) in get_processors()
        ]
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import pandas
    import numpy
    import altair
    import cv2 as opencv
    return altair, numpy, opencv, pandas


@app.cell
def __():
    import importlib
    import json
    import lib.components.dashboard as dashboardlib
    import lib.image_processing.image as imagelib
    import lib.image_processing.chart as chartlib
    import lib.image_processing.processors.equalization as equalizationlib
    import lib.image_processing.processors.noise as noiselib
    import lib.image_processing.processors.filter as filterlib
    import lib.image_processing.processors.threshold as thresholdlib
    return (
        chartlib,
        dashboardlib,
        equalizationlib,
        filterlib,
        imagelib,
        importlib,
        json,
        noiselib,
        thresholdlib,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
