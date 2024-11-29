import pandas
import altair


def histogram_chart(histograms: pandas.DataFrame):
    if "gray" in histograms.columns:
        histograms = histograms.reset_index().rename(columns={"gray": "frequency"})
        chart = (
            altair.Chart(histograms.reset_index())
            .mark_area(color="#7f7f7f")
            .encode(
                x=altair.X("index:Q", title="Pixel Intensity"),
                y=altair.Y("frequency:Q", title="Frequency"),
            )
        )

        return chart, histograms

    histograms = histograms.reset_index().melt(
        "index", var_name="channel", value_name="frequency"
    )
    channel_to_color = {"red": "#FF7F0E", "green": "#2CA02C", "blue": "#1F77B4"}
    chart = (
        altair.Chart(histograms)
        .mark_area(opacity=0.5)
        .encode(
            x=altair.X("index:Q", title="Pixel Intensity"),
            y=altair.Y("frequency:Q", title="Frequency"),
            color=altair.Color(
                "channel:N",
                scale=altair.Scale(
                    domain=list(channel_to_color.keys()),
                    range=list(channel_to_color.values()),
                ),
            ),
        )
    )

    return chart, histograms
