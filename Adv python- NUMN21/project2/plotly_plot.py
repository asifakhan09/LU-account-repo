from collections.abc import Callable, Sequence
import numpy as np
from plotly import graph_objects as go, io as pio, subplots

pio.templates.default = "plotly_dark"


def trace_on_heatmap(
    f: Callable,
    trace: np.ndarray | None,
    xr: tuple[float, float] = (-5, 5),
    yr: tuple[float, float] | None = None,
    resolution: int = 15,
    function_is_vectorized=False,
):
    if yr is None:
        yr = xr

    xx = np.linspace(xr[0], xr[1], resolution)
    yy = np.linspace(yr[0], yr[1], resolution)
    X, Y = np.meshgrid(xx, yy)
    if function_is_vectorized:
        Z = f(np.stack([X, Y], axis=-1))
    else:
        Z = np.zeros((resolution, resolution))
        for r in range(resolution):
            for c in range(resolution):
                Z[c, r] = f(np.array([xx[r], yy[c]]))

    fig = go.Figure(
        go.Heatmap(z=Z, x=xx, y=xx),
        go.Layout(yaxis_scaleanchor="x", width=400, showlegend=False),
    )

    if trace is not None:
        fig.add_traces(
            [
                go.Scatter(
                    x=trace[:, 0],
                    y=trace[:, 1],
                    line_color="white",
                    mode="lines+markers",
                    marker_size=4,
                    line_width=1,
                ),
                go.Scatter(
                    x=trace[-1, 0, None],
                    y=trace[-1, 1, None],
                    line_color="white",
                    mode="markers",
                    marker_size=10,
                ),
            ]
        )

    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False)

    return fig


def many_heat_maps(
    funs: Sequence[Callable],
    xr: tuple[float, float] = (-5, 5),
    yr: tuple[float, float] | None = None,
    resolution: int = 15,
    function_is_vectorized=False,
    log_scale=False,
    log_offset=0.1,
):
    if yr is None:
        yr = xr

    xx = np.linspace(xr[0], xr[1], resolution)
    yy = np.linspace(yr[0], yr[1], resolution)
    X, Y = np.meshgrid(xx, yy)

    fig = subplots.make_subplots(1, len(funs), subplot_titles=[str(f) for f in funs])
    for i, f in enumerate(funs):
        if function_is_vectorized:
            Z = f(np.stack([X, Y], axis=-1))
        else:
            Z = np.zeros((resolution, resolution))
            for r in range(resolution):
                for c in range(resolution):
                    Z[c, r] = f(np.array([xx[r], yy[c]]))

        if log_scale:
            Z = np.log(Z + log_offset)
        fig.add_trace(go.Heatmap(z=Z, x=xx, y=xx, showscale=False), row=1, col=i + 1)

    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False, scaleanchor="x")

    return fig
