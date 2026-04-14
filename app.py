
# --- UPDATED FUNCTIONS FOR BETTER FONT VISIBILITY ---

def plot_lines(df, cols, names, colors, title, height=350):
    fig = go.Figure()
    for c, n, clr in zip(cols, names, colors):
        fig.add_trace(go.Scatter(
            x=df["label"], y=df[c],
            mode="lines",
            name=n,
            line=dict(color=clr, width=2)
        ))

    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=title,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(
            orientation="h",
            y=-0.15,
            font=dict(color="#E6EEF8", size=11)
        ),
        paper_bgcolor="#0f1520",
        plot_bgcolor="#080c14",
        font=dict(color="#E6EEF8"),
        xaxis=dict(color="#A9B8D4", gridcolor="#1f2a3a"),
        yaxis=dict(color="#A9B8D4", gridcolor="#1f2a3a")
    )
    return fig


def plot_area(df, col, name, color, title, height=280):
    def hex_to_rgba(hex_color, alpha=0.12):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"],
        y=df[col],
        fill="tozeroy",
        name=name,
        line=dict(color=color, width=2),
        fillcolor=hex_to_rgba(color, 0.12)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=title,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="#0f1520",
        plot_bgcolor="#080c14",
        font=dict(color="#E6EEF8"),
        xaxis=dict(color="#A9B8D4", gridcolor="#1f2a3a"),
        yaxis=dict(color="#A9B8D4", gridcolor="#1f2a3a")
    )
    return fig
