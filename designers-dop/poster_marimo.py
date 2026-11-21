"""Интерактивный генератор случайных величин для постеров.

Запуск из этой папки:
    py -m pip install marimo kagglehub pandas numpy matplotlib
    py -m marimo run poster_marimo.py

Положите enhanced_poster_design_dataset.csv и poster_style.py рядом с файлом.
Если CSV отсутствует, приложение попробует скачать его через KaggleHub.
"""
import marimo

app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Генератор случайных величин для постеров

    **Содержание:** [1. Модель](#model) · [2. Настройки](#controls) ·
    [3. Результаты](#results) · [4. Вывод](#conclusion)

    <a id="model"></a>
    ## 1. Модель

    Для случайно выбранного постера:
    $X=|C|$ — число записей цветов, $Y=|\operatorname{set}(C)|$ —
    число различных цветов. Поэтому для каждого постера $Y\\leq X$.

    - **Совместная модель:** выбирается целая пара $(X,Y)$ из одной строки датасета.
    - **Независимая модель:** $X$ и $Y$ выбираются отдельно из эмпирических
      маргинальных распределений. Это модель нулевой гипотезы $H_0:X\\perp Y$.
    """)
    return (mo,)


@app.cell
def _(mo):
    import ast
    from pathlib import Path

    import pandas as pd

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "enhanced_poster_design_dataset.csv"

    if not csv_path.is_file():
        import kagglehub

        downloaded = Path(
            kagglehub.dataset_download(
                "ziya07/enhanced-poster-design-dataset",
                path="enhanced_poster_design_dataset.csv",
            )
        )
        csv_path = (
            downloaded
            if downloaded.is_file()
            else downloaded / "enhanced_poster_design_dataset.csv"
        )

    if not csv_path.is_file():
        raise FileNotFoundError(
            "CSV не найден. Положите enhanced_poster_design_dataset.csv "
            "рядом с poster_marimo.py."
        )

    raw = pd.read_csv(csv_path, usecols=["Colors"])
    parsed = raw["Colors"].map(ast.literal_eval)
    pairs = pd.DataFrame({
        "X": parsed.map(len),
        "Y": parsed.map(lambda colors: len(set(colors))),
    })

    if pairs.empty:
        raise ValueError("Датасет не содержит строк.")
    if (pairs["Y"] > pairs["X"]).any():
        raise ValueError("В данных обнаружена строка с Y > X.")

    mo.md(f"**Данные загружены:** {len(pairs):,} постеров.")
    return (pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="controls"></a>
    ## 2. Настройки генератора
    """)
    n = mo.ui.slider(
        start=100, stop=10000, step=100, value=1000,
        label="Количество испытаний",
    )
    seed = mo.ui.number(
        start=0, stop=999999, step=1, value=42, label="Seed",
    )
    model = mo.ui.radio(
        options=[
            "Совместная (по датасету)",
            "Независимая (H₀)",
        ],
        value="Совместная (по датасету)",
        label="Модель генерации",
    )
    mo.vstack([mo.hstack([n, seed]), model])
    return model, n, seed


@app.cell
def _(model, n, pairs, seed):
    # RNG и samples намеренно находятся в одной ячейке:
    # локальные переменные marimo-ячеек не разделяются автоматически.
    import numpy as _np

    rng = _np.random.default_rng(int(seed.value))
    k = int(n.value)

    if model.value == "Совместная (по датасету)":
        samples = pairs.to_numpy()[
            rng.integers(0, len(pairs), size=k)
        ]
    else:
        samples = _np.column_stack([
            rng.choice(pairs["X"].to_numpy(), size=k),
            rng.choice(pairs["Y"].to_numpy(), size=k),
        ])

    bad = int(_np.count_nonzero(samples[:, 1] > samples[:, 0]))
    return bad, k, samples


@app.cell(hide_code=True)
def _(bad, k, mo, pairs, samples):
    import numpy as _np
    import matplotlib.pyplot as plt

    from poster_style import plot_simulation

    fig = plot_simulation(
        samples,
        sorted(pairs["X"].unique()),
        sorted(pairs["Y"].unique()),
    )
    chart = mo.as_html(fig)
    plt.close(fig)

    impossible_probability = sum(
        (pairs["X"] == x).mean() * (pairs["Y"] == y).mean()
        for x in pairs["X"].unique()
        for y in pairs["Y"].unique()
        if y > x
    )

    mo.vstack([
        mo.md(r"""<a id="results"></a>
        ## 3. Результаты"""),
        mo.md(
            f"**Испытаний:** {k:,} · "
            f"$\\overline{{X}}={_np.mean(samples[:, 0]):.3f}$ · "
            f"$\\overline{{Y}}={_np.mean(samples[:, 1]):.3f}$ · "
            f"**Невозможных пар $Y>X$:** {bad} ({bad / k:.1%})"
        ),
        chart,
        mo.md(r"""<a id="conclusion"></a>
        ## 4. Вывод"""),
        mo.md(
            "Совместная модель сохраняет ограничение $Y\\leq X$. "
            f"При независимом выборе теоретическая вероятность нарушения "
            f"ограничения равна **{impossible_probability:.1%}**. "
            "Переключите модель и измените число испытаний или seed, "
            "чтобы сравнить результаты."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
