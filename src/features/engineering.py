"""Features por lectura sobre los contadores acumulados de SCANIA Component X.

Las 105 columnas operacionales son contadores y bins de histograma ACUMULADOS desde
el inicio de la vida del vehiculo (100% no decrecientes). El valor crudo mide mas
la edad del camion que su estado, asi que se transforma en:

* ``life_rate_*``   : uso medio por unidad de tiempo desde el inicio (``x / time_step``).
* ``win_rate_*``    : uso por unidad de tiempo en las ultimas ``WINDOW`` lecturas.
* ``life_frac_*``   : en variables de histograma, fraccion de cada bin sobre su total historico.
* ``recent_frac_*`` : lo mismo pero solo con el incremento de la ventana reciente.
* ``rel_rate_*``    : ``win_rate / life_rate``, intensidad reciente relativa a la de toda la vida.
* ``n_readouts``, ``dt_prev``: cuantas lecturas hay y hace cuanto fue la anterior.

**Escala distinta por conjunto.** Cada variable viene multiplicada por un factor propio que ademas cambia
entre train, validacion y test (la variable 272 vale ~0,15 veces en validacion lo que vale en train; la 397,
~3,8 veces). Las tasas ``life_rate`` y ``win_rate`` heredan ese factor; las fracciones de histograma y
``rel_rate`` no, porque el factor se cancela dentro de la variable. ``scale_invariant`` devuelve ese subconjunto.

Cada feature de la lectura ``t`` usa unicamente lecturas ``<= t`` del mismo vehiculo, de modo
que el calculo sobre la historia completa de train coincide con el que veria un modelo en
produccion que solo conoce el pasado (hay un test que lo verifica por truncamiento).
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

ID_COLUMNS = ["vehicle_id", "time_step"]
WINDOW = 5
EPS = 1e-9
INVARIANT_PREFIXES = ("life_frac_", "recent_frac_", "rel_rate_")
TIME_FEATURES = ["time_step", "n_readouts", "dt_prev"]


def variable_groups(columns: list[str]) -> dict[str, list[str]]:
    """Agrupa columnas ``<variable>_<bin>`` por variable, ordenadas por bin."""
    groups: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        if col in ID_COLUMNS:
            continue
        groups[col.rsplit("_", 1)[0]].append(col)
    return {k: sorted(v, key=lambda c: int(c.rsplit("_", 1)[1])) for k, v in groups.items()}


def feature_names(columns: list[str]) -> list[str]:
    names = ["time_step", "n_readouts", "dt_prev"]
    groups = variable_groups(columns)
    for cols in groups.values():
        for c in cols:
            names += [f"life_rate_{c}", f"win_rate_{c}", f"rel_rate_{c}"]
        if len(cols) > 1:
            for c in cols:
                names += [f"life_frac_{c}", f"recent_frac_{c}"]
    return names


def engineer_features(readouts: pl.DataFrame, window: int = WINDOW) -> pl.DataFrame:
    """Devuelve ``vehicle_id``, ``time_step`` y las features, una fila por lectura."""
    cols = [c for c in readouts.columns if c not in ID_COLUMNS]
    groups = variable_groups(cols)
    vid = "vehicle_id"

    df = readouts.sort(ID_COLUMNS)
    # Contadores acumulados: un nulo se rellena con la ultima lectura conocida del vehiculo.
    df = df.with_columns([pl.col(c).forward_fill().over(vid) for c in cols])

    t = pl.col("time_step")
    t_back = pl.coalesce([t.shift(window).over(vid), t.first().over(vid)])
    span = t - t_back

    exprs: list[pl.Expr] = [
        t,
        pl.int_range(1, pl.len() + 1).over(vid).alias("n_readouts"),
        (t - t.shift(1).over(vid)).alias("dt_prev"),
    ]
    for var, var_cols in groups.items():
        for c in var_cols:
            x = pl.col(c)
            x_back = pl.coalesce([x.shift(window).over(vid), x.first().over(vid)])
            life = x / t.clip(lower_bound=1.0)
            win = pl.when(span > EPS).then((x - x_back) / span).otherwise(None)
            exprs.append(life.alias(f"life_rate_{c}"))
            exprs.append(win.alias(f"win_rate_{c}"))
            exprs.append(pl.when(life.abs() > EPS).then(win / life).otherwise(None).alias(f"rel_rate_{c}"))
        if len(var_cols) > 1:
            total = pl.sum_horizontal(var_cols)
            recent_total = pl.sum_horizontal(
                [pl.col(c) - pl.coalesce([pl.col(c).shift(window).over(vid), pl.col(c).first().over(vid)]) for c in var_cols]
            )
            for c in var_cols:
                x = pl.col(c)
                x_back = pl.coalesce([x.shift(window).over(vid), x.first().over(vid)])
                exprs.append(pl.when(total > EPS).then(x / total).otherwise(None).alias(f"life_frac_{c}"))
                exprs.append(
                    pl.when(recent_total > EPS).then((x - x_back) / recent_total).otherwise(None).alias(f"recent_frac_{c}")
                )

    out = df.select([pl.col(vid), *exprs])
    # time_step queda en Float64: es la llave de union con las etiquetas y Float32 la corrompe.
    float_cols = [c for c in out.columns if c not in (vid, "n_readouts", "time_step")]
    return out.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])


def scale_invariant(columns: list[str]) -> list[str]:
    """Features que no cambian si una variable se multiplica por una constante: tiempo, fracciones y ``rel_rate``."""
    return [c for c in columns if c in TIME_FEATURES or c.startswith(INVARIANT_PREFIXES)]


def last_readout_per_vehicle(features: pl.DataFrame) -> pl.DataFrame:
    """Fila de cada vehiculo en su ultima lectura (el punto de decision en validacion/test)."""
    return features.sort(ID_COLUMNS).group_by("vehicle_id", maintain_order=True).last()
