"""Punto de entrada unico del proyecto.

    python -m src.cli download            # los cuatro datasets (~2,1 GB en data/raw)
    python -m src.cli scania              # SCANIA Component X: diagnostico de escala, 7 clasificadores, supervivencia
    python -m src.cli aps                 # APS Failure at Scania Trucks: 6 modelos + ensamble, costo del reto
    python -m src.cli cmapss              # NASA C-MAPSS FD001-FD004: 6 modelos de vida util restante + ensamble
    python -m src.cli metropt             # MetroPT-3: 5 detectores de anomalias contra el registro de fallas
    python -m src.cli figures             # figuras en ingles y espanol (reports/figures/{en,es})
    python -m src.cli site                # pagina de GitHub Pages (docs/)
    python -m src.cli all                 # todo lo anterior, en orden
"""
from __future__ import annotations

import sys
import time


def download() -> None:
    from src.data import aps_loader, cmapss_loader, metropt_loader, scania_loader

    scania_loader.download()
    aps_loader.download()
    cmapss_loader.download_all()
    metropt_loader.download()


def scania() -> None:
    from src.models import train_pipeline

    train_pipeline.run()


def aps() -> None:
    from src.models import aps_pipeline

    aps_pipeline.run()


def cmapss() -> None:
    from src.models import rul_benchmark

    sys.argv = sys.argv[:1]
    rul_benchmark.main()


def metropt() -> None:
    from src.models import metropt_pipeline

    metropt_pipeline.run()


def figures() -> None:
    from src.models import make_figures

    for p in make_figures.make_all():
        print(p)


def site() -> None:
    from src import site as s

    print(s.build_site())


STEPS = {"download": download, "scania": scania, "aps": aps, "cmapss": cmapss, "metropt": metropt,
         "figures": figures, "site": site}


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in [*STEPS, "all"]:
        print(__doc__)
        raise SystemExit(2)
    for name in STEPS if argv[0] == "all" else [argv[0]]:
        t0 = time.time()
        print(f"== {name}", flush=True)
        STEPS[name]()
        print(f"== {name} listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
