from pathlib import Path

import pandas as pd
from tabulate import tabulate


def generate_table(
    data: pd.DataFrame,
    path: Path,
) -> None:
    table = tabulate(
        data,
        headers='keys',
        tablefmt='latex_raw',
        floatfmt='.2f',
        showindex=False,
    )

    with open(path, 'w') as f:
        f.write(table)
