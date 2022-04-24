import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib
import matplotlib.animation as mpl_animation
import numpy as np
import seaborn as sns
from IPython.display import HTML, display
from matplotlib import gridspec
from matplotlib import pyplot as plt
from progressbar import progressbar
from typing_extensions import Literal

from generators import LXM

sns.set()


np.random.seed(117)
random.seed(117)
matplotlib.rcParams["animation.embed_limit"] = 2**128

POSIBLES_MOVIMIENTOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Shuffler:  # pylint: disable=too-few-public-methods
    """Shuffle using LXM."""

    def __init__(self):
        self.lxm = LXM()

    def __call__(self, list_like):
        for idx in range(len(list_like) - 1, 0, -1):
            sel = int(np.floor(self.lxm.generar() * (idx + 1)))
            list_like[idx], list_like[sel] = list_like[sel], list_like[idx]


@dataclass
class Caminante:
    """Random walk class."""

    identificador: int
    posicion: Tuple[int, int]
    tipo: Literal["A", "B", "C"]
    max_pos: int

    def __post_init__(self):
        self.movimientos = []

    def turno_de_moverse(self, turno):
        """Me muevo en turno t."""
        return (
            (self.tipo == "A")
            or (self.tipo == "B" and (turno % 2) == 0)
            or (self.tipo == "C" and (turno % 4) == 0)
        )

    def _out_of_bounds(self, _x: int, _y: int) -> bool:
        """Chequear que una posicion este dentro de los limites permitidos.

        Recibe
        ------
        _x: primer coordenada
        _y: segunda coordenada

        Retorna
        -------
        True si esta fuera del tablero, False en caso contrario.
        """
        if min(_x, _y) < 0:
            return True
        if max(_x, _y) >= self.max_pos:
            return True
        return False

    def mover(self, turno: int, shuffler: Shuffler, nueva_matriz: np.ndarray) -> None:
        """Mueve al caminante, ejecutando un paso de la simulacion para el mismo.

        Recibe
        ------
        t: El numero de iteracion
        nueva_matriz: Matriz donde se ubican los caminantes en el paso actual
        c: KDTree con los caminantes en el paso anterior

        Retorna
        -------
        None
        """
        if self.turno_de_moverse(turno):
            idxs = list(np.arange(0, len(POSIBLES_MOVIMIENTOS), 1))
            shuffler(idxs)

            attempts_to_move = 0
            while attempts_to_move < 2 and len(idxs) > 0:
                mov_x, mov_y = POSIBLES_MOVIMIENTOS[idxs.pop()]
                pos_x, pos_y = self.posicion
                new_pos_x, new_pos_y = (mov_x + pos_x, mov_y + pos_y)

                if self._out_of_bounds(new_pos_x, new_pos_y):
                    continue

                if nueva_matriz[new_pos_x, new_pos_y] != 0:
                    attempts_to_move += 1
                else:
                    self.movimientos.append(self.posicion)
                    self.posicion = (new_pos_x, new_pos_y)
                    break

        nueva_matriz[self.posicion] = self.identificador


def simular(
    lado_grilla: int = 100,
    caminantes: int = 50,
    iteraciones: int = 500,
):
    """Simular la pandemia.

    Recibe
    ------
    n: cantidad de individuos
    iteraciones: cuantas iteraciones correr de la simulacion
    """
    shuffler = Shuffler()

    # inicializacion
    grilla = np.zeros((lado_grilla, lado_grilla), dtype=int)

    # generamos uniformemente las posiciones iniciales
    xs, ys = np.divmod(
        np.random.choice(
            np.arange(0, lado_grilla**2, 1), size=caminantes, replace=False
        ),
        lado_grilla,
    )
    posiciones = np.c_[xs, ys]

    # seleccionamos aleatoriamente el tipo de cada caminante
    tipos = np.repeat(["A"], caminantes)
    tipos[int(0.7 * caminantes) :] = "B"
    tipos[int(0.95 * caminantes) :] = "C"
    shuffler(tipos)

    caminantes: List[Caminante] = []

    # inicializamos los caminantes
    for i, ((pos_x, pos_y), tipo) in enumerate(zip(posiciones, tipos), start=1):
        caminante = Caminante(i, (pos_x, pos_y), tipo, lado_grilla)
        grilla[pos_x, pos_y] = i
        caminantes.append(caminante)

    snapshots_path = []

    for iternum in progressbar(range(1, iteraciones + 1), max_value=iteraciones):
        shuffler(caminantes)

        nueva_grilla = np.zeros((lado_grilla, lado_grilla), dtype=int)

        for caminante in caminantes:
            # como cada posicion puede ser ocupada por un solo caminante
            # los movemos de a uno para evitar colisiones
            # con el shuffle anterior evitamos que tengan siempre prioridad los mismos
            caminante.mover(
                iternum,
                shuffler,
                nueva_grilla,
            )

        snapshots_path.append(grilla)
        grilla = nueva_grilla

    return snapshots_path, caminantes


def animacion(snapshots_path, caminantes, guardar=None, lado_grilla: int = 100):
    iteraciones = len(snapshots_path)

    dict_tipos = {"A": [], "B": [], "C": []}
    for caminante in caminantes:
        dict_tipos[caminante.tipo].append(caminante.identificador)

    fig = plt.figure(figsize=(10, 5), dpi=100)

    gs = gridspec.GridSpec(2, 2, width_ratios=[8, 12])
    ax1 = fig.add_subplot(gs[0:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    final_colors = matplotlib.colors.ListedColormap(
        ["xkcd:white", "xkcd:grey", "xkcd:red", "xkcd:purple", "xkcd:cyan"]
    )
    bg = matplotlib.colors.ListedColormap(["xkcd:blue", "xkcd:green"])

    left_A_history = []
    left_B_history = []
    left_C_history = []
    left_history = []

    right_A_history = []
    right_B_history = []
    right_C_history = []
    right_history = []

    def plot_camino(data):
        for ax in (ax1, ax2, ax3):
            ax.clear()

        idx, frame = data

        this_frame = np.zeros(frame.shape)
        this_frame[np.isin(frame, dict_tipos["A"])] = 2
        this_frame[np.isin(frame, dict_tipos["B"])] = 3
        this_frame[np.isin(frame, dict_tipos["C"])] = 4

        alphas = np.zeros(this_frame.shape)
        alphas[np.where(this_frame > 1)] = 1

        bg_frame = np.hstack(
            (
                np.zeros((lado_grilla, lado_grilla // 2), dtype=int),
                np.ones((lado_grilla, lado_grilla // 2), dtype=int),
            )
        )
        ax1.imshow(bg_frame, cmap=bg, aspect="equal", interpolation="none", alpha=0.2)

        ax1.imshow(
            this_frame,
            cmap=final_colors,
            aspect="equal",
            interpolation="none",
            alpha=alphas,
        )
        ax1.grid(False)
        ax1.set(xticklabels=[])
        ax1.set_title(f"Posición de las partículas en t={idx}")

        left_A = np.isin(frame[:, : lado_grilla // 2], dict_tipos["A"]).sum()
        left_B = np.isin(frame[:, : lado_grilla // 2], dict_tipos["B"]).sum()
        left_C = np.isin(frame[:, : lado_grilla // 2], dict_tipos["C"]).sum()
        left_total = left_A + left_B + left_C
        left_A_history.append(left_A / frame[:, : lado_grilla // 2].size)
        left_B_history.append(left_B / frame[:, : lado_grilla // 2].size)
        left_C_history.append(left_C / frame[:, : lado_grilla // 2].size)
        left_history.append(left_total / frame[:, : lado_grilla // 2].size)
        ax2.plot(left_A_history, label="A", color="xkcd:red")
        ax2.plot(left_B_history, label="B", color="xkcd:purple")
        ax2.plot(left_C_history, label="C", color="xkcd:cyan")
        ax2.plot(left_history, label="Total")
        ax2.legend()

        ax2.set_xlabel("t")
        ax2.set_ylabel("proporcion")
        ax2.set_title("proporcion de caminantes por tipo en la mitad azul")

        right_A = np.isin(frame[:, lado_grilla // 2 :], dict_tipos["A"]).sum()
        right_B = np.isin(frame[:, lado_grilla // 2 :], dict_tipos["B"]).sum()
        right_C = np.isin(frame[:, lado_grilla // 2 :], dict_tipos["C"]).sum()
        right_total = left_A + left_B + left_C
        right_A_history.append(right_A / frame[:, lado_grilla // 2 :].size)
        right_B_history.append(right_B / frame[:, lado_grilla // 2 :].size)
        right_C_history.append(right_C / frame[:, lado_grilla // 2 :].size)
        right_history.append(right_total / frame[:, lado_grilla // 2 :].size)
        ax3.plot(right_A_history, label="A", color="xkcd:red")
        ax3.plot(right_B_history, label="B", color="xkcd:purple")
        ax3.plot(right_C_history, label="C", color="xkcd:cyan")
        ax3.plot(right_history, label="Total")
        ax3.legend()

        ax3.set_xlabel("t")
        ax3.set_ylabel("proporcion")
        ax3.set_title("proporcion de caminantes por tipo en la mitad verde")

        ax1.grid(False)
        ax1.set(xticklabels=[])
        ax1.set(yticklabels=[])

        ax2.set_xlim(left=0, right=iteraciones)
        ax3.set_xlim(left=0, right=iteraciones)
        fig.suptitle(f"Tiempo t={idx}")
        fig.tight_layout()

    anim = mpl_animation.FuncAnimation(
        fig,
        plot_camino,
        frames=progressbar(enumerate(snapshots_path, 0), max_value=len(snapshots_path)),
        interval=50,
        save_count=len(snapshots_path),
    )

    if guardar is not None:
        print("Guardando")
        anim.save(f"{guardar}.mp4")
        print("Guardado")
    else:
        plt.close()
        matplotlib.rc("animation", html="jshtml")
        display(HTML(anim.to_jshtml()))
