"""Generadores aleatorios."""

import abc
from typing import Optional

import numpy as np

# pylint: disable=invalid-name


class RandomGenerator(abc.ABC):
    """Clase abstracta de un generador de números aleatorios."""

    @abc.abstractmethod
    def generar(self):
        """Método para generar un número aleatorio."""
        raise NotImplementedError()

    def generar_vector(self, cantidad: int = 100):
        """Método para generar un vector de números aleatorio."""
        return np.array([self.generar() for _ in range(cantidad)])


class GCL(RandomGenerator):
    """Generador Congruencial Lineal."""

    def __init__(
        self,
        semilla: np.uint64 = np.uint64(np.mean([94551, 98153, 103855, 103735])),
        multiplicador: np.uint64 = np.uint64(1013904223),
        incremento: np.uint64 = np.uint64(1664525),
        modulo: Optional[np.uint64] = None,
        uniforme: bool = False,
    ):
        """Generador Congruencial Lineal.

        Parametros
        ----------
            semilla: valor inicial de semilla
            mult: valor de multiplicador
            incremento: valor de incremento
            modulo: valor de modulo
            uniforme: si debe ser uniforme el resultado
        """
        self.Xi = semilla
        self.multiplicador = multiplicador
        self.incremento = incremento
        self.modulo = modulo
        self.uniforme = uniforme

    def generar(self):
        """Genera un número al azar.

        Parametros
        ----------
            n: la cantidad de numeros a generar

        Retorna
        -------
            Número aleatorio.
        """
        result = self.Xi
        if self.uniforme:
            result /= self.modulo
        self.Xi = (self.multiplicador * self.Xi) + self.incremento
        if self.modulo is not None:
            self.Xi %= self.modulo
        return result


class XBG(RandomGenerator):
    """XOR based random number generator."""

    def __init__(
        self, x0: np.uint64 = np.uint64(119823174), x1: np.uint64 = np.uint64(1234987),
    ):
        self.x0, self.x1 = x0, x1
        assert not (self.x0 == 0 and self.x1 == 0)

    def generar(self):
        def rotate_left(n, d):
            """Rotate n by d bits."""
            return np.bitwise_or(
                np.left_shift(n, np.uint64(d)),
                np.right_shift(n, np.uint64(n.nbytes * 8 - d)),
            )

        result = self.x0
        q0, q1 = self.x0, self.x1
        q1 ^= q0
        q0 = rotate_left(q0, np.uint64(24))
        q0 = q0 ^ q1 ^ (q1 << np.uint64(16))
        q1 = rotate_left(q1, np.uint64(37))
        self.x0, self.x1 = q0, q1

        return result


class LXM(RandomGenerator):
    """LXM random number generator."""

    def __init__(
        self,
        M: np.uint64 = np.uint64(0xD1342543DE82EF95),
        w: np.uint64 = np.uint64(32),
        a: np.uint64 = np.uint64(1664525),
    ):
        if (a % 2) != 1:
            raise ValueError("a debe ser impar")
        self.w = w
        self.LCG = GCL(multiplicador=M, incremento=a)
        self.XBG = XBG()

    def generar(self):
        """Generar un número aleatorio."""

        def high_order_bits(s, w):
            """Get w highest order bits from s."""
            return np.right_shift(s, np.uint64(s.nbytes * 8 - w))

        s = self.LCG.generar()
        t = self.XBG.generar()

        z = self.mix(
            self.combine(high_order_bits(s, self.w), high_order_bits(t, self.w))
        )
        return z

    def combine(self, *args):  # pylint:disable=no-self-use
        """Combination operation for the XBG and LCG numbers."""
        z = np.sum(args)
        return z

    def mix(self, z):  # pylint:disable=no-self-use
        """Mix operation."""
        z = np.multiply(
            np.bitwise_xor(z, np.right_shift(z, np.uint64(32))),
            np.uint64(0xDABA0B6EB09322E3),
        )
        z = np.multiply(
            np.bitwise_xor(z, np.right_shift(z, np.uint64(32))),
            np.uint64(0xDABA0B6EB09322E3),
        )
        z = np.bitwise_xor(z, np.right_shift(z, np.uint64(32)))
        return z
