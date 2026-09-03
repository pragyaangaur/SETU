"""Solve for geomagnetically induced current in the direct current network.

The method is the nodal admittance method of Lehtinen and Pirjola (1985). The
transmission network is treated as a resistive circuit at direct current, driven by
voltage sources in the transmission lines that come from the induced geoelectric
field. The unknowns are the currents flowing from each substation into the earth,
because that is what flows up the transformer neutrals and saturates the cores.

The governing relation is

    I = (U + Y Z) inverse, applied to J

where ``J`` is the vector of currents that would flow into the earth if every
substation were perfectly grounded, ``Y`` is the network admittance matrix seen
between the substation nodes, ``Z`` is the diagonal matrix of earthing
resistances, and ``U`` is the identity. The matrix inverse is what couples the
whole network together, so a current at one substation depends on the geology and
the topology everywhere else.
"""

from dataclasses import dataclass

import numpy as np

from setu.grid.network import Network, TRANSFORMER_TYPES


@dataclass
class GICResult:
    """Currents and their immediate electrical consequence at one instant.

    Attributes:
        neutral_current: Total current into the earth at each substation, in
            ampere. The sign says which way it flows and the magnitude is what
            matters for saturation.
        per_phase_per_unit: Current carried by one phase of one transformer, in
            ampere. This is the number that transformer limits are written
            against.
        reactive_loss_mvar: Extra reactive power drawn by the transformers at each
            substation, in megavar.
        hotspot_rise_k: Steady state rise of the winding hot spot temperature above
            its normal value, in kelvin.
        codes: Substation codes in the same order as the arrays.
    """

    neutral_current: np.ndarray
    per_phase_per_unit: np.ndarray
    reactive_loss_mvar: np.ndarray
    hotspot_rise_k: np.ndarray
    codes: list


class GICSolver:
    """Builds the constant circuit matrices once, then solves many time steps fast.

    The admittance matrix and its inverse factor do not change from minute to
    minute, so they are built in the constructor. Only the driving vector changes,
    which makes a long storm replay cheap.
    """

    def __init__(self, network: Network, blocked: set = None):
        """Assemble the circuit.

        Args:
            network: The substation and line model.
            blocked: Codes of substations fitted with a neutral blocking device.
                A blocked substation is treated as an open circuit to earth, which
                is what a series capacitor in the neutral does at direct current.
        """
        self.net = network
        self.blocked = set(blocked or ())
        n = network.n
        self.codes = [s.code for s in network.substations]

        # Effective resistance from a substation node into the earth. A very large
        # value stands in for a blocked neutral, which keeps the matrix well formed
        # and avoids a special case in the solver.
        self.earth_r = np.array(
            [1.0e6 if s.code in self.blocked else s.ground_resistance
             for s in network.substations],
            dtype=float,
        )

        # Effective per phase winding resistance at each node, with the parallel
        # transformer units and the three phases already combined.
        self.wind_r = np.array(
            [s.winding_resistance / (3.0 * s.transformer_count)
             for s in network.substations],
            dtype=float,
        )

        self.branch_conductance = []
        self.branch_nodes = []
        for ln in network.lines:
            i = network.index[ln.frm]
            j = network.index[ln.to]
            r = ln.resistance / (3.0 * ln.circuits) + self.wind_r[i] + self.wind_r[j]
            self.branch_conductance.append(1.0 / r)
            self.branch_nodes.append((i, j))
        self.branch_conductance = np.array(self.branch_conductance)

        y = np.zeros((n, n))
        for g, (i, j) in zip(self.branch_conductance, self.branch_nodes):
            y[i, i] += g
            y[j, j] += g
            y[i, j] -= g
            y[j, i] -= g
        self.y = y

        self.system = np.eye(n) + y @ np.diag(self.earth_r)

        self.k_var = np.array(
            [TRANSFORMER_TYPES[s.transformer_type]["k_var"]
             for s in network.substations]
        )
        self.hotspot_k = np.array(
            [TRANSFORMER_TYPES[s.transformer_type]["hotspot_k"]
             for s in network.substations]
        )
        self.unit_count = np.array(
            [s.transformer_count for s in network.substations], dtype=float
        )

    def driving_vector(self, ex_by_model: dict, ey_by_model: dict) -> np.ndarray:
        """Currents into a perfectly grounded network, one entry per substation.

        Each line carries an induced voltage equal to the dot product of the local
        electric field with the line vector. The field is taken as the average of
        the two end point ground models, which is a simple way to represent a line
        that crosses a conductivity boundary.

        Args:
            ex_by_model: Northward field in volt per kilometre, keyed by Earth
                model name, one scalar per model.
            ey_by_model: Eastward field in volt per kilometre, same keys.
        """
        j = np.zeros(self.net.n)
        for ln, g, (i, k) in zip(self.net.lines, self.branch_conductance,
                                 self.branch_nodes):
            model_a = self.net.substations[i].earth_model
            model_b = self.net.substations[k].earth_model
            ex = 0.5 * (ex_by_model[model_a] + ex_by_model[model_b])
            ey = 0.5 * (ey_by_model[model_a] + ey_by_model[model_b])
            dx, dy = self.net.line_vector_km(ln)
            voltage = ex * dx + ey * dy
            current = voltage * g
            j[k] += current
            j[i] -= current
        return j

    def solve(self, ex_by_model: dict, ey_by_model: dict) -> GICResult:
        """Solve one instant and return currents and their consequence."""
        j = self.driving_vector(ex_by_model, ey_by_model)
        i_earth = np.linalg.solve(self.system, j)
        per_phase = np.abs(i_earth) / (3.0 * self.unit_count)
        return GICResult(
            neutral_current=i_earth,
            per_phase_per_unit=per_phase,
            reactive_loss_mvar=self.k_var * per_phase * 3.0 * self.unit_count,
            hotspot_rise_k=self.hotspot_k * per_phase,
            codes=self.codes,
        )

    def solve_series(self, ex_series: dict, ey_series: dict) -> dict:
        """Solve a whole time series and return arrays of shape (time, substation).

        Args:
            ex_series: Northward field per Earth model, each an array over time.
            ey_series: Eastward field per Earth model, same lengths.
        """
        names = list(ex_series)
        length = len(ex_series[names[0]])
        neutral = np.zeros((length, self.net.n))
        for t in range(length):
            ex = {k: float(v[t]) for k, v in ex_series.items()}
            ey = {k: float(v[t]) for k, v in ey_series.items()}
            neutral[t] = np.linalg.solve(self.system, self.driving_vector(ex, ey))
        per_phase = np.abs(neutral) / (3.0 * self.unit_count)[None, :]
        return {
            "codes": self.codes,
            "neutral_current": neutral,
            "per_phase_per_unit": per_phase,
            "reactive_loss_mvar": per_phase * (self.k_var * 3.0 * self.unit_count)[None, :],
            "hotspot_rise_k": per_phase * self.hotspot_k[None, :],
        }


def uniform_field_case(solver: GICSolver, ex_v_per_km: float,
                       ey_v_per_km: float) -> GICResult:
    """Apply one uniform electric field everywhere, ignoring the ground models.

    This is the benchmark case used across the geomagnetically induced current
    literature, usually at one volt per kilometre. It is here so that the solver
    can be checked against a published result and so that network topology can be
    studied on its own, with the geology held constant.
    """
    models = {s.earth_model for s in solver.net.substations}
    return solver.solve({m: ex_v_per_km for m in models},
                        {m: ey_v_per_km for m in models})
