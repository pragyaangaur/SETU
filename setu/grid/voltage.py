"""Voltage consequence of the reactive power that saturated transformers absorb.

A transformer carrying a direct current bias draws extra reactive power, and
reactive power is local. It cannot be moved far across a network, so the voltage
falls near the transformer that is absorbing it. When enough transformers absorb at
once and the local reserve runs out, the voltage collapses. That is the mechanism
that took down the Hydro-Quebec system in 1989, and it is why this project does not
stop at reporting a current in ampere.

The calculation here is a linearised reactive power to voltage sensitivity, built
from the susceptance matrix of the network. It is the standard fast screening tool
in power system practice and it is the right level of detail for a system whose
job is to rank sites and to compare actions. It is not a full alternating current
power flow solution, and it will understate the size of the drop once the network
is close to collapse, because the real relationship curves downward there.
"""

import numpy as np

from setu.grid.network import Network

# Series reactance per kilometre of line, in per unit on a 100 MVA base at each
# voltage level. These come from ordinary Indian transmission conductor reactance
# of about 0.33 ohm per kilometre at 400 kV, divided by the base impedance of the
# level. The susceptance of a corridor is the inverse of its total reactance, so a
# long line couples its two ends together weakly and a short one couples them
# tightly.
REACTANCE_PU_PER_KM = {400: 2.06e-4, 220: 8.26e-4, 132: 2.41e-3}

# Reactive droop of the local voltage support. A five percent droop means a source
# gives its full reactive output for a five percent voltage drop, which is normal
# for an Indian generator automatic voltage regulator.
VOLTAGE_DROOP_PU = 0.05
BASE_MVA = 100.0

# Reactive power reserve available at each site before the storm, in megavar.
# Generation sites carry more than pure switching stations, because a machine can
# be run at leading or lagging power factor while a switching station only has its
# capacitor banks.
DEFAULT_RESERVE_MVAR = {
    "BWNC": 260.0, "PLTN": 220.0, "KMNG": 180.0, "RNGN": 120.0, "LKTK": 60.0,
    "SLKT": 190.0, "BNGR": 300.0, "ALPD": 240.0, "AZRA": 150.0, "MISA": 130.0,
}
FALLBACK_RESERVE_MVAR = 70.0


class VoltageModel:
    """Reactive power to voltage sensitivity for the regional network."""

    def __init__(self, network: Network, reserves: dict = None):
        self.net = network
        self.reserve = np.array(
            [(reserves or DEFAULT_RESERVE_MVAR).get(s.code, FALLBACK_RESERVE_MVAR)
             for s in network.substations], dtype=float)

        n = network.n
        b = np.zeros((n, n))
        for line in network.lines:
            i, j = network.index[line.frm], network.index[line.to]
            length = max(network.line_length_km(line), 1.0)
            reactance = REACTANCE_PU_PER_KM.get(line.kv, 8.26e-4) * length
            susceptance = line.circuits / reactance
            b[i, i] += susceptance
            b[j, j] += susceptance
            b[i, j] -= susceptance
            b[j, i] -= susceptance

        # A susceptance matrix built only from lines is singular, because a uniform
        # shift of every voltage costs nothing. Adding the local voltage support at
        # each bus removes that freedom, and it also carries the real effect that a
        # bus with more reactive reserve behind it is harder to pull down.
        b += np.diag(self.reserve / BASE_MVA / VOLTAGE_DROOP_PU)
        self.sensitivity = np.linalg.inv(b)

    def voltage_deviation(self, reactive_loss_mvar: np.ndarray) -> np.ndarray:
        """Per unit voltage change caused by an extra reactive demand.

        The result is negative, because absorbing reactive power pulls a bus down.
        A deviation of minus 0.1 per unit means a ten percent drop, which is past
        the point where under voltage protection starts to operate in India.
        """
        return -self.sensitivity @ (np.asarray(reactive_loss_mvar) / BASE_MVA)

    def reserve_margin(self, reactive_loss_mvar: np.ndarray) -> np.ndarray:
        """Fraction of the local reactive reserve still unused, per substation.

        A value at or below zero means the site has run out of reserve and the
        voltage there is being held up by its neighbours alone.
        """
        return 1.0 - np.asarray(reactive_loss_mvar) / self.reserve

    def assess(self, reactive_loss_mvar: np.ndarray,
               limit_pu: float = 0.10) -> dict:
        """Summarise the voltage state of the region under one loading."""
        deviation = self.voltage_deviation(reactive_loss_mvar)
        margin = self.reserve_margin(reactive_loss_mvar)
        exhausted = margin <= 0.0
        violating = deviation <= -limit_pu
        return {
            "voltage_deviation_pu": deviation,
            "reserve_margin": margin,
            "worst_deviation_pu": float(deviation.min()),
            "buses_below_limit": [self.net.substations[i].code
                                  for i in np.where(violating)[0]],
            "reserve_exhausted": [self.net.substations[i].code
                                  for i in np.where(exhausted)[0]],
            "total_reactive_mvar": float(np.sum(reactive_loss_mvar)),
        }

    def load_at_risk_mw(self, reactive_loss_mvar: np.ndarray,
                        limit_pu: float = 0.10,
                        regional_load_mw: float = 3200.0) -> float:
        """Load exposed to disconnection, in megawatt.

        Load is attributed to each substation by its criticality weight. A site is
        counted as exposed for either of two reasons. The first is that its voltage
        has gone past the limit, and the exposure grows as the drop deepens because
        under voltage protection acts faster at a deeper drop. The second is that
        the site has used up its own reactive reserve, which means it is being held
        up by its neighbours and has nothing left for the next disturbance. Reserve
        runs out before the voltage limit is reached, so this is the term that
        gives useful early warning.

        The regional load default of 3200 MW is the order of the North Eastern
        Regional peak demand. This is a screening estimate and not a settlement
        grade figure.
        """
        deviation = self.voltage_deviation(reactive_loss_mvar)
        weights = np.array([s.criticality for s in self.net.substations])
        share = weights / weights.sum() * regional_load_mw
        overshoot = np.clip(-deviation - limit_pu, 0.0, None)
        voltage_exposure = np.clip(overshoot / limit_pu, 0.0, 1.0)
        margin = self.reserve_margin(reactive_loss_mvar)
        reserve_exposure = np.clip(-margin, 0.0, 1.0)
        exposure = np.maximum(voltage_exposure, reserve_exposure)
        return float(np.sum(share * exposure))
