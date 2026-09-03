# SETU

SETU is a forecasting and decision support system for geomagnetically induced currents in the North East Indian power grid. The name stands for Solar Event To Utility, and setu also means bridge, which is what the system does. It carries a measurement made in space all the way through to a switching decision made in a control room.

This repository is a working artefact, so every layer described below is implemented and runs. Nothing here is a mock.

## The problem

A geomagnetic storm does not damage a power grid the way lightning does. The storm changes the magnetic field at the surface of the Earth, that changing field induces an electric field in the crust, and the electric field drives a slow quasi-DC current up the neutrals of grounded transformers. A transformer carrying that current saturates on one half of the cycle, which makes it draw large amounts of reactive power, inject harmonics into the network, and heat its own windings. The grid then fails through voltage collapse and protection misoperation. Hydro-Quebec lost its whole system in ninety two seconds this way in March 1989.

## Why the North East

The usual assumption is that India sits at low geomagnetic latitude and is therefore safe. For the North East Region that assumption does not hold, for reasons that have nothing to do with latitude.

- **Conductivity contrast.** The Shillong Plateau is resistive Precambrian gneiss. It sits directly against the deep conductive sediments of the Brahmaputra valley and the Bengal basin. Induced electric fields concentrate at boundaries like this, so the same magnetic disturbance produces a much larger driving voltage here than it would over uniform ground.
- **Network topology.** The regional grid is long and radial, and it connects to the rest of India through the narrow Siliguri corridor. Induced voltage on a line grows with its length, and a radial network has little margin to redistribute reactive power when a transformer starts absorbing it.
- **The HVDC terminal.** The Biswanath Chariali end of the 800 kV link uses single phase converter transformers, which are far more vulnerable to saturation than the three phase three limb units used elsewhere, and it has a ground electrode that couples directly into the earth return path.
- **Restoration time.** Terrain, seismic risk, and monsoon access make the North East the slowest region in the country to repair. Resilience is about recovery as much as it is about resistance, so the same event costs more here.

## Architecture

The system is five layers, and each one comes from a different discipline.

| Layer | Question it answers | Method |
| --- | --- | --- |
| Sensing | What is the Sun doing right now | Solar wind at the L1 point, ground magnetometers |
| Forecast | How hard will the ground field shake in 30 to 60 minutes | Physics informed dilated convolution network with quantile heads |
| Induction | What electric field appears in the crust | Plane wave response of a layered Earth model |
| Grid | Which transformers carry how much current | Lehtinen and Pirjola nodal admittance solver, then AC power flow |
| Decision | What should the operator do about it | Scenario based optimisation under a conditional value at risk constraint |

## Install and run

```bash
pip install -r requirements.txt
python -m setu.cli pipeline --event 2024-05-10
```

The pipeline runs offline out of the box using a bundled physics based storm generator, so you can try the whole system without downloading anything. Pass `--fetch` to pull real solar wind data from NASA OMNIWeb instead.

## Scope

In scope are the 400 kV and 220 kV transmission levels of the North East Region, quasi-DC geomagnetically induced current effects, the chain from L1 solar wind to transformer consequence, and operator decision support.

Out of scope are distribution networks below 220 kV, three dimensional magnetotelluric inversion, pipeline and railway induction effects, direct control actuation, and security of the data path. These are named as future work rather than treated as solved.

## Honest limitations

There is no public dataset of measured transformer neutral currents in India, so the consequence model is validated against physics and against published international measurements rather than against Indian ground truth. The network model is synthetic, because real network parameters are restricted, and it is built to be topologically and electrically representative rather than exact. Real time access to Aditya-L1 solar wind telemetry is not public, so the system is designed against that interface and demonstrated on DSCOVR and OMNI data.

## Licence

MIT. See `LICENSE`.
