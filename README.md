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

## Results so far

Every number below comes from the code in this repository. The two storms used for testing, the Gannon storm of 10 May 2024 and the Halloween storm of 29 October 2003, were held out completely. They never touched the model, the scaler, or the choice of alarm threshold.

### Forecast skill on the held out storms

The network has 30,450 parameters, reaches back 315 minutes, and was trained in about 51 minutes on a laptop with no graphics card. It is scored against the 0.1 nT per second alert level at Alibag and Hyderabad, over 3,532 held out samples with a base rate of 14 percent.

| Horizon | Probability of detection | False alarm ratio | Heidke skill score | Peirce score | Brier skill score |
| --- | --- | --- | --- | --- | --- |
| 30 minutes | 0.47 | 0.40 | 0.46 | 0.41 | 0.23 |
| 45 minutes | 0.58 | 0.38 | 0.54 | 0.52 | 0.17 |
| 60 minutes | 0.50 | 0.41 | 0.47 | 0.44 | 0.13 |

The alarm threshold for each horizon was chosen on a validation split made of whole storms and then applied to the test storms unchanged.

### The result that matters most, which is a failure

The replay of the May 2024 storm shows the model missing the storm sudden commencement. At Alibag the ground went from 0.04 to 0.50 nT per second in a single step at about 16:10 UT on 10 May, and the forecast probability only crossed its alarm level three steps later. The model predicts the sustained main phase and it does not anticipate the onset.

The cause is in the training data and not in the network. OMNI has already shifted the solar wind measurement forward from the spacecraft to the nose of the bow shock, so a shock front appears in the input at the same moment it strikes the Earth. No model trained on OMNI can warn about a sudden commencement. Training on the raw first Lagrange point record instead puts the travel time back in front of the data, which is worth thirty to sixty minutes, and that is the first item of future work.

Calibration on the test storms is also poor in a way that is worth stating. Only 69 percent of observations fell below the predicted ninetieth percentile, where 90 percent should have. The two test storms are the two largest in the catalogue, so a model trained mostly on weaker events under predicts them. The right fix is to weight the training loss toward the tail rather than to widen the intervals after the fact.

### What the physics chain says about the North East

- The Shillong Plateau produces an induced electric field 7.3 times larger than the Bengal and Tripura basin does, for the same magnetic disturbance at a five minute period.
- On the standard one volt per kilometre benchmark, the network is more than three times more exposed to an eastward electric field than to a northward one. The corridor through Siliguri and the 361 km direct current route to Biswanath Chariali both run east to west, and the storm time current systems that reach Indian latitudes produce exactly an eastward field.
- Earth currents sum to zero across the network to one part in ten to the fourteen, which is the Kirchhoff check on the solver.

### What the decision layer says

Against a hypothetical extreme event, five actions costing 486 lakh rupees remove 4,551 lakh rupees of tail risk, a return of 9.4 for each rupee. The first action chosen is moving the direct current link off earth return.

Eight neutral blocking devices cut the regional reactive absorption by 75 percent. The third site the search picks is Binaguri, which does not appear in the top five of a ranking that scores each site on its own. Blocking one substation pushes its current into the neighbours, so the sites cannot be ranked independently, and that gap between the two lists is the reason the search exists.

At the disturbance levels actually observed during the May 2024 storm, the model recommends no action at all. That is the correct answer and it is reported as such.

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
