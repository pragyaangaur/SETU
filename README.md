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
| Sensing | What is the Sun doing right now | Solar wind at the L1 point, live or from the archive, plus ground magnetometers |
| Forecast | How hard will the ground field shake in 30 to 90 minutes | Physics informed dilated convolution network with calibrated quantile heads |
| Induction | What electric field appears in the crust | Plane wave response of a layered Earth model |
| Grid | Which transformers carry how much current | Lehtinen and Pirjola nodal admittance solver, then AC power flow |
| Decision | What should the operator do about it | Scenario based optimisation under a conditional value at risk constraint |

## Results

Full tables are in [RESULTS.md](RESULTS.md), which is written by a script from the saved training reports so the numbers cannot drift away from the runs that produced them. The short version follows.

Three storms are held out completely: the Gannon storm of May 2024, the Halloween storm of October 2003, and the October 2024 storm. They never touched the model, the scaler, the alarm threshold, or the calibration map.

| Horizon | Probability of detection | False alarm ratio | Heidke skill score | Brier skill score |
| --- | --- | --- | --- | --- |
| 30 minutes | 0.83 | 0.37 | 0.67 | 0.34 |
| 45 minutes | 0.78 | 0.38 | 0.63 | 0.28 |
| 60 minutes | 0.78 | 0.43 | 0.60 | 0.28 |
| 90 minutes | 0.71 | 0.44 | 0.57 | 0.28 |

The model has 19,000 parameters, reaches back 315 minutes, and trains in 24 minutes on a laptop with no graphics card.

### The clock the solar wind sits on

This is the central design decision and it is worth stating on its own.

The OMNI archive publishes the solar wind against the time it reaches the nose of the bow shock rather than the time it was measured. That is the right convention for studying what the magnetosphere did, and it is exactly the wrong one for building a warning system, because under it a shock front appears in the record at the same instant it strikes the Earth. The first version of this project trained on that convention and had no warning time to forecast into at all.

The archive records the shift it applied, so it can be undone. Each measurement is moved back to the moment the spacecraft actually saw it, using the travel time implied by spacecraft distance and measured speed, which needs nothing a real time system would not already have. Training the same model both ways on the same storms gives the comparison in [RESULTS.md](RESULTS.md). The spacecraft clock wins at every horizon, and the gain in probabilistic skill is much larger than the gain in detection, which is what you would expect from a model that now has real information about the future rather than a better fit to the present.

### The limit on how much warning is possible

There is a hard ceiling here that no model can raise, and it is the most useful thing in this project to be able to explain.

A disturbance can only be forecast at a horizon shorter than its own travel time from the spacecraft to the Earth. At any longer horizon it had not yet reached the spacecraft when the forecast had to be issued. During the fast solar wind of 10 May 2024 that travel time was about thirty minutes, so no forecast at forty five minutes or more could have caught that shock, however good the model was. The delay runs from roughly twenty five minutes to eighty depending on speed, which means the available warning is shortest exactly when the storm is fastest.

The system handles this by forecasting at four horizons that straddle the delay, and by giving the network the propagation delay as an input so it can tell which situation it is in.

### What the replay shows, including where it falls short

Through the body of the May 2024 storm, 87 percent of disturbed decision steps carried a standing alarm. The system did not have an alarm standing before the very first disturbed minute, and the reason is the limit above rather than a defect in the model. The shock reached the spacecraft eleven minutes after the forecast for that minute had to be issued. The alarm probability rises from 0.04 to 0.20 within one decision step of the shock becoming visible, which is as fast as the physics allows.

Both numbers are reported side by side, because the first alone flatters the system and the second alone condemns it.

### Calibration

A stated ninetieth percentile should sit above the observation ninety percent of the time, since the decision layer treats these as real probabilities. The predicted quantiles are corrected by a map fitted on part of the validation set and judged on the rest, following Kuleshov, Fenner, and Ermon (2018). Fitting and judging on the same data made the map look almost perfect and it was not, so the check is now on storms the map has never seen.

Calibration remains the weakest part of the system. It improves the whole distribution and it does not fully transfer from the validation storms to the test storms, which is stated in [RESULTS.md](RESULTS.md) with the numbers rather than smoothed over.

### What the physics chain says about the North East

- The Shillong Plateau produces an induced electric field 7.3 times larger than the Bengal and Tripura basin does, for the same magnetic disturbance at a five minute period.
- On the standard one volt per kilometre benchmark the network is more than three times more exposed to an eastward electric field than to a northward one. The Siliguri corridor and the 361 km direct current route to Biswanath Chariali both run east to west, and the storm time current systems that reach Indian latitudes produce exactly an eastward field. The alignment between the driver and the network is a consequence of the geometry rather than an assumption.
- Earth currents sum to zero across the network to one part in ten to the fourteen, which is the Kirchhoff check on the solver.

### What the decision layer says

Against a hypothetical extreme event, five actions costing 486 lakh rupees remove 4,551 lakh rupees of tail risk, a return of about nine for each rupee. The first action chosen is moving the direct current link off earth return.

Eight neutral blocking devices cut the regional reactive absorption by 75 percent. The third site the search picks is Binaguri, which does not appear anywhere in the top eight of a ranking that scores each site on its own, because it only starts to matter once the two sites above it are blocked.

At the disturbance levels actually observed during the May 2024 storm, the model recommends no action at all. That is the correct answer and it is reported as such.

## Running it live

The archive is what the model learns from and it appears months after the fact. The operational counterpart reads the NOAA real time solar wind feed, which needs no account, and returns the same columns, so the same features and the same trained model run on either one.

```bash
python -m setu.cli live
```

This prints the current solar wind, how long it will take to arrive, the probabilistic forecast at all four horizons, the resulting current and voltage consequence across the network, and the recommended action. It also checks this project's propagation delay against the operational feed's own propagated product, which agrees to within about half a minute.

## Figures

Every figure in [docs/figures](docs/figures) is generated by `scripts/make_figures.py` from the real data and the real model, so none of them can drift away from the result it shows.

## Data and collaboration requests

Four gaps can only be closed by people outside this project, and drafted requests for all of them are in [outreach](outreach/README.md). The largest is that the model trains on Alibag and Hyderabad, near ten degrees geomagnetic latitude, and is applied to a region near sixteen. Shillong observatory data would remove that entirely.

## Install and run

```bash
pip install -r requirements.txt

python -m setu.cli benchmark                  # physics checks and the network model
python -m setu.cli live                       # the whole chain on the solar wind right now
python -m setu.cli replay --event 2024-05-10  # walk a held out storm minute by minute
python -m setu.cli placement --budget 8       # where to put blocking devices
python -m setu.cli train --time-base l1 --tag l1
```

The trained model is in the repository, so `live` and `replay` work straight after cloning. Data is downloaded on first use and cached under `data/raw`, so the first run of a command that needs history is slow and later ones are not.

The only dependencies are numpy, scipy, pandas, matplotlib, and requests. The neural network and the power flow are written out by hand rather than pulled from a library, so this installs and runs anywhere.

## Scope

In scope are the 400 kV and 220 kV transmission levels of the North East Region, quasi-DC geomagnetically induced current effects, the chain from L1 solar wind to transformer consequence, and operator decision support.

Out of scope are distribution networks below 220 kV, three dimensional magnetotelluric inversion, pipeline and railway induction effects, direct control actuation, and security of the data path. These are named as future work rather than treated as solved.

## Honest limitations

There is no public dataset of measured transformer neutral currents in India, so the consequence model is validated against physics and against published international measurements rather than against Indian ground truth.

The network model is synthetic, because real network parameters are restricted. It is built to be topologically and electrically representative rather than exact, so a conclusion about which parts of the network are exposed will hold while an absolute current in ampere should be read as an estimate.

The forecast model is trained on Alibag and Hyderabad, which sit near ten degrees geomagnetic latitude, and applied to a region near sixteen. Shillong and Tirunelveli are held by the Indian Institute of Geomagnetism and a request for them is drafted in [outreach](outreach/README.md).

The Halloween storm of 2003 loses most of its record on the spacecraft clock, because the particle instruments at the first Lagrange point were saturated by the event itself and the archive carries no usable speed for long stretches of it. That is a real property of that storm rather than a modelling choice, and it is why the October 2024 storm was added as a third test event.

Real time access to Aditya-L1 solar wind telemetry is not public, so the system is designed against that interface and demonstrated on the operational NOAA feed and the OMNI archive.

## Licence

MIT. See `LICENSE`.
