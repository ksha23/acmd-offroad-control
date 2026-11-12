# Identification of the derived filters' bounds

The bounds in `simulation/safety/ident/plant_bounds_chrono9.json` (README of
`simulation/safety`, "Derived parameters") come from these two scripts, run on
north's Chrono 9.0.0 (the plant of every study):

    python ident_runs.py <out>/cert_shafts --powertrain shafts --certificate
    python ident_runs.py <out>/cert_simple --powertrain simple --certificate
    python ident_analyze.py <out>/cert_out <out>/cert_shafts <out>/cert_simple
    cp <out>/cert_out/ident.json ../../simulation/safety/ident/plant_bounds_chrono9.json

630 plant runs, flat SCM, no filter, about a minute on 14 workers. The
manoeuvres are not any evaluation scenario: open-loop steering, throttle,
coasting, launches and partial braking for 1 or 2 s from a cruise of 3, 5 or
7 m/s (or from rest), then straight full braking to a stop. `ACMD_CODE` points
both scripts at another code copy.
