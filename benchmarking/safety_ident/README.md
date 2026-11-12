# Identification of the safety filters' bounds

The bounds in `simulation/safety/ident/plant_bounds_chrono9_box.json` (README of
`simulation/safety`, "Parameters") come from these two scripts, run on
north's Chrono 9.0.0 (the plant of every study):

    python ident_runs.py <out>/cert_shafts --powertrain shafts --certificate
    python ident_runs.py <out>/cert_simple --powertrain simple --certificate
    python ident_analyze.py <out>/cert_out <out>/cert_shafts <out>/cert_simple
    cp <out>/cert_out/ident.json ../../simulation/safety/ident/plant_bounds_chrono9_box.json

1,026 plant runs (927 with a log: the full powertrain cannot reach 11 m/s on clay or dirt, the simple
one on sand), flat SCM, no filter, a few minutes on 12 workers. The
manoeuvres are not any evaluation scenario: open-loop steering, throttle,
coasting, launches and partial braking for 1 or 2 s from a cruise of 3, 5, 7, 9
or 11 m/s (or from rest), then straight full braking to a stop. `ACMD_CODE` points
both scripts at another code copy.

`plant_bounds_chrono9.json` is the first version (three footprint discs, one round
tube, cruise speeds up to 7 m/s), kept so commit c18b386 reproduces.
