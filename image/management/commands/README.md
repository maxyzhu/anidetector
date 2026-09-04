## Command Wiki

`python manage.py ingest <folder> --deployment <id>`
scan a folder, register new images against a deployment, run MegaDetector.

`python manage.py classify_species`
run SpeciesNet on pending animal detections.

`python manage.py profile <dataset>`
time each pipeline stage against a throwaway test database.
