1. CL einlesen von Matrix pfad und bytes Shmem für Indices

2. Iteraties Berechnen des Tilings so dass Tiles klein genug für ShMem sind.
    Berücksichtigen von Minimalwert für ShMem für Reduction.

3.  Bessere aufteilung in HeaderFiles finden.

4.  Kernel weiter in einzelne (inline device) Funktionen aufteilen

5. Unittests für einzelne Device Funktionen.

6. Cub-Reductions überprüfen