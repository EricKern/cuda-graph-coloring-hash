1. CL einlesen von Matrix pfad und bytes Shmem für Indices

2. Iteraties Berechnen des Tilings so dass Tiles klein genug für ShMem sind.
    Berücksichtigen von Minimalwert für ShMem für Reduction.

3.  Bessere aufteilung in HeaderFiles finden.

4.  Kernel weiter in einzelne (inline device) Funktionen aufteilen

5. Unittests für einzelne Device Funktionen.

6. Cub-Reductions überprüfen


## 17.01
- kernel in funktionen unterteilen -> compute / reduction -> Eric
- andere hashfunktionen mit testen 
- Kernel nur Daten laden + schreiben -> nvbench -> Daniel
- Aufräumen Dist1 + kernel setup
- Dist2 ohne sorting network
- Tiles in subtiles unterteilen um für mehr coalescedes laden -> allignment beachten
- coalesced reduction AOS? -> in global memory
- simples tiling -> ohne workload zu beachten -> baseline
- launch bounds
- shared memory reusage -> SO artikel
- sorting network für dist2