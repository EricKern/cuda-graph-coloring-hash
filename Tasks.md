1. CL einlesen von Matrix pfad und bytes Shmem für Indices

2. Iteraties Berechnen des Tilings so dass Tiles klein genug für ShMem sind.
    Berücksichtigen von Minimalwert für ShMem für Reduction.

3.  Bessere aufteilung in HeaderFiles finden.

4.  Kernel weiter in einzelne (inline device) Funktionen aufteilen

5. Unittests für einzelne Device Funktionen.

6. Cub-Reductions überprüfen


## 17.01
- kernel in funktionen unterteilen -> compute / reduction -> Eric
    - Done. Erste Reduction ist 4 Zeiler da hab ich nichts gemacht
- andere hashfunktionen mit testen
    - Done. Aber Cpu-Coloring versionen sollte man noch zusätzlichen int Parameter geben mit welcher Hash funktion gerechnet werden soll.
    - Unittest muss man wieder zum laufen bringen mit neuer GPU und CPU version.
- Kernel nur Daten laden + schreiben -> nvbench -> Daniel -> erledigt
- Aufräumen Dist1 + kernel setup
    - Dist1 ist finde ich clean. GoogleStyleGuide vielleicht nicht immer eingehalten, aber ok.
    - Kernel-Setup ist wieder schlecht. ShMem kann man nicht wiederverwenden wenn man mehrere HashFunktionen
    verwendet und zwischendrin schon reduziert. Status output wie groß finales Tiling und shmem consumption is
    finde ich gut.
    - Inkonsistente verwendung von printf in main und cout in kernel_setup schlecht. Ich mag std::printf mehr.
- Dist2 ohne sorting network
- Tiles in subtiles unterteilen um für mehr coalescedes laden -> allignment beachten
- coalesced reduction AOS? -> in global memory
    - Done. Braucht man für mehrere hash functionen.
- simples tiling -> ohne workload zu beachten -> baseline
- launch bounds
- shared memory reusage -> SO artikel
    - zu static und dynamic Sharedmem mixen gibts wenig online docu dazu und weiß ich nicht ob da Fehler passieren können. Hinsichtlich mehrerer Blocks pro SM?
- sorting network für dist2


## 23.01
- baseline ohne tiling muss irgendwie berechnen wie viel
 pro block loop geladen werden kann. Wie berechnen?
- berechnung wann tiling fein genug ist braucht lange und mit permutation noch länger.
Shared memory für dist2 mit sorting network braucht genaue größe der partition. (und dann vllt nochmal neu tilen)
- launch bounds
- cooperative launch
- unique pointer für cuda malloc?
- vllt test aufräumen
- Objekt für benchmark setup und cleanup


# 24.01
- Singleton in benchmark
- Statisches Partitionieren

- kernel teilen in initial block reduction und final reduction              Eric
- block reduction auskommentieren und laufzeit testen (nur hash berechnen)  Daniel

- klären wie viele Hash funktionen gleichzeitig gebencht werden
- google päsentation für aktuelle entwicklung und notizen

- preprocessing evtl mit thrust/cub wenn nicht aufwändig
- Streaming variante erst mal später

# 30.01
- constexpression struct nicht einfach so in device code sichtbar?
- Übergabeparameter vs gecached laden?
- wie viele Hash funktionen
- wieso dist2 so langsam?

# 31.01 
- plot mit verschiedenen hash funktionen machen -> kollisionen pro bit width
- using für unsigned hash ergebnisse in kernel
- hash funktionen in constant memory?
- cashing in godbolt überprüfen
- timing vergleichen mit altem projekt und cusparse
- padding mit cub max value
- profile dist1 mit 2 blöcken
