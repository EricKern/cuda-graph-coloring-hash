1. Soll unser Kernel auch getemplated sein damit man RowIdx und ColIdx mit uint32 oder uint64 verwenden kann?
cpumultiplyDloadMTX gibt nur int zurück.

2. Wie soll mit Werten auf der Diagonalen der Inputmatrix umgegangen werden?
Entspricht verbindung zu sich selbst. Resultiert in immer mindestens eine Collision pro Node.
(Außer man schließt Verbindungen zur gleichen Node in computation aus).

3. Coloring von cusparse bei minimal Beispiel https://sparse.tamu.edu/vanHeukelum/cage3 mit 5 vertices
kommt auf 15 Farben??? Naive würden wir sagen, dass höchstens 5 Farben reichen sollten.

4. Warum verändert sich die Anzahl nötiger Farben von CusparseColoring, wenn man die Anzahl der Partitions ändert?
Sollte ja eigentlich gleich bleiben. Matrix wird anders permutiert, aber sollte Ergebniss nicht beeinflussen oder?

5. Aktuell sehen wir keinen Grund die Nodes am Rand anders zu behandeln als die Nodes im Inneren einer Partition.

6. Letzte gruppe hat unterschiedliche Hashfunktionen verwendet (k_param). Wir testen nur eine? Welche?

6. Dist1 coloring:
    Wenn es keine Collisions bei kurzen Bitmasken gibt, wird es auch keine Collisions bei längeren
    bitmasken geben. Man könnte mit break im else-Zweig die Schleife über die unterschiedlichen masken längen
    vorzeitig verlassen. Auf CPU würde das Sinn machen. Auf GPU wegen Branchdivergence nur vielleicht?

7. Wie findet man bottleneck innerhalb eines Kernels?
NsightCompute? NVTX + NsightSystems geht nur außerhalb von Kernels oder?

8. Distance2
    - distance 1/2 collisions getrennnt bestimmen?
    - dist2 algorithmus richtig verstanden
    - sorting erzwingt monolitischen Kernel?
    - memory layout in ShMem u. Wieviel shared mem?
    - sorting network?

9. Wie können wir das alte 2018 Projekt bauen? CMake Package Dependency "AscMatrixIO" finden wir nicht.

10. Distance2 Coloring mit cusparse? Matrix mit sichselbst multiplizieren und auf ergebnis dann Distance1Coloring von Cusparse anwenden?