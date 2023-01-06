Soll unser Kernel auch getemplated sein damit man RowIdx und ColIdx mit uint32 oder uint64 verwenden kann?
cpumultiplyDloadMTX gibt nur int zurück.

Wie soll mit Werten auf der Diagonalen der Inputmatrix umgegangen werden?
Entspricht verbindung zu sich selbst. Resultiert in immer mindestens eine Collision pro Node.
(Außer man schließt Verbindungen zur gleichen Node in computation aus).

Coloring von cusparse bei minimal Beispiel https://sparse.tamu.edu/vanHeukelum/cage3 mit 5 vertices
kommt auf 15 Farben??? Naive würden wir sagen, dass höchstens 5 Farben reichen sollten.

Warum verändert sich die Anzahl nötiger Farben von CusparseColoring, wenn man die Anzahl der Partitions ändert?
Sollte ja eigentlich gleich bleiben. Matrix wird anders permutiert, aber sollte Ergebniss nicht beeinflussen oder?

Dist1 coloring:
    Wenn es keine Collisions bei kurzen Bitmasken gibt, wird es auch keine Collisions bei längeren
    bitmasken geben. Man könnte mit break im else-Zweig die Schleife über die unterschiedlichen masken längen
    vorzeitig verlassen. Auf CPU würde das Sinn machen. Auf GPU wegen Branchdivergence nur vielleicht?

Wie findet man bottleneck innerhalb eines Kernels?
NsightCompute? NVTX + NsightSystems geht nur außerhalb von Kernels oder?

Distance2
- dist2 algorithmus richtig verstanden
- Wieviel shared mem? max Nodedegree vs exact
- sorting erzwingt monolitischen Kernel?
- memory layout in ShMem
- umgang mit variablem Nodedegree
- sorting network?

Wie können wir das alte 2018 Projekt bauen? Wo CMake Package Dependency "AscMatrixIO"?

Distance2 Coloring mit cusparse? Matrix mit sichselbst multiplizieren und auf ergebnis dann Distance1Coloring von Cusparse anwenden?