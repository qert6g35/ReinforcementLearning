
/*  0) inicjalizacja 
        *game0 = [0,0,0,1,0]
    1) wykonaj aproksymację
        *q0 = [0.6,0.8]

    2) wykonaj akcje
        *game1 = [0,0,0,0,1]
        *reward0 = 1

    3) szacujemy tablice-Q
        *q1 = [0.1,0.4] //?(na początku losowa sieć) więc może być cokolwiek
  3.1) wybieramy najleprzy szacunek tutaj
        *maxIndex = 0
        *max = 0.4
  3.2) liczymy współczynnik
        *qsa = reward0 + //!max * 0.8

    4)  zapisujemy przerobione wyjście //! reward za krok 0 ocenia
        *q1_mod = [0.1,reward0 + 0.4*0.8]

    5) sieć uczymy za pomocą błędu obliczanego jako
        *J = q1 - q1_mod = [0.1-0.1,0.4 - 1.32] = [0,-0.98]
        !czemu to nie ma sensu??
        J = [R+]



[[0 0]
 [0 1]
 [1 0]
 [1 1]] [[0]
 [1]
 [1]
 [0]] [[0 0]] [[0]]
[[0 0]
 [0 1]
 [1 0]
 [1 1]] [[0]
 [1]
 [1]
 [0]] [[0 0]] [[0]]
[[0 0]
 [0 1]
 [1 0]
 [1 1]] [[0]
 [1]
 [1]
 [0]] [[0 0]] [[0]]
[[0 0]
 [0 1]
 [1 0]
 [1 1]] [[0]
...

*/   