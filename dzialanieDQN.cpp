
/*  0) inicjalizacja 
        *game0 = [0,0,0,1,0]
    1) wybierz akcje losowo lub aproksymacją
        *act0 = 1

    2) wykonaj akcje
        *game1 = [0,0,0,0,1]
        *reward0 = 1

    3) szacujemy tablice-Q
        *q = [0.6,0.2] //?(na początku losowa sieć) więc może być cokolwiek
  3.1) wybieramy najleprzy szacunek tutaj
        *maxIndex = 0
        *max = 0.6
  3.2) liczymy współczynnik
        *qsa = reward0 + //!max * 0.8
*/   