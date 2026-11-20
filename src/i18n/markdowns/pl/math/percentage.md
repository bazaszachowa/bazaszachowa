Model służy do przewidywania procentowego udziału danego ruchu szachowego w ogólnej liczbie rozegranych partii oraz jego skuteczności punktowej w kolejnym roku. Wykorzystuje historyczne dane roczne oraz sieć neuronową GRU (Gated Recurrent Unit) do wykrywania zależności czasowych.

## Architektura sieci

Model składa się z następujących warstw:

1. **Input** — przyjmuje sekwencję danych historycznych, zawierającą trzy cechy dla każdego roku.
2. **Masking** — ignoruje wartości dopełniające sekwencje o różnej długości (`-999`).
3. **GRU** — warstwa rekurencyjna z 32 jednostkami i dropoutem o wartości `0.1`.
4. **Dense** — warstwa w pełni połączona z 32 neuronami i funkcją aktywacji ReLU.
5. **Dropout** — ogranicza ryzyko przeuczenia, wykorzystując wartość `0.2`.
6. **Output** — warstwa wyjściowa Dense z dwoma neuronami i niezależną funkcją aktywacji Sigmoid.

### Parametry treningu

| Parametr | Wartość |
|---|---:|
| Typ sieci rekurencyjnej | GRU |
| Liczba jednostek GRU | 32 |
| Dropout GRU | 0.1 |
| Liczba neuronów Dense | 32 |
| Aktywacja Dense | ReLU |
| Dropout Dense | 0.2 |
| Optymalizator | Adam |
| Learning rate | 0.0001 |
| Funkcja straty | MSE |
| Metryka | MAE |
| Rozmiar batcha | 128 |
| Domyślna liczba epok | 30 |
| Udział walidacji | 20% |
| Ucinanie gradientów | `clipnorm = 1.0` |

## Kodowanie danych wejściowych

Dla każdego roku model otrzymuje informacje dotyczące występowania danego ruchu oraz jego skuteczności punktowej.

Oznaczenia:

- **$g_y$** — liczba partii, w których wystąpił dany ruch.
- **$G_y$** — całkowita liczba partii rozegranych w danym roku.
- **$p_y$** — liczba punktów zdobytych w partiach, w których wystąpił dany ruch.

Na podstawie tych wartości wyznaczane są trzy cechy wejściowe.

### Udział ruchu w partiach

$$
u_y =
\begin{cases}
\frac{g_y}{G_y}, & \text{gdy } G_y > 0, \\
0, & \text{gdy } G_y = 0.
\end{cases}
$$

Gdzie:

- **$u_y$** — udział danego ruchu w ogólnej liczbie partii.
- **$g_y$** — liczba partii z danym ruchem.
- **$G_y$** — całkowita liczba partii w danym roku.

Normalizacja pozwala porównywać popularność ruchu pomiędzy latami, nawet jeśli liczba rozegranych partii znacząco się różni.

### Skuteczność punktowa

$$
r_y =
\begin{cases}
\frac{p_y}{g_y}, & \text{gdy } g_y > 0, \\
0, & \text{gdy } g_y = 0.
\end{cases}
$$

Gdzie:

- **$r_y$** — średnia skuteczność punktowa ruchu.
- **$p_y$** — liczba zdobytych punktów.
- **$g_y$** — liczba partii z danym ruchem.

Wartość skuteczności określa średnią liczbę zdobytych punktów przypadającą na jedną partię, w której wystąpił dany ruch.

Przy standardowej punktacji szachowej (1 punkt za zwycięstwo, 0.5 za remis i 0 za porażkę) skuteczność mieści się w przedziale $[0,1]$.

### Informacja o wystąpieniu ruchu

$$
h_y =
\begin{cases}
1, & \text{gdy } g_y > 0, \\
0, & \text{gdy } g_y = 0.
\end{cases}
$$

Gdzie:

- **$h_y$** — informacja określająca, czy ruch wystąpił w danym roku.

Dodatkowa cecha pozwala odróżnić brak wystąpienia ruchu od sytuacji, w której ruch wystąpił, ale jego skuteczność wyniosła zero.

### Wektor wejściowy

$$
X_y = [u_y,\ r_y,\ h_y]
$$

Gdzie:

- **$X_y$** — wektor trzech cech wejściowych opisujących dany rok.

Wszystkie cechy wejściowe mieszczą się w przedziale $[0,1]$. Dzięki temu model operuje na znormalizowanych wartościach i nie wymaga dodatkowego logarytmowania liczby partii.

## Warstwa wyjściowa

Model wykorzystuje warstwę Dense z dwoma niezależnymi neuronami wyjściowymi, z których każdy posiada funkcję aktywacji Sigmoid.

$$
z_1 = \operatorname{sigmoid}(a_1)
$$

$$
z_2 = \operatorname{sigmoid}(a_2)
$$

Gdzie:

- **$z_1$** — przewidywany udział ruchu w ogólnej liczbie partii.
- **$z_2$** — przewidywana skuteczność punktowa ruchu.
- **$a_1, a_2$** — wartości neuronów przed zastosowaniem funkcji aktywacji.

Funkcja Sigmoid ogranicza oba wyniki do przedziału $[0,1]$, dzięki czemu mogą one reprezentować odpowiednio udział procentowy oraz skuteczność punktową.

## Dekodowanie wyniku

Wyniki sieci są przekształcane do przedziału dopuszczalnego dla przewidywanych wartości.

### Przewidywany udział ruchu

$$
\hat{u} = \operatorname{clip}(z_1,0,1)
$$

Gdzie:

- **$\hat{u}$** — przewidywany udział ruchu w partiach.
- **$z_1$** — wynik pierwszego neuronu wyjściowego.

### Przewidywana skuteczność punktowa

$$
\hat{r} = \operatorname{clip}(z_2,0,1)
$$

Gdzie:

- **$\hat{r}$** — przewidywana skuteczność punktowa ruchu.
- **$z_2$** — wynik drugiego neuronu wyjściowego.

Funkcja $\operatorname{clip}$ ogranicza wartość do przedziału $[0,1]$.

Podczas prezentacji wyników w procentach obie wartości są mnożone przez 100.

## Wynik końcowy

$$
f_{\text{percentage}}(x) = [\hat{u},\hat{r}]
$$

Gdzie:

- **$f_{\text{percentage}}(x)$** — wynik predykcji modelu.
- **$\hat{u}$** — przewidywany udział ruchu w partiach.
- **$\hat{r}$** — przewidywana skuteczność punktowa.

Wynik opisuje względną popularność ruchu oraz jego przewidywaną skuteczność w kolejnym roku.

Obie wartości są przewidywane niezależnie. Model nie wykorzystuje mechanizmu normalizacji pomiędzy różnymi ruchami, dlatego suma przewidywanych udziałów wszystkich ruchów nie musi wynosić 100%.

## Przygotowanie danych treningowych

Dane historyczne są przekształcane w próbki uczące na podstawie kolejnych lat. Każda próbka zawiera sekwencję danych z poprzednich lat oraz wartości docelowe z następnego roku.

Dla historii obejmującej lata 2020–2023 powstają następujące próbki:

| Dane wejściowe | Oczekiwana odpowiedź |
|---|---|
| 2020 | 2021 |
| 2020–2021 | 2022 |
| 2020–2022 | 2023 |

Model uczy się przewidywać udział ruchu w partiach oraz jego skuteczność punktową w kolejnym roku na podstawie dostępnych danych historycznych.

Sekwencje o różnej długości są uzupełniane wartością `-999`, która jest ignorowana przez warstwę Masking.

### Podział danych

Dane są dzielone na zbiór treningowy i walidacyjny na poziomie serii zawodnik–rola.

- **Zbiór treningowy** — wykorzystywany do aktualizacji wag sieci.
- **Zbiór walidacyjny** — wykorzystywany do oceny jakości predykcji podczas treningu.

Domyślny udział zbioru walidacyjnego wynosi 20%.

## Funkcja straty

Model wykorzystuje funkcję Mean Squared Error (MSE):

$$
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

Gdzie:

- **$MSE$** — średni kwadrat błędu predykcji.
- **$N$** — liczba przewidywanych wartości.
- **$y_i$** — rzeczywista wartość.
- **$\hat{y}_i$** — wartość przewidziana przez model.

Funkcja straty uwzględnia błędy obu przewidywanych wartości: udziału ruchu w partiach oraz jego skuteczności punktowej.

Optymalizator Adam aktualizuje wagi sieci w celu minimalizacji błędu predykcji.

## Mechanizmy kontroli treningu

Podczas treningu wykorzystywane są dodatkowe mechanizmy kontrolujące przebieg uczenia:

- **TerminateOnNaN** — przerywa trening w przypadku wystąpienia wartości NaN.
- **ModelCheckpoint** — jeśli podano ścieżkę zapisu, zapisuje model z najlepszym wynikiem na zbiorze walidacyjnym.
- **ReduceLROnPlateau** — zmniejsza learning rate, gdy błąd walidacyjny przestaje się poprawiać.
- **EarlyStopping** — kończy trening po określonej liczbie epok bez poprawy błędu walidacyjnego.

## Oznaczenia wyników modelu M2 — Percentage

Model M2 (Percentage) przewiduje wartości procentowe opisujące popularność ruchu oraz jego skuteczność punktową.

| Oznaczenie | Znaczenie |
|---|---|
| **M2O** | Przewidywany procentowy udział partii z danym ruchem we wszystkich partiach rozegranych w danym roku (*Occurrence*). |
| **M2P** | Przewidywana skuteczność punktowa ruchu, wyrażona w procentach (*Points Percentage*). |

### Interpretacja wyników

- **M2O (Occurrence)** — szacowany udział partii, w których wystąpił dany ruch, w ogólnej liczbie partii rozegranych w danym roku.
- **M2P (Points Percentage)** — szacowana średnia liczba punktów przypadająca na jedną partię, wyrażona w procentach. Przykładowo wartość 65% oznacza średnio 0,65 punktu na partię.

M2P nie oznacza udziału ruchu we wszystkich zdobytych punktach. Jest miarą średniego wyniku punktowego w partiach, w których występuje dany ruch.

M2O opisuje popularność ruchu w ujęciu procentowym, natomiast M2P przedstawia jego skuteczność punktową.
