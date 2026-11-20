Model służy do przewidywania bezwzględnej liczby partii, w których wystąpi dany ruch szachowy, oraz liczby zdobytych punktów w kolejnym roku. Wykorzystuje historyczne dane roczne oraz sieć neuronową GRU (Gated Recurrent Unit) do wykrywania zależności czasowych.

## Architektura sieci

Model składa się z następujących warstw:

1. **Input** — przyjmuje sekwencję danych historycznych, zawierającą trzy cechy dla każdego roku.
2. **Masking** — ignoruje wartości dopełniające sekwencje o różnej długości (`-999`).
3. **GRU** — warstwa rekurencyjna z 32 jednostkami i dropoutem o wartości `0.1`.
4. **Dense** — warstwa w pełni połączona z 32 neuronami i funkcją aktywacji ReLU.
5. **Dropout** — ogranicza ryzyko przeuczenia, wykorzystując wartość `0.2`.
6. **Output** — dwa niezależne neurony wyjściowe: pierwszy z aktywacją Softplus, drugi z aktywacją Sigmoid.

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

Dla każdego roku model otrzymuje informacje dotyczące liczby wystąpień danego ruchu oraz jego skuteczności punktowej.

Oznaczenia:

- **$g_y$** — liczba partii, w których wystąpił dany ruch.
- **$p_y$** — liczba punktów zdobytych w partiach, w których wystąpił dany ruch.

Na podstawie tych wartości wyznaczane są trzy cechy wejściowe.

### Logarytmiczna reprezentacja liczby partii

$$
q_y = \ln(1+g_y)
$$

Gdzie:

- **$q_y$** — logarytmicznie zakodowana liczba partii.
- **$g_y$** — liczba partii z danym ruchem.

Zastosowanie logarytmowania ogranicza wpływ bardzo dużych wartości i zmniejsza różnice skali pomiędzy małymi i dużymi liczbami wystąpień ruchu.

Dzięki temu model może efektywniej uczyć się zależności występujących w danych o zróżnicowanej liczbie partii.

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

Skuteczność punktowa pozwala modelowi uwzględniać jakość danego ruchu niezależnie od liczby jego wystąpień.

Przy standardowej punktacji szachowej (1 punkt za zwycięstwo, 0.5 za remis i 0 za porażkę) wartość skuteczności mieści się w przedziale $[0,1]$.

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
X_y = [q_y,\ r_y,\ h_y]
$$

Gdzie:

- **$X_y$** — wektor trzech cech wejściowych opisujących dany rok.

Zastosowanie logarytmicznej reprezentacji liczby partii ogranicza różnice skali pomiędzy małymi i dużymi wartościami. Skuteczność punktowa oraz informacja o wystąpieniu ruchu uzupełniają opis danych historycznych.

## Warstwa wyjściowa

Model wykorzystuje dwa niezależne neurony wyjściowe, z których każdy posiada osobną funkcję aktywacji.

### Przewidywana liczba partii

$$
z_1 = \operatorname{softplus}(a_1)
$$

Gdzie:

- **$z_1$** — przewidywana logarytmicznie zakodowana liczba partii.
- **$a_1$** — wartość pierwszego neuronu przed zastosowaniem funkcji aktywacji.

Funkcja Softplus zapewnia nieujemny wynik, co jest odpowiednie dla przewidywanej wartości logarytmicznie zakodowanej liczby partii.

### Przewidywana skuteczność punktowa

$$
z_2 = \operatorname{sigmoid}(a_2)
$$

Gdzie:

- **$z_2$** — przewidywana skuteczność punktowa.
- **$a_2$** — wartość drugiego neuronu przed zastosowaniem funkcji aktywacji.

Funkcja Sigmoid ogranicza wynik do przedziału $[0,1]$, dzięki czemu może on reprezentować skuteczność punktową.

## Dekodowanie wyniku

Wyniki sieci są przekształcane do pierwotnej skali, aby uzyskać przewidywaną liczbę partii oraz liczbę zdobytych punktów.

### Przewidywana liczba partii

$$
\hat{g} = \max(0,\ e^{z_1}-1)
$$

Gdzie:

- **$\hat{g}$** — przewidywana liczba partii.
- **$z_1$** — wynik pierwszego neuronu wyjściowego.

Dekodowanie odwraca logarytmiczne kodowanie zastosowane podczas przygotowania danych:

$$
q_y = \ln(1+g_y)
$$

Dzięki temu przewidywana wartość zostaje przekształcona z przestrzeni logarytmicznej do pierwotnej skali liczby partii.

### Przewidywana liczba punktów

$$
\hat{p} =
\hat{g} \times \operatorname{clip}(z_2,0,1)
$$

Gdzie:

- **$\hat{p}$** — przewidywana liczba zdobytych punktów.
- **$\hat{g}$** — przewidywana liczba partii.
- **$z_2$** — przewidywana skuteczność punktowa.
- **$\operatorname{clip}$** — funkcja ograniczająca wartość do przedziału $[0,1]$.

Liczba punktów jest wyznaczana na podstawie przewidywanej liczby partii oraz przewidywanej skuteczności punktowej.

Takie podejście zapewnia spójność pomiędzy obydwoma przewidywanymi parametrami.

## Wynik końcowy

$$
f_{\text{absolute}}(x) = [\hat{g},\hat{p}]
$$

Gdzie:

- **$f_{\text{absolute}}(x)$** — wynik predykcji modelu.
- **$\hat{g}$** — przewidywana liczba partii.
- **$\hat{p}$** — przewidywana liczba zdobytych punktów.

Wynik opisuje bezwzględną skalę występowania ruchu oraz jego przewidywaną skuteczność wyrażoną liczbą zdobytych punktów.

## Przygotowanie danych treningowych

Dane historyczne są przekształcane w próbki uczące na podstawie kolejnych lat. Każda próbka zawiera sekwencję danych z poprzednich lat oraz wartości docelowe z następnego roku.

Dla historii obejmującej lata 2020–2023 powstają następujące próbki:

| Dane wejściowe | Oczekiwana odpowiedź |
|---|---|
| 2020 | 2021 |
| 2020–2021 | 2022 |
| 2020–2022 | 2023 |

Model uczy się przewidywać liczbę partii oraz liczbę zdobytych punktów w kolejnym roku na podstawie dostępnych danych historycznych.

Wartości docelowe obejmują logarytmicznie zakodowaną liczbę partii oraz skuteczność punktową. Po zakończeniu predykcji wyniki są przekształcane do pierwotnej skali.

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

Funkcja straty uwzględnia błędy obu przewidywanych wartości: logarytmicznie zakodowanej liczby partii oraz skuteczności punktowej.

Oznacza to, że błąd liczby partii jest obliczany w przestrzeni logarytmicznej, a nie w pierwotnej skali liczby wystąpień.

Optymalizator Adam aktualizuje wagi sieci w celu minimalizacji błędu predykcji.

## Mechanizmy kontroli treningu

Podczas treningu wykorzystywane są dodatkowe mechanizmy kontrolujące przebieg uczenia:

- **TerminateOnNaN** — przerywa trening w przypadku wystąpienia wartości NaN.
- **ModelCheckpoint** — jeśli podano ścieżkę zapisu, zapisuje model z najlepszym wynikiem na zbiorze walidacyjnym.
- **ReduceLROnPlateau** — zmniejsza learning rate, gdy błąd walidacyjny przestaje się poprawiać.
- **EarlyStopping** — kończy trening po określonej liczbie epok bez poprawy błędu walidacyjnego.

## Oznaczenia wyników modelu M1 — Absolute

Model M1 (Absolute) przewiduje wartości bezwzględne opisujące popularność ruchu oraz liczbę punktów zdobytych w partiach, w których ten ruch wystąpił.

| Oznaczenie | Znaczenie |
|---|---|
| **M1C** | Przewidywana liczba partii, w których wystąpi dany ruch (*Games Count*). |
| **M1P** | Przewidywana liczba punktów zdobytych w partiach, w których wystąpi dany ruch (*Points*). |

### Interpretacja wyników

- **M1C (Games Count)** — szacowana liczba partii, w których wystąpi dany ruch. Wartość ta opisuje przewidywaną popularność ruchu w ujęciu bezwzględnym.
- **M1P (Points)** — szacowana łączna liczba punktów zdobytych w partiach, w których wystąpi dany ruch.

Obie wartości są przewidywane przez model M1. M1C opisuje częstotliwość występowania ruchu, natomiast M1P określa przewidywaną liczbę punktów uzyskanych w partiach z tym ruchem.
