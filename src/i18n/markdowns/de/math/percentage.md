Das Modell dient dazu, den prozentualen Anteil eines bestimmten Schachzugs an allen gespielten Partien sowie dessen Punktleistung im folgenden Jahr vorherzusagen. Es verwendet historische Jahresdaten und ein neuronales GRU-Netzwerk (Gated Recurrent Unit), um zeitliche Abhängigkeiten zu erkennen.

## Netzwerkarchitektur

Das Modell besteht aus folgenden Schichten:

1. **Input** — nimmt eine Sequenz historischer Daten entgegen, die drei Merkmale für jedes Jahr enthält.
2. **Masking** — ignoriert Padding-Werte, mit denen Sequenzen unterschiedlicher Länge vereinheitlicht werden (`-999`).
3. **GRU** — rekurrente Schicht mit 32 Einheiten und einer Dropout-Rate von `0.1`.
4. **Dense** — vollständig verbundene Schicht mit 32 Neuronen und ReLU-Aktivierungsfunktion.
5. **Dropout** — reduziert das Risiko von Overfitting mit einer Rate von `0.2`.
6. **Output** — Dense-Ausgabeschicht mit zwei Neuronen, die jeweils die Sigmoid-Aktivierungsfunktion verwenden.

### Trainingsparameter

| Parameter | Wert |
|---|---:|
| Typ des rekurrenten Netzwerks | GRU |
| Anzahl der GRU-Einheiten | 32 |
| GRU-Dropout | 0.1 |
| Anzahl der Dense-Neuronen | 32 |
| Dense-Aktivierung | ReLU |
| Dense-Dropout | 0.2 |
| Optimierer | Adam |
| Learning Rate | 0.0001 |
| Verlustfunktion | MSE |
| Metrik | MAE |
| Batch-Größe | 128 |
| Standardanzahl der Epochen | 30 |
| Anteil der Validierungsdaten | 20% |
| Gradient Clipping | `clipnorm = 1.0` |

## Kodierung der Eingabedaten

Für jedes Jahr erhält das Modell Informationen über das Auftreten eines bestimmten Zuges und dessen Punktleistung.

Definitionen:

- **$g_y$** — Anzahl der Partien, in denen der betreffende Zug vorkam.
- **$G_y$** — Gesamtzahl der im jeweiligen Jahr gespielten Partien.
- **$p_y$** — Gesamtzahl der Punkte, die in Partien erzielt wurden, in denen der betreffende Zug vorkam.

Aus diesen Werten werden drei Eingabemerkmale abgeleitet.

### Anteil des Zuges an allen Partien

$$
u_y =
\begin{cases}
\frac{g_y}{G_y}, & \text{wenn } G_y > 0, \\
0, & \text{wenn } G_y = 0.
\end{cases}
$$

Dabei gilt:

- **$u_y$** — Anteil der Partien mit dem betreffenden Zug an allen Partien.
- **$g_y$** — Anzahl der Partien mit dem betreffenden Zug.
- **$G_y$** — Gesamtzahl der im jeweiligen Jahr gespielten Partien.

Die Normalisierung ermöglicht den Vergleich der Popularität eines Zuges zwischen verschiedenen Jahren, auch wenn sich die Gesamtzahl der gespielten Partien erheblich unterscheidet.

### Punktleistung

$$
r_y =
\begin{cases}
\frac{p_y}{g_y}, & \text{wenn } g_y > 0, \\
0, & \text{wenn } g_y = 0.
\end{cases}
$$

Dabei gilt:

- **$r_y$** — durchschnittliche Punktleistung des Zuges.
- **$p_y$** — Gesamtzahl der erzielten Punkte.
- **$g_y$** — Anzahl der Partien mit dem betreffenden Zug.

Dieser Wert gibt die durchschnittliche Anzahl der erzielten Punkte pro Partie an, in der der betreffende Zug vorkam.

Bei der üblichen Schachwertung (1 Punkt für einen Sieg, 0.5 Punkte für ein Remis und 0 Punkte für eine Niederlage) liegt die Punktleistung im Bereich $[0,1]$.

### Indikator für das Auftreten des Zuges

$$
h_y =
\begin{cases}
1, & \text{wenn } g_y > 0, \\
0, & \text{wenn } g_y = 0.
\end{cases}
$$

Dabei gilt:

- **$h_y$** — gibt an, ob der betreffende Zug im jeweiligen Jahr aufgetreten ist.

Dieses zusätzliche Merkmal ermöglicht es dem Modell, zwischen einem Zug, der überhaupt nicht vorkam, und einem Zug zu unterscheiden, der zwar vorkam, aber eine Punktleistung von null erzielte.

### Eingabevektor

$$
X_y = [u_y,\ r_y,\ h_y]
$$

Dabei gilt:

- **$X_y$** — Vektor aus drei Eingabemerkmalen, die das jeweilige Jahr beschreiben.

Alle Eingabemerkmale liegen im Bereich $[0,1]$. Dadurch arbeitet das Modell mit normalisierten Werten und benötigt keine zusätzliche logarithmische Transformation der Partienanzahl.

## Ausgabeschicht

Das Modell verwendet eine Dense-Schicht mit zwei unabhängigen Ausgabeneuronen, die jeweils die Sigmoid-Aktivierungsfunktion besitzen.

$$
z_1 = \operatorname{sigmoid}(a_1)
$$

$$
z_2 = \operatorname{sigmoid}(a_2)
$$

Dabei gilt:

- **$z_1$** — vorhergesagter Anteil des Zuges an allen Partien.
- **$z_2$** — vorhergesagte Punktleistung des Zuges.
- **$a_1, a_2$** — Werte der Neuronen vor Anwendung der Aktivierungsfunktion.

Die Sigmoid-Funktion begrenzt beide Ergebnisse auf den Bereich $[0,1]$. Dadurch können sie den prozentualen Anteil beziehungsweise die Punktleistung darstellen.

## Dekodierung der Ergebnisse

Die Netzwerkausgaben werden in den zulässigen Wertebereich der vorhergesagten Größen überführt.

### Vorhergesagter Zuganteil

$$
\hat{u} = \operatorname{clip}(z_1,0,1)
$$

Dabei gilt:

- **$\hat{u}$** — vorhergesagter Anteil des Zuges an allen Partien.
- **$z_1$** — Ausgabe des ersten Neurons.

### Vorhergesagte Punktleistung

$$
\hat{r} = \operatorname{clip}(z_2,0,1)
$$

Dabei gilt:

- **$\hat{r}$** — vorhergesagte Punktleistung des Zuges.
- **$z_2$** — Ausgabe des zweiten Neurons.

Die Funktion $\operatorname{clip}$ begrenzt den Wert auf den Bereich $[0,1]$.

Bei der Darstellung der Ergebnisse in Prozent werden beide Werte mit 100 multipliziert.

## Endergebnis

$$
f_{\text{percentage}}(x) = [\hat{u},\hat{r}]
$$

Dabei gilt:

- **$f_{\text{percentage}}(x)$** — Vorhersageergebnis des Modells.
- **$\hat{u}$** — vorhergesagter Anteil des Zuges an allen Partien.
- **$\hat{r}$** — vorhergesagte Punktleistung.

Das Ergebnis beschreibt die relative Popularität des Zuges sowie dessen vorhergesagte Punktleistung im folgenden Jahr.

Beide Werte werden unabhängig voneinander vorhergesagt. Das Modell verwendet keinen Normalisierungsmechanismus zwischen verschiedenen Zügen. Daher muss die Summe der vorhergesagten Anteile aller Züge nicht 100% ergeben.

## Vorbereitung der Trainingsdaten

Die historischen Daten werden anhand aufeinanderfolgender Jahre in Trainingsbeispiele umgewandelt. Jedes Beispiel enthält eine Datensequenz aus den vorherigen Jahren sowie die Zielwerte des darauffolgenden Jahres.

Für historische Daten aus den Jahren 2020–2023 entstehen folgende Beispiele:

| Eingabedaten | Erwartete Ausgabe |
|---|---|
| 2020 | 2021 |
| 2020–2021 | 2022 |
| 2020–2022 | 2023 |

Das Modell lernt, den Anteil eines Zuges an allen Partien sowie dessen Punktleistung für das folgende Jahr anhand der verfügbaren historischen Daten vorherzusagen.

Sequenzen unterschiedlicher Länge werden mit dem Wert `-999` aufgefüllt. Dieser Wert wird von der Masking-Schicht ignoriert.

### Aufteilung der Daten

Die Daten werden auf Ebene der Spieler–Rollen-Serien in einen Trainings- und einen Validierungsdatensatz aufgeteilt.

- **Trainingsdatensatz** — wird zur Aktualisierung der Netzwerkgewichte verwendet.
- **Validierungsdatensatz** — dient zur Bewertung der Vorhersagequalität während des Trainings.

Der Standardanteil der Validierungsdaten beträgt 20%.

## Verlustfunktion

Das Modell verwendet die Mean Squared Error (MSE)-Verlustfunktion:

$$
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

Dabei gilt:

- **$MSE$** — mittlerer quadratischer Vorhersagefehler.
- **$N$** — Anzahl der vorhergesagten Werte.
- **$y_i$** — tatsächlicher Wert.
- **$\hat{y}_i$** — vorhergesagter Wert.

Die Verlustfunktion berücksichtigt die Fehler beider vorhergesagten Werte: den Anteil des Zuges an allen Partien und dessen Punktleistung.

Der Adam-Optimierer aktualisiert die Netzwerkgewichte, um den Vorhersagefehler zu minimieren.

## Mechanismen zur Trainingskontrolle

Während des Trainings werden zusätzliche Mechanismen zur Steuerung des Lernprozesses eingesetzt:

- **TerminateOnNaN** — bricht das Training ab, wenn ein NaN-Wert auftritt.
- **ModelCheckpoint** — speichert, sofern ein Speicherpfad angegeben wurde, das Modell mit dem besten Ergebnis auf dem Validierungsdatensatz.
- **ReduceLROnPlateau** — reduziert die Learning Rate, wenn sich der Validierungsfehler nicht weiter verbessert.
- **EarlyStopping** — beendet das Training nach einer festgelegten Anzahl von Epochen ohne Verbesserung des Validierungsfehlers.



## Modell M2 — Percentage: Bezeichnungen der Ausgabewerte

Das Modell M2 (Percentage) prognostiziert Prozentwerte, die die Popularität eines Zuges sowie dessen Punktleistung beschreiben.

| Bezeichnung | Bedeutung |
| ---------- | -------------------------------------------------------------------------------------------------------------------- |
| **M2O** | Prognostizierter prozentualer Anteil der Partien mit einem bestimmten Zug an allen Partien, die in einem bestimmten Jahr gespielt wurden (*Occurrence*). |
| **M2P** | Prognostizierte Punktleistung eines Zuges, ausgedrückt in Prozent (*Points Percentage*). |

### Interpretation der Ausgabewerte

- **M2O (Occurrence)** — der geschätzte Anteil der Partien, in denen ein bestimmter Zug vorkam, bezogen auf die Gesamtzahl der in einem bestimmten Jahr gespielten Partien.
- **M2P (Points Percentage)** — die geschätzte durchschnittliche Punktzahl pro Partie, ausgedrückt in Prozent. Beispielsweise entspricht ein Wert von 65 % durchschnittlich 0,65 Punkten pro Partie.

M2P bezeichnet nicht den Anteil des Zuges an allen erzielten Punkten. Es ist ein Maß für das durchschnittliche Punktergebnis in den Partien, in denen der betreffende Zug vorkommt.

M2O beschreibt die Popularität eines Zuges in Prozent, während M2P dessen Punktleistung darstellt.
