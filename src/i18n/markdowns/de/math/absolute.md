Das Modell dient dazu, die absolute Anzahl der Partien vorherzusagen, in denen ein bestimmter Schachzug vorkommen wird, sowie die Anzahl der im folgenden Jahr erzielten Punkte. Es verwendet historische Jahresdaten und ein neuronales GRU-Netzwerk (Gated Recurrent Unit), um zeitliche Abhängigkeiten zu erkennen.

## Netzwerkarchitektur

Das Modell besteht aus folgenden Schichten:

1. **Input** — nimmt eine Sequenz historischer Daten entgegen, die drei Merkmale für jedes Jahr enthält.
2. **Masking** — ignoriert Padding-Werte, mit denen Sequenzen unterschiedlicher Länge vereinheitlicht werden (`-999`).
3. **GRU** — rekurrente Schicht mit 32 Einheiten und einer Dropout-Rate von `0.1`.
4. **Dense** — vollständig verbundene Schicht mit 32 Neuronen und ReLU-Aktivierungsfunktion.
5. **Dropout** — reduziert das Risiko von Overfitting mit einer Rate von `0.2`.
6. **Output** — zwei unabhängige Ausgabeneuronen: Das erste verwendet die Softplus-Aktivierungsfunktion, das zweite die Sigmoid-Aktivierungsfunktion.

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

Für jedes Jahr erhält das Modell Informationen über die Anzahl der Vorkommen eines bestimmten Zuges und dessen Punktleistung.

Definitionen:

- **$g_y$** — Anzahl der Partien, in denen der betreffende Zug vorkam.
- **$p_y$** — Gesamtzahl der Punkte, die in Partien erzielt wurden, in denen der betreffende Zug vorkam.

Aus diesen Werten werden drei Eingabemerkmale abgeleitet.

### Logarithmische Darstellung der Partienanzahl

$$
q_y = \ln(1+g_y)
$$

Dabei gilt:

- **$q_y$** — logarithmisch kodierte Anzahl der Partien.
- **$g_y$** — Anzahl der Partien mit dem betreffenden Zug.

Die logarithmische Transformation reduziert den Einfluss sehr großer Werte und verringert die Skalendifferenzen zwischen kleinen und großen Häufigkeitswerten.

Dadurch kann das Modell Zusammenhänge in Daten mit stark unterschiedlichen Partienanzahlen effektiver lernen.

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

Die Punktleistung ermöglicht es dem Modell, die Qualität eines Zuges unabhängig von seiner Häufigkeit zu berücksichtigen.

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
X_y = [q_y,\ r_y,\ h_y]
$$

Dabei gilt:

- **$X_y$** — Vektor aus drei Eingabemerkmalen, die das jeweilige Jahr beschreiben.

Die logarithmische Darstellung der Partienanzahl reduziert die Skalendifferenzen zwischen kleinen und großen Werten. Die Punktleistung und der Indikator für das Auftreten des Zuges ergänzen die Beschreibung der historischen Daten.

## Ausgabeschicht

Das Modell verwendet zwei unabhängige Ausgabeneuronen, die jeweils eine eigene Aktivierungsfunktion besitzen.

### Vorhergesagte Anzahl der Partien

$$
z_1 = \operatorname{softplus}(a_1)
$$

Dabei gilt:

- **$z_1$** — vorhergesagte logarithmisch kodierte Anzahl der Partien.
- **$a_1$** — Wert des ersten Neurons vor Anwendung der Aktivierungsfunktion.

Die Softplus-Funktion gewährleistet ein nicht negatives Ergebnis. Dies ist für die vorhergesagte logarithmisch kodierte Anzahl der Partien geeignet.

### Vorhergesagte Punktleistung

$$
z_2 = \operatorname{sigmoid}(a_2)
$$

Dabei gilt:

- **$z_2$** — vorhergesagte Punktleistung.
- **$a_2$** — Wert des zweiten Neurons vor Anwendung der Aktivierungsfunktion.

Die Sigmoid-Funktion begrenzt das Ergebnis auf den Bereich $[0,1]$, sodass es die Punktleistung darstellen kann.

## Dekodierung der Ergebnisse

Die Netzwerkausgaben werden in ihre ursprüngliche Skala zurücktransformiert, um die vorhergesagte Anzahl der Partien und die Anzahl der erzielten Punkte zu erhalten.

### Vorhergesagte Anzahl der Partien

$$
\hat{g} = \max(0,\ e^{z_1}-1)
$$

Dabei gilt:

- **$\hat{g}$** — vorhergesagte Anzahl der Partien.
- **$z_1$** — Ausgabe des ersten Neurons.

Die Dekodierung kehrt die bei der Datenvorbereitung angewendete logarithmische Transformation um:

$$
q_y = \ln(1+g_y)
$$

Dadurch wird der vorhergesagte Wert aus dem logarithmischen Raum in die ursprüngliche Skala der Partienanzahl zurücktransformiert.

### Vorhergesagte Anzahl der Punkte

$$
\hat{p} =
\hat{g} \times \operatorname{clip}(z_2,0,1)
$$

Dabei gilt:

- **$\hat{p}$** — vorhergesagte Anzahl der erzielten Punkte.
- **$\hat{g}$** — vorhergesagte Anzahl der Partien.
- **$z_2$** — vorhergesagte Punktleistung.
- **$\operatorname{clip}$** — Funktion, die einen Wert auf den Bereich $[0,1]$ begrenzt.

Die vorhergesagte Anzahl der Punkte wird anhand der vorhergesagten Partienanzahl und der vorhergesagten Punktleistung berechnet.

Dieser Ansatz gewährleistet Konsistenz zwischen den beiden vorhergesagten Größen.

## Endergebnis

$$
f_{\text{absolute}}(x) = [\hat{g},\hat{p}]
$$

Dabei gilt:

- **$f_{\text{absolute}}(x)$** — Vorhersageergebnis des Modells.
- **$\hat{g}$** — vorhergesagte Anzahl der Partien.
- **$\hat{p}$** — vorhergesagte Anzahl der erzielten Punkte.

Das Ergebnis beschreibt das absolute Ausmaß des Auftretens eines Zuges sowie dessen vorhergesagte Leistung, ausgedrückt durch die Anzahl der erzielten Punkte.

## Vorbereitung der Trainingsdaten

Die historischen Daten werden anhand aufeinanderfolgender Jahre in Trainingsbeispiele umgewandelt. Jedes Beispiel enthält eine Datensequenz aus den vorherigen Jahren sowie die Zielwerte des darauffolgenden Jahres.

Für historische Daten aus den Jahren 2020–2023 entstehen folgende Beispiele:

| Eingabedaten | Erwartete Ausgabe |
|---|---|
| 2020 | 2021 |
| 2020–2021 | 2022 |
| 2020–2022 | 2023 |

Das Modell lernt, die Anzahl der Partien und die Anzahl der erzielten Punkte für das folgende Jahr anhand der verfügbaren historischen Daten vorherzusagen.

Die Zielwerte bestehen aus der logarithmisch kodierten Partienanzahl und der Punktleistung. Nach der Vorhersage werden die Ergebnisse in ihre ursprüngliche Skala zurücktransformiert.

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

Die Verlustfunktion berücksichtigt die Fehler beider vorhergesagten Werte: der logarithmisch kodierten Partienanzahl und der Punktleistung.

Das bedeutet, dass der Fehler für die Partienanzahl im logarithmischen Raum und nicht auf der ursprünglichen Skala der Häufigkeitswerte berechnet wird.

Der Adam-Optimierer aktualisiert die Netzwerkgewichte, um den Vorhersagefehler zu minimieren.

## Mechanismen zur Trainingskontrolle

Während des Trainings werden zusätzliche Mechanismen zur Steuerung des Lernprozesses eingesetzt:

- **TerminateOnNaN** — bricht das Training ab, wenn ein NaN-Wert auftritt.
- **ModelCheckpoint** — speichert, sofern ein Speicherpfad angegeben wurde, das Modell mit dem besten Ergebnis auf dem Validierungsdatensatz.
- **ReduceLROnPlateau** — reduziert die Learning Rate, wenn sich der Validierungsfehler nicht weiter verbessert.
- **EarlyStopping** — beendet das Training nach einer festgelegten Anzahl von Epochen ohne Verbesserung des Validierungsfehlers.



## Modell M1 — Absolute: Bezeichnungen der Ausgabewerte

Das Modell M1 (Absolute) prognostiziert absolute Werte, die die Popularität eines Zuges sowie die Anzahl der in den Partien erzielten Punkte beschreiben, in denen dieser Zug vorkam.

| Bezeichnung | Bedeutung |
| ---------- | ----------------------------------------------------------------------------------------- |
| **M1C** | Prognostizierte Anzahl der Partien, in denen ein bestimmter Zug vorkommen wird (*Games Count*). |
| **M1P** | Prognostizierte Gesamtzahl der Punkte, die in den Partien erzielt werden, in denen ein bestimmter Zug vorkam (*Points*). |

### Interpretation der Ausgabewerte

- **M1C (Games Count)** — die geschätzte Anzahl der Partien, in denen ein bestimmter Zug vorkommen wird. Dieser Wert beschreibt die prognostizierte Popularität des Zuges in absoluten Zahlen.
- **M1P (Points)** — die geschätzte Gesamtzahl der Punkte, die in den Partien erzielt werden, in denen ein bestimmter Zug vorkam.

Beide Werte werden vom Modell M1 prognostiziert. M1C beschreibt, wie häufig der Zug voraussichtlich vorkommen wird, während M1P die prognostizierte Anzahl der Punkte angibt, die in Partien mit diesem Zug erzielt werden.
