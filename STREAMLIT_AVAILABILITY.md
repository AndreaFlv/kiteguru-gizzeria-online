# Disponibilità della dashboard Streamlit

Streamlit Community Cloud iberna le app senza traffico per 12 ore. In tale
stato la route pubblica restituisce comunque HTTP 200, ma mostra la schermata
`Zzzz` e non il contenuto KiteGuru. Un semplice controllo dello status HTTP non
è quindi una prova sufficiente.

Il workflow `Keep Streamlit app available` viene eseguito ogni quattro ore e:

1. apre la dashboard con un browser headless reale;
2. rileva e preme il pulsante di riattivazione se l'app è sospesa;
3. verifica l'endpoint del backend Streamlit;
4. verifica che nel frame applicativo sia visibile `KiteGuru · Gizzeria`;
5. conserva per 30 giorni un report JSON del controllo come artefatto GitHub.

Il controllo può essere avviato manualmente da GitHub Actions tramite
`workflow_dispatch`. Il watchdog non modifica forecast, metriche o dataset.
