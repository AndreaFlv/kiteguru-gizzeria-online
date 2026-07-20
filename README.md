# KiteGuru Gizzeria Online

Dashboard pubblica per consultare dal browser la previsione del vento del giorno
successivo a Gizzeria, con lettura termica locale, confronto dei modelli e dato
live della centralina Holfuy.

## Avvio

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

La dashboard pubblica e' stateless: non contiene il database o file privati del
computer che raccoglie lo storico.
