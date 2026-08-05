# KiteGuru Gizzeria Online

Dashboard mobile pubblica per consultare dal browser la previsione del vento di
domani e dopodomani a Gizzeria, con lettura termica locale, confronto dei modelli
e dato live della centralina Holfuy.

## Avvio

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

La dashboard pubblica e' stateless: non contiene il database o file privati del
computer che raccoglie lo storico.

## Telegram

`python -m cloud.send_telegram_forecast --dry-run` genera localmente il testo
esatto dell'avviso per domani e dopodomani senza trasmetterlo. Il workflow
`Send Telegram forecast` lo invia ogni giorno alle 07:30 Europe/Rome quando nel
repository sono configurati i secret `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`.
