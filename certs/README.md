# Корневой сертификат

Публичный Russian Trusted Root CA, скачанный 20.10.2026 по ссылке из документации GigaChat:

- [Инструкция GigaChat](https://developers.sber.ru/docs/ru/gigachat/certificates)
- [Оригинал сертификата](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt)

SHA-256 fingerprint DER:
`D2:6D:2D:02:31:B7:C3:9F:92:CC:73:85:12:BA:54:10:35:19:E4:40:5D:68:B5:BD:70:3E:97:88:CA:8E:CF:31`.

Сертификат подключается только к HTTP-клиенту приложения через дополнительный CA bundle.
Системное хранилище не изменяется; TLS hostname/certificate verification включена.
Это публичный сертификат, не клиентский ключ и не секрет.
