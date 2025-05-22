import poplib
from email.parser import BytesParser

class EmailReader:
    """
    Connects to an email server via POP3, fetches and parses messages.
    """
    def __init__(self, host: str, user: str, password: str, port: int = 995):
        self.host = host
        self.user = user
        self.password = password
        self.port = port

    def fetch_emails(self, count=1):
        mail = poplib.POP3_SSL(self.host, self.port)
        mail.user(self.user)
        mail.pass_(self.password)

        num_emails = len(mail.list()[1])
        if num_emails == 0:
            mail.quit()
            return []

        fetch_count = min(num_emails, count)
        messages = []

        for i in range(num_emails - fetch_count + 1, num_emails + 1):
            resp, lines, octets = mail.retr(i)
            raw_email = b"\r\n".join(lines)
            msg = BytesParser().parsebytes(raw_email)

            # extract a text body
            body = "(No plain-text body found)"
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and part.get_content_disposition() in (None, "inline"):
                        charset = part.get_content_charset() or "utf-8"
                        body = part.get_payload(decode=True).decode(charset, errors="replace")
                        break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")

            messages.append({
                "id": i,
                "from": msg.get("From"),
                "to": msg.get("To"),
                "date": msg.get("Date"),
                "subject": msg.get("Subject"),
                "body": body
            })

        mail.quit()
        return messages