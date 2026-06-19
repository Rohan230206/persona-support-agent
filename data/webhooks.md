# Webhook Subscriptions and Delivery

Webhooks allow integrations to receive real-time HTTP POST notifications when events occur.

## Configurable Webhook Events
- `user.created`: Fired when a new user registers.
- `payment.succeeded`: Fired when an invoice is successfully processed.
- `payment.failed`: Fired when a charge attempt fails.

## Webhook Signatures & Security
To verify that webhooks are sent by our platform, we include a signature in the header:
`X-Platform-Signature: t=162387126,v1=98af21d0a87c12f23`

You should verify this signature using your secret webhook key:
```python
import hmac
import hashlib

# Concatenate timestamp and raw body payload
payload = f"t={timestamp}.{raw_body}".encode('utf-8')
expected = hmac.new(webhook_secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()
```
