# API Authentication Guide

All requests to the platform API must be authenticated using API keys or Access Tokens.

## API Key Authentication
API keys are long-lived credentials suitable for server-to-server integrations.

### Generating API Keys
1. Go to the Developer Dashboard.
2. Select **API Credentials**.
3. Click **Generate New API Key**.
4. Copy the key immediately. It will not be shown again for security reasons.

### Usage in Headers
All HTTP requests must include the API key in the custom header `X-API-KEY`.
Example request:
```http
GET /v1/customers HTTP/1.1
Host: api.platform.com
X-API-KEY: plat_key_9823hf98a23gf
Accept: application/json
```
Do not send API keys in URL query parameters, as they are logged by web servers.
