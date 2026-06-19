# Bearer Token Authentication

Bearer tokens are short-lived JWTs (JSON Web Tokens) used for authorized user sessions.

## Header Format
Bearer tokens must be passed in the `Authorization` header of every API request.
The format is:
`Authorization: Bearer <token>`

Example request headers:
```http
GET /v1/user/profile HTTP/1.1
Host: api.platform.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

## Token Lifespan & Expiration
- Access Tokens: Valid for exactly 1 hour (3600 seconds).
- Refresh Tokens: Valid for 14 days. Used to obtain new access tokens without prompting credentials.
- Expired Token Response: Requests with expired bearer tokens will fail with a `401 Unauthorized` status and error payload `{"error": "token_expired"}`.
