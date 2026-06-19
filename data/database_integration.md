# Database Integration Guidelines

This document outlines configurations for integrating database adapters and connection pool sizes.

## Connection Strings & Settings
For production database integrations, use a structured connection URI:
`postgresql://db_user:password@db_host:5432/platform_prod`

## Connection Pool Configurations
To optimize performance and avoid database connection exhaustion:
- **Default Pool Size**: Set the maximum pool size to 20 connections per server instance.
- **Timeout**: Set the connection timeout to 30 seconds.
- **Idle Connections**: Maintain a minimum of 5 idle connections.

Example database connection setup in configuration:
```python
db_config = {
    "pool_size": 20,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800
}
```
If connection pool limit is exceeded, database connections fail with `InternalServerError` or connection timeout errors.
