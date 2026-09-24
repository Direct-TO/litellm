Read @CLAUDE.md for coding guidelines

The root Docker Compose file mounts `toapis_config.yaml` for the local ToAPIs model routes. Credentials are read from `TOAPIS_API_KEY` and `LITELLM_MASTER_KEY` in the ignored `.env` file. The `gemini-3.0-flash` route maps to the upstream `gemini-3-flash-official` model. Compose uses its own PostgreSQL service and overrides the `.env` database URL
