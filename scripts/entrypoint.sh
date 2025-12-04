#!/bin/bash
set -e

# Run database migrations unless explicitly disabled (e.g. worker process)
if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
	echo "Running database migrations..."
	alembic upgrade head
else
	echo "Skipping database migrations (RUN_MIGRATIONS=${RUN_MIGRATIONS})"
fi

# Execute the passed command
exec "$@"
