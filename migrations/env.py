from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import Settings
from app.db.models import Base

config = context.config
target_metadata = Base.metadata
url = Settings().database_url
if not url:
    raise ValueError("Set DATABASE_URL before running migrations")

if context.is_offline_mode():
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
