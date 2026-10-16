"""Inicializacion explicita de las tablas ORM existentes."""
from smartenergy.infrastructure.database import Base, get_engine
from smartenergy.infrastructure.models import Metrica, Registro


def init_db():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL no configurada. No se puede inicializar el esquema.")
    Base.metadata.create_all(bind=engine)
    print("Tablas creadas correctamente.")


def main():
    init_db()


if __name__ == "__main__":
    main()
