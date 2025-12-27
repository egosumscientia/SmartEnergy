# scripts/seed_db.py
from core.database import Base, engine
from core.models import Registro, Metrica


def init_db():
    print("Creando tablas en la base de datos...")
    Base.metadata.create_all(bind=engine)
    print("Tablas creadas correctamente.")


if __name__ == "__main__":
    init_db()
