# core/config.py
import os
from dotenv import load_dotenv

# Carga el archivo .env desde la raiz del proyecto
load_dotenv()

# URL completa: postgresql://smartuser:smartpass@localhost:5432/smartenergy
DATABASE_URL = os.getenv("DATABASE_URL")
