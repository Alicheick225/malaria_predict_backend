FROM python:3.11-slim

WORKDIR /app

# Dépendances système requises par geopandas/shapely/rasterio (libgeos, libgdal...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Render (et d'autres PaaS) injectent un $PORT dynamique à l'exécution ; on retombe
# sur 8000 par défaut pour un usage local/Docker Compose. La forme shell (et non exec)
# est nécessaire pour que $PORT soit interpolé par le shell au démarrage du conteneur.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
