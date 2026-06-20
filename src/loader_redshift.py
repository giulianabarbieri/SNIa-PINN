#!/usr/bin/env python3
from alerce.core import Alerce

alerce_client = Alerce()
OBJECT_ID = "ZTF25aaxeojh"  # Tu objeto actual

print(f"📡 Buscando catálogos externos cruzados para {OBJECT_ID}...")

try:
    # Le pedimos a la API los catálogos cruzados (Cross-match)
    cats = alerce_client.query_catshtm(OBJECT_ID, format="pandas")
    
    if not cats.empty:
        print("\n🎉 ¡Encontré catálogos asociados a la posición del objeto!")
        print("Columnas disponibles en los catálogos cruzados:")
        print(list(cats.columns))
        print("\nContenido:")
        print(cats)
    else:
        print("\n⚠️ No se encontraron catálogos cruzados para este objeto en CatsHTM.")
        
        # Plan C: Probamos con el catálogo 'cats' general por si las moscas
        print("Probando con el segundo buscador de catálogos (query_lightcurve_cats)...")
        cats_2 = alerce_client.query_lightcurve_cats(OBJECT_ID, format="pandas")
        print(cats_2)

except Exception as e:
    print(f"\n❌ Error al buscar catálogos: {e}")