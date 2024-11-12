import psutil

def get_chunksize(memory_fraction=0.1, bytes_per_row=1000):
    """
    Calcula un tamaño de chunk basado en una fracción de la memoria disponible.

    :param memory_fraction: Fracción de la memoria disponible que deseas utilizar para el chunk.
    :param bytes_per_row: Estimación del uso de memoria por fila en bytes.
    :return: Número de filas por chunk.
    """
    mem = psutil.virtual_memory()
    available_memory = mem.available * memory_fraction
    chunksize = available_memory // bytes_per_row
    return int(chunksize)

# Uso
import psutil
TAMANIO_CHUNK = get_chunksize(memory_fraction=0.1, bytes_per_row=1000)  # Ajusta según tus necesidades
print(f"Tamaño de chunk calculado: {TAMANIO_CHUNK} filas")
