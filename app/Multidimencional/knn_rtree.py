from rtree import index
from image_caracteristicas import load_images, get_vector, get_pos_to_id, get_data_images, EXPECTED_LENGTH_DATA

class KNN_R_Tree:
    def __init__(self, n_data: int, load_data: bool):
        self.n_data = n_data
        p = index.Property()
        p.dimension = EXPECTED_LENGTH_DATA
        if load_data:
            self.ids, self.id_to_pos = load_images("images1.csv", "output.bin", "position_data.bin", n=n_data)
            print(f"id_to_pos: {self.id_to_pos}")  # Agregar impresión del diccionario id_to_pos
        self.idx = index.Index(properties=p)
        
        for i in range(n_data):
            vector = get_vector("output.bin", "position_data.bin", i)
            self.idx.insert(i, tuple(vector + vector))

    def knn_search(self, id: int, k: int = 8):
        point = get_pos_to_id(id, self.id_to_pos)
        assert point >= 0, "No existe el ID"

        result_ids = list(self.idx.nearest(tuple(get_vector("output.bin", "position_data.bin", point)), num_results=k))
        return get_data_images(result_ids)

# Ejemplo de uso
if __name__ == "__main__":
    knn_rtree = KNN_R_Tree(n_data=100, load_data=True)
    vecinos = knn_rtree.knn_search(id=15970, k=3)
    print(f"Vecinos más cercanos: {vecinos}")
