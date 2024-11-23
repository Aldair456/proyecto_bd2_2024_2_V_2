document.getElementById('image-search-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData(this); // Crear el FormData a partir del formulario
    const resultsContainer = document.getElementById('image-results-index');
    const resultsSection = document.getElementById('tablaA-container');
    resultsContainer.innerHTML = ''; // Limpiar resultados anteriores
    resultsSection.style.display = 'none'; // Ocultar resultados hasta que haya datos

    try {
        // Muestra un mensaje de carga
        resultsContainer.innerHTML = '<p style="text-align: center; color: blue;">Buscando imágenes similares...</p>';

        const response = await fetch('/knn/priority', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const results = await response.json();
            resultsContainer.innerHTML = ''; // Limpiar el mensaje de carga

            if (results.length === 0) {
                resultsContainer.innerHTML = '<p style="color: red; text-align: center;">No se encontraron imágenes similares.</p>';
            } else {
                results.forEach(result => {
                    // Crear un contenedor para cada resultado
                    const imageDiv = document.createElement('div');
                    imageDiv.style.textAlign = 'center';

                    // Crear elemento <img> para mostrar la imagen
                    const img = document.createElement('img');
                    img.src = result.Link; // Usar la URL proporcionada por el backend
                    img.alt = result.Filename;
                    img.style.width = '200px';
                    img.style.height = '200px';
                    img.style.objectFit = 'cover';
                    img.style.border = '1px solid #ddd';
                    img.style.borderRadius = '5px';

                    // Crear un elemento <p> para mostrar el nombre y la distancia
                    const caption = document.createElement('p');
                    caption.innerHTML = `<strong>${result.Filename}</strong><br>Distancia: ${result.Distance.toFixed(2)}`;

                    // Añadir imagen y texto al contenedor
                    imageDiv.appendChild(img);
                    imageDiv.appendChild(caption);
                    resultsContainer.appendChild(imageDiv);
                });
            }

            // Mostrar resultados
            resultsSection.style.display = 'block';
        } else {
            const error = await response.json();
            resultsContainer.innerHTML = `<p style="color: red;">Error: ${error.error || 'Ocurrió un error desconocido.'}</p>`;
        }
    } catch (err) {
        console.error('Error al conectar con el servidor:', err);
        resultsContainer.innerHTML = '<p style="color: red; text-align: center;">Error al conectar con el servidor. Inténtalo nuevamente.</p>';
    }
});
