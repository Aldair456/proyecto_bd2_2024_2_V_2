document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('search-form');
    const tablaAContainer = document.getElementById('tablaA-container');
    const tablaBContainer = document.getElementById('tablaB-container');
    const tablaABody = document.getElementById('tablaA-body');
    const tablaBBody = document.getElementById('tablaB-body');
    const tiempoConsultaSpan = document.getElementById('tiempo_consulta');

    let currentQueryType = 'index'; // Tipo de consulta actual

    // Funciones para manejar la selección de tablas
    function seleccionarTabla(tablaId) {
        if (tablaId === 'tablaA') {
            currentQueryType = 'index'; // Actualiza el tipo de consulta
            tablaAContainer.style.display = 'block';
            tablaBContainer.style.display = 'none';
        } else if (tablaId === 'tablaB') {
            currentQueryType = 'postgres'; // Actualiza el tipo de consulta
            tablaAContainer.style.display = 'none';
            tablaBContainer.style.display = 'block';
        }

        // Manejar la clase 'active' para los botones
        const botones = ['tablaA', 'tablaB'];
        botones.forEach(id => {
            document.getElementById(id).classList.toggle('active', id === tablaId);
        });
    }

    searchForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Evitar el envío tradicional del formulario

        const consulta = document.getElementById('searchQuery').value.trim();
        const top_k = parseInt(document.getElementById('topk').value);

        if (consulta === '' || isNaN(top_k) || top_k < 1) {
            alert('Por favor, ingresa una consulta válida y un Top K válido.');
            return;
        }

        // Limpiar resultados anteriores
        tablaABody.innerHTML = '';
        tablaBBody.innerHTML = '';
        tablaAContainer.style.display = 'none';
        tablaBContainer.style.display = 'none';
        tiempoConsultaSpan.textContent = '0';

        // Registrar el tiempo de inicio
        const startTime = performance.now();

        // Definir la URL según el tipo de consulta actual
        const url = currentQueryType === 'index' ? '/consulta' : '/consulta/postgres';

        // Enviar la solicitud al back-end usando Fetch API
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ consulta: consulta, top_k: top_k })
        })
        .then(response => response.json())
        .then(data => {
            // Calcular el tiempo de ejecución
            const endTime = performance.now();
            const tiempoEjecucion = ((endTime - startTime) / 1000).toFixed(3);
            tiempoConsultaSpan.textContent = tiempoEjecucion;

            // Verificar si hay resultados
            if (Object.keys(data).length === 0) {
                if (currentQueryType === 'index') {
                    tablaABody.innerHTML = '<tr><td colspan="6">No se encontraron resultados.</td></tr>';
                } else {
                    tablaBBody.innerHTML = '<tr><td colspan="11">No se encontraron resultados.</td></tr>';
                }
            } else {
                // Insertar resultados en la tabla correspondiente
                if (currentQueryType === 'index') {
                    for (const id in data) {
                        const doc = data[id];
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${doc.track_id || 'N/A'}</td>
                            <td>${doc.track_name || 'N/A'}</td>
                            <td>${doc.track_artist || 'N/A'}</td>
                            <td>${doc.lyrics || 'N/A'}</td>
                            <td>${doc.playlist_name || 'N/A'}</td>
                            <td>${doc.similitud_coseno || '0.0'}</td>
                        `;
                        tablaABody.appendChild(row);
                    }
                    tablaAContainer.style.display = 'block';
                } else {
                    for (const row of data) {
                        const resultRow = document.createElement('tr');
                        resultRow.innerHTML = `
                            <td>${row.track_id || 'N/A'}</td>
                            <td>${row.track_name || 'N/A'}</td>
                            <td>${row.track_artist || 'N/A'}</td>
                            <td>${row.lyrics || 'N/A'}</td>
                            <td>${row.playlist_name || 'N/A'}</td>
                            <td>${row.similitud || 'N/A'}</td>
                        `;
                        tablaBBody.appendChild(resultRow);
                    }
                    tablaBContainer.style.display = 'block';
                }
            }
        })
        .catch(error => {
            console.error('Error al realizar la búsqueda:', error);
            alert('Ocurrió un error al realizar la búsqueda. Por favor, intenta nuevamente.');
        });
    });

    // Funciones para mostrar mensajes y seleccionar tablas
    window.mostrarMensajePostgreSQL = function() {
        document.getElementById('database-message').style.display = 'block';
        document.getElementById('algorithm-message').style.display = 'none'; // Oculta el mensaje del algoritmo
        seleccionarTabla('tablaB'); // Llama a la función para mostrar la tabla de PostgreSQL
    }

    window.mostrarMensajeIndiceInvertido = function() {
        document.getElementById('algorithm-message').style.display = 'block';
        document.getElementById('database-message').style.display = 'none'; // Oculta el mensaje de la base de datos
        seleccionarTabla('tablaA'); // Llama a la función para mostrar la tabla de índice invertido
    }
});
