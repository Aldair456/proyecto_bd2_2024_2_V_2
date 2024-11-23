document.getElementById('image-search-form').addEventListener('submit', function (event) {
    event.preventDefault();
  
    const imageFile = document.getElementById('image-upload').files[0];
    if (!imageFile) {
      alert('Por favor, sube una imagen para realizar la búsqueda.');
      return;
    }
  
    // Simulación de búsqueda de imágenes similares (lógica real para búsqueda aquí)
    displaySearchResults('index');
  });
  
  function seleccionarTabla(tabla) {
    if (tabla === 'tablaA') {
      document.getElementById('tablaA-container').style.display = 'block';
      document.getElementById('tablaB-container').style.display = 'none';
      displaySearchResults('index');
    } else if (tabla === 'tablaB') {
      document.getElementById('tablaA-container').style.display = 'none';
      document.getElementById('tablaB-container').style.display = 'block';
      displaySearchResults('postgres');
    }
  }
  
  function displaySearchResults(method) {
    const resultsContainer = method === 'index' ? document.getElementById('image-results-index') : document.getElementById('image-results-postgres');
    resultsContainer.innerHTML = ''; // Limpiar resultados anteriores
  
    // Simular resultados de búsqueda con imágenes
    for (let i = 0; i < 5; i++) {
      const resultItem = document.createElement('div');
      resultItem.classList.add('image-result-item');
  
      const img = document.createElement('img');
      img.src = 'https://via.placeholder.com/150'; // Placeholder de imagen similar
      img.alt = 'Imagen Similar';
      img.classList.add('result-image');
  
      const similarity = document.createElement('p');
      similarity.innerText = `Similitud: ${90 - i * 3}%`;
  
      resultItem.appendChild(img);
      resultItem.appendChild(similarity);
      resultsContainer.appendChild(resultItem);
    }
  }
  