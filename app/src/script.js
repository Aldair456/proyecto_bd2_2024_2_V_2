const searchButton = document.getElementById('search-button');
searchButton.addEventListener('click', () => {
    const query = document.getElementById('query-input').value;
    const topk = document.getElementById('topk-input').value;
    const url = `/buscar?query=${query}&topk=${topk}`;
    const tablaResultados = document.getElementById('tabla-resultados');
    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState === 4 && this.status === 200) {
            const response = JSON.parse(this.responseText);
            // Use the response to fill the table with the search results
        }
    };
    xhr.open('GET', url, true);
    xhr.send();
});