const input = document.getElementById('inputBox');
let currentSuggestion = '';

input.addEventListener('input', async e => {
  const text = e.target.value;
  const words = text.split(/\s+/);
  // Solo predecir si hay al menos 1 letra
  if (words.length > 0) {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text })
    });
    const { suggestion } = await response.json();
    const last = words[words.length - 1];
    if (suggestion && suggestion.startsWith(last)) {
      currentSuggestion = suggestion.substring(last.length);
      // Mostrar texto completado y seleccionar la sugerencia
      input.value = text + currentSuggestion;
      input.setSelectionRange(text.length, text.length + currentSuggestion.length);
    } else {
      currentSuggestion = '';
    }
  }
});

input.addEventListener('keydown', e => {
  if (e.key === 'Tab' && currentSuggestion) {
    e.preventDefault();
    // Confirmar la sugerencia
    const start = input.selectionStart;
    const end = input.selectionEnd;
    input.setSelectionRange(end, end);
    currentSuggestion = '';
  }
});