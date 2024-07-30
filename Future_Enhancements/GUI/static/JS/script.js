document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');
    const uploadedImage = document.getElementById('uploaded-image');

    formData.append('image', fileField.files[0]);

    // Display the uploaded image
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = 'block';
    }
    reader.readAsDataURL(fileField.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    const resultDiv = document.getElementById('prediction-result');

    if (result.error) {
        resultDiv.innerHTML = `<p style="color: red;">${result.error}</p>`;
    } else {
        const prediction = result.prediction[0];
        resultDiv.innerHTML = `
            <p>Prediction: ${prediction}</p>
            <p class="explanation">The predicted geographical coordinate is ${prediction}. This value represents the latitude or longitude based on the trained model's output. It provides an estimate of the location based on the features extracted from the uploaded image.</p>
        `;
    }

    resultDiv.scrollIntoView({ behavior: 'smooth' });
});
