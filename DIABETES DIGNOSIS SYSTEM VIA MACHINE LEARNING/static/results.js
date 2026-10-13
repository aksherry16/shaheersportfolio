const predictButton = document.querySelector("#btn");
predictButton.addEventListener("pointerdown", async()=>{
    const formData = new FormData(document.getElementById('diabetesForm'));
    const inputData = {
        pregnancies: parseInt(formData.get('pregnancies')),
        glucose: parseInt(formData.get('glucose')),
        bloodPressure: parseInt(formData.get('bloodPressure')),
        skinThickness: parseInt(formData.get('skinThickness')),
        insulin: parseInt(formData.get('insulin')),
        bmi: parseFloat(formData.get('bmi')),
        dpf: parseFloat(formData.get('dpf')),
        age: parseInt(formData.get('age'))
    };

    console.log(inputData);
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            throw new Error('Server error: ' + response.status);
        }

        const result = await response.json();
        console.log(result.prediction);
        document.getElementById('result').textContent = `Prediction: ${result.prediction}`;
    } catch (error) {
        document.getElementById('result').textContent = `Error: ${error.message}`;
        console.error('Error:', error);
    }
})