// index.js
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

async function runInference() {
    // This engine supports file:// URLs natively
    const modelPath = 'file://' + path.join(__dirname, 'model', 'model.json');
    const model = await tf.loadLayersModel(modelPath);

    const input = tf.zeros([1, 224, 224, 3]);
    const predictions = model.predict(input);
    predictions.print();

    // Cleanup
    input.dispose();
    predictions.dispose();
}

runInference().catch(console.error);
