// const fs = require('fs');
// const path = require('path');
// require() só funciona server-side (node.js). Esse javascript ta sendo executado no cliente (ou seja, não tem acesso ao filesystem)
// https://www.w3schools.com/js/js_modules.asp
// O fine-tuning com os dados da base de dados já coletada só funcionaria num ambiente nodejs

// Notice there is no 'import' statement. 'mobilenet' and 'tf' is
// available on the index-page because of the script tag above.
let modelo;

const STATUS = document.getElementById('status');
STATUS.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
const INPUT = document.getElementById("input-news");
const PREDICT_BUTTON = document.getElementById('bt-predict');
const RESULT = document.getElementById("resultado")
const CLASS_NAMES = ["REAL", "FAKE"];

const loaderContainer = document.querySelector('.loader-container');
const MSG_RESULTADO = document.getElementById("msg-resultado")

PREDICT_BUTTON.addEventListener('click', predict);
INPUT.addEventListener("keypress", function (event) {
	if (event.key === "Enter") {
		if (INPUT.value.length == 0) {
			alert("Write something");
			return;
		}
		event.preventDefault();
		predict();
	}
});

// Chamar função async pra considerar o gif de loading
// (async() => {
// 	loaderContainer.style.display = 'block';
// 	await loadModel();
// 	loaderContainer.style.display = 'none';
// })();

loadModel();

async function predict() {
	let inputText = INPUT.value
	input = [inputText]
	console.log(input)
	const resultado = modelo.predict([inputText]); // O resultado é um tensor do tensorflow
	prediction = resultado.squeeze();
	let highestIndex = prediction.argMax().arraySync(); // arraySync transforma o tensor em um array javascript
	console.log("highestindex: ", highestIndex)

	let probabilidades = tf.softmax(prediction) // tensor do tensorflow
	probabilidades = probabilidades.arraySync() // array javascript
	console.log("Probabilidades: ", probabilidades)

	RESULT.scrollIntoView({behavior: 'smooth'});
	MSG_RESULTADO.innerText = "According to the model, this news article is "
	RESULT.innerText = CLASS_NAMES[highestIndex] + ' (confidence of ' + (probabilidades[highestIndex]*100).toFixed(1) + '%)';
}

async function loadModel() {
	console.log('Loading model...')
	const URL =
	'https://raw.githubusercontent.com/Jp9910/NLP-Project/main/models/bert_classifierJS/content/tfjs_model/';
	//https://js.tensorflow.org/api/latest/#loadGraphModel
	
	modelo = await tf.loadGraphModel(URL, {fromTFHub: true});
	STATUS.innerText = 'Model loaded';
	
	console.log('Model loaded...')
	// Warm up the model by passing zeros through it once.
	tf.tidy(function () {
	  let answer = modelo.predict(["Aquecendo o modelo!","Aquecendo o modelo!","Aquecendo o modelo!"]);
	  console.log(answer);
	});
}
