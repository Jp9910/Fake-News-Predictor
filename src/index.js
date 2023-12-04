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

const PREDICT_BUTTON = document.getElementById('bt-predict');
const CANVAS = document.getElementById("canvas")
const RESULT = document.getElementById("resultado")
const MOBILE_NET_INPUT_WIDTH = 512;
const MOBILE_NET_INPUT_HEIGHT = 512;
const CLASS_NAMES = ["aurelion sol", "kindred", "teemo"];
const ctx = CANVAS.getContext("2d");
const MSG_RESULTADO = document.getElementById("msg-resultado")

PREDICT_BUTTON.addEventListener('click', predict);

const loaderContainer = document.querySelector('.loader-container');

// Chamar função async pra considerar o gif de loading
// (async() => {
// 	loaderContainer.style.display = 'block';
// 	await loadModel();
// 	loaderContainer.style.display = 'none';
// })();

loadModel();
setupCanvas();
setupCanvasNavbar();

async function predict() {
	let imageAsTensor = tf.browser.fromPixels(CANVAS); // Parameters:
													// 1 - pixels (ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) 
													//	The input image to construct the tensor from. The supported image types are all 4-channel.
													// 2 - numChannels (number) The number of channels of the output tensor. A numChannels 
												 	//	value less than 4 allows you to ignore channels. Defaults to 3 (ignores alpha channel of input image)

	let resized = tf.image.resizeBilinear(imageAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
	let normalized = resized.div(255); // dividir por 255
	const resultado = modelo.predict(normalized.expandDims()); // O resultado é um tensor do tensorflow

	let prediction = resultado.squeeze(); //Removes dimensions of size 1 from the shape of a tensor.

	let highestIndex = prediction.argMax().arraySync(); // arraySync transforma o tensor em um array javascript
	console.log("highestindex: ", highestIndex)

	let probabilidades = tf.softmax(prediction) // tensor do tensorflow
	probabilidades = probabilidades.arraySync() // array javascript
	console.log("Probabilidades: ", probabilidades)

	RESULT.scrollIntoView({behavior: 'smooth'});
	MSG_RESULTADO.innerText = "De acordo com nossa IA, você desenhou:"
	RESULT.innerText = CLASS_NAMES[highestIndex] + ' (confiança de ' + (probabilidades[highestIndex]*100).toFixed(1) + '%)';
}

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadModel() {
	const URL = 
		'https://raw.githubusercontent.com/Jp9910/Projeto-AM/main/models/tensorflowjs/';

	//https://js.tensorflow.org/api/latest/#loadGraphModel
	modelo = await tf.loadGraphModel(URL, {fromTFHub: true});
	STATUS.innerText = 'Modelo carregado com sucesso!';

	// Warm up the model by passing zeros through it once.
	tf.tidy(function () {
	  let answer = modelo.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
	  console.log(answer);
	});
}
