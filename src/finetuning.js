/**
 * _____________________________________________________
 * Fine-tuning
 * _____________________________________________________
 */



// Define the new model head
function getModeloTreinamento() {
	let model = tf.sequential();
	
	// Add a dense layer as the input layer to this model. This has an input
	// shape of 1024 as the outputs from the MobileNet v3 features are of this 
	// size. This layer has 128 neurons that use the ReLU activation function.
	model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
	
	// Output layer (softmax para classificação)
	model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));
	
	// Print resumo do modelo no console
	model.summary();
	
	// Compile the model with the defined optimizer and specify a loss function to use.
	model.compile({
		optimizer: 'adam', //tf.train.adam(0.0001)
		loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
		metrics: ['accuracy']  
	});

	return model
}

async function train() {
	console.log("treinando")
	let model = getModeloTreinamento();
	data = getData();
	trainingDataInputs = data.getTrainData().images
	trainingDataOutputs = data.getTrainData().labels
	console.log("Training Images (Shape): " + trainImages.shape);
	console.log("Training Labels (Shape): " + trainLabels.shape);
	tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
	let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
	let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
	let inputsAsTensor = tf.stack(trainingDataInputs);

	// Unlike in Python, Dataset-based training is done through a dedicated method, namely fitDataset.
	// The fit() method is only for tensor-based model training.
	let results = 
		await model.fit(
			inputsAsTensor,
			oneHotOutputs,
			{
				shuffle: true, batchSize: 5, epochs: 10, 
				callbacks: {onEpochEnd: logProgress} 
			}
		);
	outputsAsTensor.dispose();
	oneHotOutputs.dispose();
	inputsAsTensor.dispose();
}

function logProgress(epoch, logs) {
	console.log('Data for epoch ' + epoch, logs);
}


/**
 * _______________________________
 * Load data
 * _______________________________
 */

//https://medium.com/analytics-vidhya/classification-model-on-custom-dataset-using-tensorflow-js-9458da5f2301
function loadImages(dataDir) {
	const images = [];
	const labels = [];
	
	var champions = fs.readdirSync(dataDir);
	console.log(champions.length)
	for (let j = 0; j < champions.length; j++) {
		var files = fs.readdirSync(dataDir+champions[j].toLocaleLowerCase())
		for (let i = 0; i < files.length; i++) {
			if (!files[i].toLocaleLowerCase().endsWith(".jpg")) {
				continue;
			}
	
			var filePath = path.join(dataDir, files[i]);
			console.log("filePath: "+filePath)
	
			var buffer = fs.readFileSync(filePath);
			let imageFeatures = tf.tidy(function() {
				let imageAsTensor = tf.browser.fromPixels(buffer);
				let resized = tf.image.resizeBilinear(imageAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true); //Redimensionar a imagem para 224x224.
				let normalized = resized.div(255); // dividir por 255
				return mobilenet.predict(normalized.expandDims()).squeeze();
			});
	
			images.push(imageFeatures);
			
			// here we assume every folder has file with name n_flower.png
			var asol = champions[j].toLocaleLowerCase().endsWith("aurelion sol");
			var kindred = champions[j].toLocaleLowerCase().endsWith("kindred");
			var teemo = champions[j].toLocaleLowerCase().endsWith("teemo");
		
			
			if (asol == true) {
				labels.push(1)
			} else if (kindred == true) {
				labels.push(2)
			} else {
				labels.push(3)
			}
	
		}
	}
	console.log('Labels are');
	console.log(labels);
	return [images, labels];
}

/**
 * Carregar dados para fine-tuning
 **/
function getData() {
	// Buscar arquivos
	// Fazer image augmentation?

	const TRAIN_IMAGES_DIR = '../scrapping/images/train';
	const VALIDATION_IMAGES_DIR = '../scrapping/images/validation';

	/** Helper class to handle loading training and validation data. */
	class LeagueDataset {
		constructor() {
			this.trainData = [];
			this.validationData = [];
			this.loadData();
		}

		/** Loads training and validation data. */
		loadData() {
			console.log('Loading images...');
			this.trainData = loadImages(TRAIN_IMAGES_DIR);
			this.validationData = loadImages(VALIDATION_IMAGES_DIR);
			console.log('Images loaded successfully.')
		}

		getTrainData() {
			return {
				images: tf.concat(this.trainData[0]),
				labels: tf.oneHot(tf.tensor1d(this.trainData[1], 'int32'), 3).toFloat() // here 5 is class
			}
		}

		getValidationData() {
			return {
				images: tf.concat(this.validationData[0]),
				labels: tf.oneHot(tf.tensor1d(this.validationData[1], 'int32'), 3).toFloat()
			}
		}
	}

	//module.exports = new LeagueDataset();
	return new LeagueDataset();
}