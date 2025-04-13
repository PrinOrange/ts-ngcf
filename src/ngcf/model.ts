import * as tf from "@tensorflow/tfjs";

// NGCF model hyperparameters
export const EMBEDDING_DIM = 64; // Dimensionality of embeddings
export const NUM_LAYERS = 3; // Number of graph convolution layers
export const LEAKY_RELU_ALPHA = 0.2; // LeakyReLU negative slope
export const LEARNING_RATE = 0.001; // Learning rate for optimizer
export const REGULARIZATION = 1e-4; // L2 regularization coefficient
export const BATCH_SIZE = 1024; // Batch size for training
export const EPOCHS = 50; // Number of training epochs
export const NODE_DROPOUT_RATIO = 0.1; // Node dropout rate
export const MESSAGE_DROPOUT_RATIO = 0.1; // Message dropout rate

export class NGCF {
	numUsers: number;
	numItems: number;
	embeddingDim: number;
	layers: number;

	R: tf.Tensor2D; // User-item interaction matrix (binary)
	A: tf.Tensor2D; // Adjacency matrix A = [[0, R], [R^T, 0]]
	L: tf.Tensor2D; // Symmetrically normalized Laplacian

	E: tf.Variable<tf.Rank.R2>; // User and item embeddings
	W1: tf.Variable[] = []; // Weight matrices for first-order messages
	W2: tf.Variable[] = []; // Weight matrices for second-order messages
	optimizer: tf.Optimizer;

	constructor(
		numUsers: number,
		numItems: number,
		R: tf.Tensor2D,
		layers = NUM_LAYERS,
		embeddingDim = EMBEDDING_DIM,
	) {
		this.numUsers = numUsers;
		this.numItems = numItems;
		this.embeddingDim = embeddingDim;
		this.layers = layers;
		this.R = R;

		// Construct adjacency and normalized Laplacian matrix
		this.A = this.buildAdjacencyMatrix();
		this.L = this.buildLaplacian(this.A);

		// Initialize embeddings (users + items)
		this.E = tf.variable(tf.randomNormal([numUsers + numItems, embeddingDim]));

		// Initialize learnable parameters for each layer
		for (let l = 0; l < layers; l++) {
			this.W1.push(
				tf.variable(tf.randomNormal([embeddingDim, embeddingDim], 0, 0.1)),
			);
			this.W2.push(
				tf.variable(tf.randomNormal([embeddingDim, embeddingDim], 0, 0.1)),
			);
		}

		this.optimizer = tf.train.adam(LEARNING_RATE);
	}

	// Construct adjacency matrix A = [[0, R], [R^T, 0]]
	buildAdjacencyMatrix(): tf.Tensor2D {
		const zeroUU = tf.zeros([this.numUsers, this.numUsers]);
		const zeroII = tf.zeros([this.numItems, this.numItems]);
		const RT = this.R.transpose();
		return tf.concat(
			[tf.concat([zeroUU, this.R], 1), tf.concat([RT, zeroII], 1)],
			0,
		) as tf.Tensor2D;
	}

	// Compute the normalized Laplacian: L = D^{-1/2} A D^{-1/2}
	buildLaplacian(A: tf.Tensor2D): tf.Tensor2D {
		const degrees = tf.sum(A, 1); // Degree vector
		const D_inv_sqrt = tf.diag(degrees.pow(-0.5));
		return D_inv_sqrt.matMul(A).matMul(D_inv_sqrt) as tf.Tensor2D;
	}

	// Apply node dropout by masking nodes in the Laplacian matrix
	applyNodeDropout(L: tf.Tensor2D, dropRate: number): tf.Tensor2D {
		const numNodes = L.shape[0];
		const mask = tf
			.randomUniform([numNodes])
			.greaterEqual(dropRate) as tf.Tensor1D;
		const maskMat = tf.outerProduct(mask, mask); // Create a mask matrix
		return L.mul(maskMat);
	}

	// Perform forward propagation through NGCF layers
	propagate(): tf.Tensor2D {
		const embeddings: tf.Tensor[] = [this.E];
		const Ldrop =
			NODE_DROPOUT_RATIO > 0
				? this.applyNodeDropout(this.L, NODE_DROPOUT_RATIO)
				: this.L;

		for (let l = 0; l < this.layers; l++) {
			const E_prev = embeddings[embeddings.length - 1]!;
			const msg1 = tf.matMul(Ldrop, E_prev).matMul(this.W1[l]!); // First-order
			const msg2 = tf.matMul(Ldrop, E_prev.mul(E_prev)).matMul(this.W2[l]!); // Second-order
			let E_new = tf.add(msg1, msg2);

			if (MESSAGE_DROPOUT_RATIO > 0)
				E_new = tf.dropout(E_new, MESSAGE_DROPOUT_RATIO);

			E_new = tf.leakyRelu(E_new); // Non-linearity
			embeddings.push(E_new);
		}

		// Concatenate all layer outputs as final embedding
		return tf.concat(embeddings, 1) as tf.Tensor2D;
	}

	// Predict score for a given user and item pair
	predict(userId: number, itemId: number, finalE: tf.Tensor2D): tf.Scalar {
		const userVec = finalE.slice([userId, 0], [1, -1]);
		const itemVec = finalE.slice([this.numUsers + itemId, 0], [1, -1]);
		return tf.sum(tf.mul(userVec, itemVec));
	}

	// Sample training triplets (user, positive item, negative item)
	sampleTriplets(): number[][] {
		const triplets: number[][] = [];
		const Rarray = this.R.arraySync();

		for (let u = 0; u < this.numUsers; u++) {
			const posItems: number[] = [];
			for (let i = 0; i < this.numItems; i++) {
				if (Rarray[u]![i]! > 0) posItems.push(i);
			}
			for (const i of posItems) {
				let j = Math.floor(Math.random() * this.numItems);
				while (Rarray[u]![j]! > 0) {
					j = Math.floor(Math.random() * this.numItems);
				}
				triplets.push([u, i, j]);
			}
		}
		return triplets;
	}

	// Compute BPR loss + L2 regularization
	BPRLoss(triplets: number[][]): tf.Scalar {
		const finalE = this.propagate();
		const losses: tf.Tensor[] = [];

		for (const [u, i, j] of triplets) {
			const y_ui = this.predict(u!, i!, finalE);
			const y_uj = this.predict(u!, j!, finalE);
			const diff = tf.sub(y_ui, y_uj);
			const loss = tf.neg(tf.log(tf.sigmoid(diff))); // BPR loss
			losses.push(loss);
		}

		const meanLoss = tf.addN(losses).div(losses.length);

		// L2 regularization on embeddings and weights
		const reg = tf
			.addN([
				this.E.square().sum(),
				...this.W1.map((w) => w.square().sum()),
				...this.W2.map((w) => w.square().sum()),
			])
			.mul(REGULARIZATION);

		return meanLoss.add(reg);
	}

	// Train the NGCF model for a number of epochs
	async train(epochs = EPOCHS): Promise<void> {
		for (let epoch = 0; epoch < epochs; epoch++) {
			const triplets = this.sampleTriplets();

			for (let start = 0; start < triplets.length; start += BATCH_SIZE) {
				const batch = triplets.slice(start, start + BATCH_SIZE);
				this.optimizer.minimize(() => this.BPRLoss(batch));
			}

			console.log(`Epoch ${epoch + 1} completed.`);
		}
	}

	// Recommend top-K items for a given user
	recommend(userId: number, topK = 5): { itemId: number; score: number }[] {
		const finalE = this.propagate();
		const scores: { itemId: number; score: number }[] = [];
	
		for (let itemId = 0; itemId < this.numItems; itemId++) {
			const score = this.predict(userId, itemId, finalE).arraySync() as number;
			scores.push({ itemId, score });
		}
	
		return scores.sort((a, b) => b.score - a.score).slice(0, topK);
	}
}
