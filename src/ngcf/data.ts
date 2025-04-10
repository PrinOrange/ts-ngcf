import * as tf from "@tensorflow/tfjs";

export function loadRandomDataset(
	numUsers: number,
	numItems: number,
	minInteractions: number,
	maxInteractions: number,
): tf.Tensor2D {
	const interactionData: number[][] = [];

	for (let u = 0; u < numUsers; u++) {
		const row = new Array(numItems).fill(0);
		const numInteractions =
			Math.floor(Math.random() * (maxInteractions - minInteractions + 1)) +
			minInteractions;

		const itemIndices = new Set<number>();
		while (itemIndices.size < numInteractions) {
			itemIndices.add(Math.floor(Math.random() * numItems));
		}

		for (const i of itemIndices) {
			row[i] = 1;
		}

		interactionData.push(row);
	}

	const R = tf.tensor2d(interactionData, [numUsers, numItems], "float32");
	return R;
}
