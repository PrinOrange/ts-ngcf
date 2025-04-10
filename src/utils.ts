import * as tf from "@tensorflow/tfjs";
import { LEAKY_RELU_ALPHA } from "./consts";

export function leakyRelu(x: tf.Tensor): tf.Tensor {
	return tf.leakyRelu(x, LEAKY_RELU_ALPHA);
}

export function sigmoid(x: tf.Tensor): tf.Tensor {
	return tf.sigmoid(x);
}

export function dropout(tensor: tf.Tensor, dropRate: number): tf.Tensor {
	const keepProb = 1 - dropRate;
	const mask = tf.randomUniform(tensor.shape).greaterEqual(dropRate);
	return tensor.mul(mask).div(keepProb);
}
