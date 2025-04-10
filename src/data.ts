import * as tf from '@tensorflow/tfjs';

export function loadToyDataset(): { R: tf.Tensor2D; numUsers: number; numItems: number } {
  // A small 5x6 binary matrix: 5 users, 6 items
  // Rows = users, Columns = items
  const interactionData = [
    [1, 0, 0, 1, 0, 1], // user 0 interacted with item 0, 3, 5
    [0, 1, 0, 1, 0, 0], // user 1 interacted with item 1, 3
    [0, 1, 1, 0, 0, 0], // user 2 interacted with item 1, 2
    [1, 0, 1, 0, 0, 1], // user 3 interacted with item 0, 2, 5
    [0, 0, 0, 1, 1, 1], // user 4 interacted with item 3, 4, 5
  ];

  const R = tf.tensor2d(interactionData, [5, 6], 'float32');
  return { R, numUsers: 5, numItems: 6 };
}