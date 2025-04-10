import { loadRandomDataset } from "./ngcf/data";
import { NGCF } from "./ngcf/model";

// ==== Step 1 : Generate matrix of user-item interaction. ====
console.log("Creating random interaction dataset.");
const numUsers = 50;
const numItems = 50;
const minInteractions = 5;
const maxInteractions = 10;
const R = loadRandomDataset(
	numUsers,
	numItems,
	minInteractions,
	maxInteractions,
);

// ==== Step 2 : Load models and input matrix ====
console.log("Start loading NGCF model.");
const model = new NGCF(numUsers, numItems, R);

// ==== Step 3 : Training model through propagations ====
console.log("Training model.");
const trainingNum = 20;
await model.train(trainingNum);

// ==== Step 4 : Generate recommendations according to prediction ====
console.log("Generating recommendations.");
const userId = 10;
const topK = 5;
const recommendations = model.recommend(userId, topK);
console.log(`Top-${topK} Recommendations for user ${userId}`, recommendations);
