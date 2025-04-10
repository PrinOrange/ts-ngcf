import { loadRandomInteractionDataset } from "./ngcf/data";
import { NGCF } from "./ngcf/model";

// ==== Step 1 : Generate matrix of user-item interaction. ====
console.log("Creating random interaction dataset.");
const numUsers = 100;
const numItems = 100;
const minInteractions = 10;
const maxInteractions = 20;
const R = loadRandomInteractionDataset(
	numUsers,
	numItems,
	minInteractions,
	maxInteractions,
);

// ==== Step 2 : Load models and input matrix. ====
console.log("Start loading NGCF model.");
const model = new NGCF(numUsers, numItems, R);

// ==== Step 3 : Training model through propagations. ====
console.log("Training model.");
const trainingNum = 20;
await model.train(trainingNum);

// ==== Step 4 : Generate recommendations according to prediction. ====
console.log("Generating recommendations.");
const userId = 10;
const topK = 5;
const recommendations = model.recommend(userId, topK);
console.log(`Top-${topK} Recommendations for user ${userId}`, recommendations);
