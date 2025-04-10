import { loadRandomInteractionDataset } from "./ngcf/data";
import { NGCF } from "./ngcf/model";
import chalk from "chalk";

// ==== Step 1 : Generate matrix of user-item interaction. ====
console.log(chalk.cyan("Creating random interaction dataset."));
const numUsers = 100;
const numItems = 100;
const minInteractions = 10;
const maxInteractions = 15;
const R = loadRandomInteractionDataset(
	numUsers,
	numItems,
	minInteractions,
	maxInteractions,
);

// ==== Step 2 : Load models and input matrix. ====
console.log(chalk.cyan("Start loading NGCF model."));
const model = new NGCF(numUsers, numItems, R);

// ==== Step 3 : Training model through propagations. ====
console.log(chalk.cyan("Training model."));
const trainingNum = 10;
await model.train(trainingNum);

// ==== Step 4 : Generate recommendations according to prediction. ====
console.log(chalk.cyan("Generating recommendations."));
const userId = 10;
const topK = 5;
const recommendations = model.recommend(userId, topK);
console.log(`Top-${topK} Recommendations for user ${userId}`, recommendations);
