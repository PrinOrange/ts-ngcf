import { loadRandomDataset } from "./data";
import { NGCF } from "./model";

console.log("Creating random interaction dataset.");
const { R, numUsers, numItems } = loadRandomDataset();

console.log("Start loading NGCF model.");
const model = new NGCF(numUsers, numItems, R);

console.log("Training model.");
const trainingNum = 10;
await model.train(trainingNum);

console.log("Generating recommendations.");
const userId = 0;
const topK = 5;
const recommendations = model.recommend(userId, topK);
console.log(`Top-${topK} Recommendations for user ${userId}`, recommendations);
