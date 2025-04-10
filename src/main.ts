import { loadRandomDataset } from "./data";
import { NGCF } from "./model";

console.log("Creating random interaction dataset.");
const { R, numUsers, numItems } = loadRandomDataset();

console.log("Start loading NGCF model.");
const model = new NGCF(numUsers, numItems, R);

console.log("Training model.");
await model.train(10);

console.log("Generating recommendations.");
const recommendations = model.recommend(0, 5);
console.log("Top-3 Recommendation", recommendations);
