import { loadSyntheticDataset } from "./data";
import { NGCF } from "./model";

const { R, numUsers, numItems } = loadSyntheticDataset();
const model = new NGCF(numUsers, numItems, R);
await model.train(10);
const recommendations = model.recommend(0, 5);

console.log('Top-3 Recommendation', recommendations);
