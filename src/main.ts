import { loadToyDataset } from "./data";
import { NGCF } from "./model";

const { R, numUsers, numItems } = loadToyDataset();
const model = new NGCF(numUsers, numItems, R);
await model.train(10); // 简单训练一下
const recommendations = model.recommend(0, 3); // 给 user 0 推荐 top-3

console.log('Top-3 推荐：', recommendations);
