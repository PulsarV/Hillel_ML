# Machine Learning, Hillel IT School

Coursework for a machine learning course: nineteen assignments, from array manipulation to
generative models. Each one is a self contained Jupyter notebook with its data alongside it.

Most assignments follow the same pipeline: exploratory analysis, visual analysis, feature
engineering, train/validation/test split, baseline model, hyperparameter search, evaluation.

---

## Assignments

| # | Topic | Data | What was done | Result |
|---|-------|------|---------------|--------|
| [1](HW_1.NumPy) | NumPy | Iris | 14 array operations: inserting `nan` at guaranteed unique positions, conditional filtering, splitting and stacking, a custom column wise function | |
| [3](HW_3.Pandas) | Pandas | UCI Adult | 10 analytical queries: grouping, aggregation, shares within subsets | |
| [4](HW_4.Pandas.Visualisation) | Visualisation and feature engineering | NBA player statistics | 10 plots, each with a written takeaway; manual per game features; t-SNE of playing style | Plus FeatureTools: 25 features generated automatically |
| [5](HW_5.LinearRegression) | Linear regression | Wine Quality | Red and white merged with a `type` feature; Ridge, Lasso, ElasticNet | test R² 0.31; regularisation did not beat the baseline |
| [6](HW_6.MetricModel.Regression) | Metric model (kNN) | Abalone | Age regression plus sex classification, tuning `k` and the distance metric | R² 0.57, MAE 1.5 |
| [7](HW_7.DecisionTree) | Decision tree | Credit Approval | Tree with `GridSearchCV`, compared against logistic regression and kNN | test accuracy 0.833, AUC 0.959 |
| [8](HW_8.EnsembleMethods) | Ensembles | Auction Verification | Bagging, boosting and stacking on imbalanced data | f1 of the positive class: 0.971 (HistGB) |
| [9](HW_9.ClassImbalance) | Class imbalance | Glass | Eight balancing techniques compared through cross validation | balanced accuracy 0.887 versus 0.844 without balancing |
| [10](HW_10.Clustering) | Clustering | Iris | k-Means, Agglomerative, DBSCAN, Clustergram, silhouette | Metrics point to k=2, the actual species to k=3; the disagreement is shown explicitly |
| [11](HW_11.Clustering.DimensionReduction) | Clustering with PCA | South German Credit | 52 features compressed to 10, partitions compared before and after | ARI 0.992, the clusters survived compression |
| [12](HW_12.NeuralNetworkBasics.MNIST) | Neural networks (dense) | MNIST | Two parts: the spiral in TensorFlow Playground (6 layers of 7 neurons, L2) and an own network, 5 architectures with dropout and early stopping | Playground: test loss 0.005. MNIST: test 98.2% |
| [13](HW_13.NeuralNetworkBasics.Fashion-MNIST) | Convolutional networks | Fashion-MNIST | Optimiser comparison: SGD, momentum, Nesterov, RMSprop, Adam | test 93.3% |
| [14](HW_14.ComputerVision.OpenCV-Albumentations) | Computer vision | Own photograph | Load and display helpers, an augmentation pipeline | |
| [15](HW_15.IntroNLP) | NLP: bag of words, TF-IDF | IMDB, 50k reviews | Four vectorisation setups, plus what stop word removal does to negations | accuracy 0.905 |
| [16](HW_16.TextProcessing.spaCy-NER) | Named entity recognition | Own text | `uk_core_news_md`, EntityRuler, word vectors, Matcher, error analysis | 41 entities found, 12 more added by rules |
| [17](HW_17.Transformer.Translation) | Transformers, translation | Own text | opus-mt against NLLB, BLEU, round trip translation, error analysis | BLEU 41.75 versus 36.82 |
| [18](HW_18.ReinforcementLearning) | Reinforcement learning | LunarLander | Pretrained PPO from HuggingFace, state values, landing animation | mean reward 244.4, landed in 10 scenes out of 10 |
| [19](HW_19.DiffusionModels) | Generative models | | 10 images from custom prompts, with notes on what the model adds unasked | |

Numbering skips a step: there was never an assignment number 2, it is a numbering mix up in the
course itself.

---

## Stack

**Data and visualisation:** NumPy, pandas, matplotlib, seaborn, FeatureTools
**Classical ML:** scikit-learn, imbalanced-learn, clustergram
**Neural networks:** TensorFlow / Keras
**Computer vision:** OpenCV, Albumentations
**NLP:** spaCy, HuggingFace Transformers
**Reinforcement learning:** Gymnasium, Stable-Baselines3
**Generative models:** OpenAI API

---

## Running the notebooks

Most folders ship a `requirements.txt` pinned to the versions the work was done with:

```bash
cd HW_13.NeuralNetworkBasics.Fashion-MNIST
pip install -r requirements.txt
jupyter lab HW_13.ipynb
```

Data sits next to the notebook, except where the dataset ships with a library (MNIST,
Fashion-MNIST, Iris from `sklearn`) or is pulled from HuggingFace.

Assignment 19 calls a paid API and needs a key in a `.env` file. The generated images are stored
in the folder, so no rerun is needed just to look at them.
