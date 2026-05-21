## Finalizing your model
Continue working on the branch for your model! If you run:
``` shell
git branch
```
you should see the branch for the model you worked on last week.

In your terminal, run
```shell
git pull origin main
```
in order to see my updated logistic-regressor.py file, which you'll use as a reference to finish up your model. All the extra code you need to add (besides the new imports) starts at line 90, and the only parts that you should change are the hyperparameters and lists of values in param_grid (lines 94-98 in my file). Code from last week shouldn't be modified.

### Step 1: Tune your model
First, we'll need to add some extra imports for fine-tuning and visualization. At the top of the logistic-regressor.py file, you can see what to import:
```python
#IMPORT STRATIFIEDKFOLD, GRIDSEARCHCV, SEABORN, MATPLOTLIB
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
```

Each model has hyperparameters whose values affect how well it predicts the target variable.
In lines 94-98 in logistic-regressor.py (shown below), I list potential hyperparameter values in param_grid. In your own file, replace the hyperparameters and their corresponding lists of values with the hyperparameters specific to your model.
```python
# LIST POSSIBLE VALUES FOR HYPERPARAMETERS SPECIFIC TO YOUR MODEL
param_grid = {
    "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    "class_weight": [None, "balanced"]
}
```

### Step 2: Measure the best model's performance.
GridSearchCV tests which parameters in your param_grid work best. When fine-tuning your model, GridSearchCV may take several minutes to run depending on your model and parameter grid (because it essentially has to train and evaluate every possible combination of hyperparameters for the model!). For me, it took a few minutes, but if you're waiting for much longer, you can reduce the number of parameters to test in your param_grid.

We calculated the accuracy and macro F1 scores of our baseline model last week. For the final model, we'll be recalculating these values to see how our model has improved. 

Note down these values!

### Step 3: Visualize the best model's performance.
We'll each be creating two visualizations for our models.
1. A heatmap of the confusion matrix. This helps us visualize which categories the model predicts correctly and which categories get confused.
2. A bar chart of F1 scores per category. This helps us visualize how well the model performs for each news category.


## Some notes
Make sure to save your changes by going to File > Save All. You can also turn on autosave.

Here's the sequence of commands for committing your changes. Each commit should be a version of your program that runs properly!
1. Check that you have saved changes to commit.
``` shell
git status
```
2. Add your changes to the staging area.
``` shell
git add .
```
3. Check that you have added your changes to the staging area.
``` shell
git status
```
4. Commit your staged changes.
``` shell
git commit -m "explain what changes you made in these quotation marks"
```

Here's the sequence of commands to push your committed changes.
1. Make sure you have committed all of your changes first. If not, go through the above sequence of commands.
2. Make sure you are on the right branch.
``` shell
git branch
```
If you're not, run:
```shell
git checkout your-branch-name
```
3. Update your branch with the main branch in the remote repo. 
``` shell
git pull origin main
```

4. Push your changes. If it's your first time pushing from this branch, run:
``` shell
git push -u origin your-branch-name
```
Otherwise, do:
``` shell
git push
```
