## Initial model instructions

### Step 1: Choose a model.
Choose the model that you want to work on [here](https://docs.google.com/document/d/1b0MIjiUsNkucRr9dzQmKuuD5JdMtrOxLesxPLw9loNU/edit?tab=t.0).

Read through the resources to learn more about the model. Feel free to find more articles/videos. Next meeting, we'll teach each other about the model you trained. 


### Step 2: Pull from remote repo
Update your local repo to get the dataset with the new feature columns you guys created as well as the example model training script (elasticnet.py).

First, after you open the project in VS Code, make sure you are on the main branch:
``` shell
git branch
```

If you see main in green with an asterisk to the left of it, great! Stay where you are. Otherwise, switch to the main branch:
``` shell
git checkout main
```

Now, pull from the remote repo:
``` shell
git pull
```

Your local repo should be up to date now!


### Step 3: Create a new local branch
Create a local branch to train your model.
``` shell
git checkout -b model-name
```

Check to make sure you're on the newly-created branch:
``` shell
git branch
```


### Step 4: Create your file (and install any dependencies)
Create a Python file in the src/models folder. Name it your-model-name and don't forget the .py extension at the end.

You'll be using modules from the scikit-learn library.

### Step 5: Read through the example model
I trained a logistic regression model on our data. You likely won't need to change any of the code except for replacing LogisticRegression with the name of your model and changing the model hyperparameters.

However, please read through the code and understand the steps of training a model: (1) preparing the data, (2) splitting the data into train/test sets, (3) creating the model pipelines, and (4) performing cross validation.

Also, learn more about what each scikit-learn class and function does by reading its documentation or other articles. 


### Step 6: Train your model!
Again, you won't be changing much from the example code, but rather than copy and pasting, try typing each block out to better understand what it's doing.

You will need to change the code to import your model, i.e. this part of the code:
```python
#IMPORT YOUR MODEL HERE
from sklearn.linear_model import LogisticRegression
```

Also, you will need to decide the baseline hyperparameters for your model, i.e. this part of the code:
```python
#instantiate model
#REPLACE WITH YOUR MODEL AND BASELINE HYPERPARAMETERS! Keep random_state = 42 for reproducibility.
model = LogisticRegression(
    max_iter=3000,
    C=2.0,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)
```

Different models have different hyperparameters, so you will need to do some research into what they are and how they affect the model's performance. Next week, we'll tune our models' hyperparameters to find the combination that results in the most performant model. 

Note down your model's score!


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
3. Update your branch with the main branch in the remote repo. For the most part, there won't be any changes, but it's a good habit to develop when working in a team.
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