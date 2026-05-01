## EDA and feature engineering instructions

### Step 1: Pull from remote repo
As you can see, I've made some changes to the remote repo, including adding these instructions and creating the cleaned dataset. You'll need to update your local repos with these changes.

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


### Step 2: Create a new local branch
Create a local branch to do your EDA and feature engineering.
``` shell
git checkout -b yourname-eda
```

Check to make sure you're on the newly-created branch:
``` shell
git branch
```


### Step 3: Create your file (and install any dependencies)
Create a Python file in the news_headline_categorizer folder. Name it yourname-eda and don't forget the .py extension at the end.

You'll mainly be using pandas for data wrangling and either matplotlib.pyplot or seaborn (maybe both) for visualization. Import these libraries at the top of your file and run them. You may need to pip install them.

You may also need to install the Python extension in VS Code if it's your first time coding in Python in the application.


### Step 4: Load the data
Turn the .csv file into a DataFrame. Your Python file should look something like this:

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/CleanedNews.csv')
```

You can make sure the data was loaded correctly by running df.head() to show the top 5 rows or df.shape to get the number of rows and columns of the DataFrame.


### Step 5: Start exploring the data!
Start on your [tasks](https://docs.google.com/document/d/1dqou90RACXlTGCMI1P5ABOtWaYsKlpMVF-qQ8WduHE4/edit?tab=t.0)! Message me with any questions or issues that you have.


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
