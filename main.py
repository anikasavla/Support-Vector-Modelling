import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

url = "https://philchodrow.github.io/PIC16A/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)

train, test = train_test_split(penguins, test_size=0.2) #splits the data into a training and testing set

def clean_prep(df, str_feature, num_features, target_col="Species", split_Xy=True):
    """
    Cleans and prepares the DataFrame; handles missing values, encodes qualitative features, and optionally splits
    into X and y.

    Args:
        df (pandas.DataFrame): The input DataFrame to process.
        str_feature (list of str): List of qualitative columns to encode.
        num_features (list of str): List of quantitative columns to include in the features.
        target_col (str): Name of the target column to encode. Defaults to "Species".
        split_Xy (bool): Whether to split the DataFrame into X (features) and y (target). Defaults to True.

    Returns:
        pandas.DataFrame or tuple: Processed DataFrame if split_Xy is False, or (X, y) tuple if split_Xy is True.
    """
    df = df.dropna(subset = str_feature + num_features + [target_col]) #drop NaNs in specified columns
    if "Sex" in str_feature: #only take rows with male or female in the sex column if sex is one of the features
        df = df[df["Sex"].isin(["MALE", "FEMALE"])]

    #encode the qualitative feature values into numbers
    df_label = df.copy()
    le = preprocessing.LabelEncoder()
    df_label[target_col] = le.fit_transform(df_label[target_col])
    for col in str_feature: #encode for every given quantitative feature
        df_label[col] = le.fit_transform(df_label[col])
    features = str_feature + num_features #put together the features to be used

    if split_Xy: #split into and return features and targets if told to
        X = df_label[features]
        y = df_label[target_col]
        return X, y
    return df_label

#create a separate data frame with the columns Species, Culmen Length(mm) and Sex from the training data when the sex is valid and sorts by Species
train_clean3 = clean(train, ["Species", "Culmen Length (mm)", "Culmen Depth (mm)", "Sex"], split_Xy = False).sort_values(by="Species")
train_clean3["Species"] = train["Species"].str.split().str[0]

#creates a pink-purple palette for the boxplots and sets the styles of the graphs
pink=sns.blend_palette(("pink", "purple"), n_colors=3)
sns.set_theme(style='whitegrid',font_scale=1.0)

#creates the figure for the two subplots which share the same y-axis and x-axis scales
fig, ax = plt.subplots(2, 2, figsize=(7, 8), sharex = True, sharey = 'row')

#function which creates the graphs
def graph(y_var, sex, row, col):
  '''
  Plots boxplots showing the statistics of Culmen Lengths and Depths of penguin Species by Sex.

    Args:
        y_var (column name in dataframe): Whether the boxplots are for Culmen Length or Depth
        sex (str): The sex of the penguins to plot.
        row (int): The axes row to plot the chart on.
        col (int): The axes colum to plot the chart on.
    Returns:
        None: Displays boxplots on the provided axes.
  '''
  sns.boxplot(data=train_clean3[train_clean3["Sex"] == sex], x="Species", y=y_var, hue="Species", palette = pink, width=0.5, ax=ax[row][col], legend = True)
  if col == 0:
    ax[row][col].set(ylabel = y_var)
  if row == 1:
    ax[row][col].set(xlabel = "Species")
  else:
    ax[row][col].set(title = sex.lower().capitalize())
  ax[row][col].legend(title = "Species")

#plotting
graph("Culmen Length (mm)", "MALE", 0, 0)
graph("Culmen Length (mm)", "FEMALE", 0, 1)
graph("Culmen Depth (mm)", "MALE", 1, 0)
graph("Culmen Depth (mm)", "FEMALE", 1, 1)

plt.tight_layout()

train, test = train_test_split(penguins, test_size=0.2)
X_train, y_train = clean_prep(train, ["Sex"], ["Culmen Depth (mm)", "Culmen Length (mm)"])
X_test, y_test = clean_prep(test, ["Sex"], ["Culmen Depth (mm)", "Culmen Length (mm)"])

#create support vector machine model
svm_model = SVC()
#creates lists for potential c and gamma values
C_values = [0.1, 1, 10, 100, 1000]
gamma_values = [1, 0.1, 0.01, 0.001, 0.0001]

#intializes best values
best_C = 0.1
best_gamma = 1
best_cvs = -np.inf

#find the parameters and updates best values for highest cross_val_score producing parameters
for C in C_values:
  for gamma in gamma_values:
    svm_model.set_params(C=C, gamma=gamma)
    new_cvs = cross_val_score(svm_model, X_train, y_train, cv=5).mean()
    if new_cvs > best_cvs:
      best_cvs = new_cvs
      best_C = C
      best_gamma = gamma

print(f"best c value: {best_C}, best gamma value: {best_gamma}, best cross validation score: {best_cvs}")

svm_model = SVC(C = best_C, gamma = best_gamma)
#fit the model to the training data
svm_model.fit(X_train, y_train)
#scores for the model when using it on the training and testing data
training_score = svm_model.score(X_train, y_train)
test_score = svm_model.score(X_test, y_test)
print(f"Training score: {training_score}, Testing score: {test_score}")

y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")

def plot_regions_by_sex(model, X, y):
  """
    Plots the side by side decision regions for both sexes, using Culmen Depth and Culmen Length to visualize species classification boundaries.

    Args:
        c (sklearn classifier/model): The trained classifier used for predictions.
        X (pandas.DataFrame): Feature DataFrame containing the Sex, Culmen Depth, and Culmen Length.
        y (pandas.Series): Target Series containing the species labels.

    Returns:
        None: Displays a plot of the decision regions with the data points for both sexes using given model and test data.
    """
  sex_name = {0: "Male", 1: "Female"}
  md = X[X["Sex"] == 0]
  fd = X[X["Sex"] == 1]

  #create the global min and max values for both Sex graphs to encompass all the data
  x0_min, x0_max = min(md['Culmen Depth (mm)'].min(), fd['Culmen Depth (mm)'].min()), max(md['Culmen Depth (mm)'].max(), fd['Culmen Depth (mm)'].max())
  x1_min, x1_max = min(md['Culmen Length (mm)'].min(), fd['Culmen Length (mm)'].min()), max(md['Culmen Length (mm)'].max(), fd['Culmen Length (mm)'].max())

  #creates the figure
  fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex= True, sharey = True)

  #parses through both Sex values to compare the data side by side
  for i in range(2):
      X_sex = X[X["Sex"] == i]
      y_sex = y[X["Sex"] == i]

      # Extract the features for plotting
      x0 = X_sex['Culmen Depth (mm)']
      x1 = X_sex['Culmen Length (mm)']

      # create a grid
      grid_x = np.linspace(x0_min-1,x0_max+1,101)
      grid_y = np.linspace(x1_min-1,x1_max+1,101)
      xx, yy = np.meshgrid(grid_x, grid_y)

      XX = xx.ravel()
      YY = yy.ravel()
      sex_column = np.full(XX.shape, i)  # Island column is fixed
      input_data = pd.DataFrame({"Sex": sex_column, "Culmen Depth (mm)": XX, "Culmen Length (mm)": YY })

      #use the model to make predictions
      p = model.predict(input_data)
      p = p.reshape(xx.shape)

      # use contour plot to visualize the predictions
      #create color map based on species for contour map
      species_labels = ["Adelie", "Chinstrap", "Gentoo"]
      species_colors = ["blue", "red", "green"]
      cmap = ListedColormap(species_colors)  # Color map for species
      ax[i].contourf(xx, yy, p, cmap=cmap, alpha=0.2)

      # scatter data points and assign same colors and contour map
      s_colors = [species_colors[species] for species in y_sex]
      ax[i].scatter(x0, x1, c=s_colors)
      ax[i].set(xlabel = "Culmen Depth (mm)" , ylabel = "Culmen Length (mm)", title = f"Decision Regions for Island {sex_name[i]}")

      #creates legend for the sex graphs
      unique_species = [0, 1, 2]
      legend_handles = [Patch(color=species_colors[species_id], label=species_labels[species_id]) for species_id in unique_species]
      ax[i].legend(handles=legend_handles, title="Species")

  #formatting
  plt.tight_layout()

plot_regions_by_sex(svm_model, X_test, y_test)
