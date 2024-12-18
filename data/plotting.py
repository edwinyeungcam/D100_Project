import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_binary(df, column, class_labels=None):
    """
    Visualizes the distribution of a binary variable
    There is an option to rename class 0 and class 1 for easier visualization

    """
    # Count the occurrences of each class
    class_counts = df[column].value_counts()
    
    if class_labels is None:
        class_labels = ['Class 0', 'Class 1']
    
    # Plot the distribution
    #plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title(f'Distribution of Target Variable: {column}', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks([0, 1], class_labels, fontsize=10)
    plt.show()
    
    # Print percentages for each class
    class_percentages = class_counts / len(df) * 100
    print(f"{class_labels[0]}: {class_percentages[0]:.2f}%")
    print(f"{class_labels[1]}: {class_percentages[1]:.2f}%")

def boxplot_for_categorical_target(df, numerical_feature, target, hue=None):
    """
    Creates a boxplot for a numerical feature vs a categorical target variable with an optional hue.
    """
    sns.boxplot(x=target, y=numerical_feature, data=df, hue=hue)
    plt.title(f'Boxplot of {numerical_feature} by {target}')
    plt.xlabel(target)
    plt.ylabel(numerical_feature)
    if hue:
        plt.legend(title=hue)
    plt.show()

def histogram_for_categorical_target(df, numerical_feature, target, hue=None):
    """
    Creates histograms for a numerical feature grouped by each class of the categorical target variable with an optional hue.
    """
    sns.histplot(data=df, x=numerical_feature, hue=hue if hue else target, kde=False, multiple='stack')
    plt.title(f'Histogram of {numerical_feature} by {target}')
    plt.xlabel(numerical_feature)
    plt.ylabel('Count')
    plt.show()

def count_plot_categorical_target(df, categorical_feature, target, hue=None):
    """
    Creates a count plot for a categorical feature grouped by the classes of a categorical target variable with an optional hue.
    """
    sns.countplot(x=categorical_feature, hue=hue if hue else target, data=df)
    plt.title(f'Count Plot of {categorical_feature} by {target}')
    plt.xlabel(categorical_feature)
    plt.ylabel('Count')
    if hue:
        plt.legend(title=hue)
    plt.show()

    proportions = (
        df.groupby(categorical_feature)[target]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    # Display proportions
    print("\nProportion Table:")
    print(proportions)

def add_combined_column(df: pd.DataFrame, col1: str, col2: str, value1, value2, new_col_name: str) -> pd.DataFrame:
    """
    Create a copy of the DataFrame and add a new column that returns 1 
    only if the specified values in the two columns are matched.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        value1: The value to check in the first column.
        value2: The value to check in the second column.
        new_col_name (str): The name of the new column to be added.
        
    Returns:
        pd.DataFrame: A copy of the input DataFrame with the new column added.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Add the new column based on the condition
    df_copy[new_col_name] = (df_copy[col1] == value1) & (df_copy[col2] == value2)
    
    # Convert boolean values to integers (1 for True, 0 for False)
    df_copy[new_col_name] = df_copy[new_col_name].astype(int)
    
    return df_copy

def kde_plot(data, column, title="KDE Plot", ylabel="Density", color="blue"):
    """
    Generates a KDE plot for a specified numerical column in a dataset.
    """
    sns.kdeplot(data=data, x=column, shade=True, color=color)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel(ylabel)
    plt.show()

def analyze_and_plot_diabetes_by_age(df, age_bins, age_labels, target_col, hue=None):
    """
    Analyzes and plots the proportion of diabetes by age intervals.

    Parameters:
        df (pd.DataFrame): The input dataset.
        age_bins (list): List of bin edges for grouping age.
        age_labels (list): List of labels corresponding to the age bins.
        target_col (str): Target column for calculating proportions (e.g., "diabetes").
        hue_col (str, optional): Column for grouping by hue (e.g., "heart_disease"). Default is None.

    Returns:
        None: Displays the plot.
    """
    # Step 1: Bin the age variable
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    # Step 2: Calculate proportions
    if hue:
        # Group by age group and hue
        proportions = df.groupby(['age_group', hue])[target_col].mean().reset_index()
    else:
        # Group by age group only
        proportions = df.groupby('age_group')[target_col].mean().reset_index()

    # Step 3: Plot the proportions
    #plt.figure(figsize=(10, 6))
    sns.lineplot(data=proportions, x='age_group', y=target_col, hue=hue, marker='o')
    plt.title('Proportion of Diabetes by Age Interval', fontsize=14)
    plt.xlabel('Age Interval', fontsize=12)
    plt.ylabel(f'Proportion of {target_col.capitalize()}', fontsize=12)
    if hue:
        plt.legend(title=hue)
    plt.show()
