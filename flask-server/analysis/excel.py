import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from data.constants import word_list

##############################
# Description: Computes Fleiss' Kappa statistic for measuring inter-rater reliability.
# Parameters:
#     - table: A 2D array-like structure representing the ratings for each subject (n_sub x n_cat).
#     - method: The method to compute expected agreement ('fleiss', 'rand', or 'unif'; default is 'fleiss').
# Returns: The computed Fleiss' Kappa value (float).
##############################
def fleiss_kappa(table, method='fleiss'):

    table = 1.0 * np.asarray(table)   #avoid integer division
    n_sub, n_cat =  table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    #assume fully ranked
    #assert n_total == n_sub * n_rat

    #marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    if method == 'fleiss':
        p_mean_exp = (p_cat*p_cat).sum()
    elif method.startswith('rand') or method.startswith('unif'):
        p_mean_exp = 1 / n_cat

    kappa = (p_mean - p_mean_exp) / (1- p_mean_exp)
    return kappa

##############################
# Description: Calculates Cohen's Kappa statistic for measuring inter-rater agreement.
# Parameters:
#     - df: A pandas DataFrame containing two columns of ratings from different raters.
# Returns: The calculated Cohen's Kappa value (float).
##############################
def calculate_cohens_kappa(df):
    total_ratings = len(df)

    # Extract ratings from DataFrame
    rater1_ratings = df.iloc[:, 0].tolist()
    rater2_ratings = df.iloc[:, 1].tolist()

    unique_labels = set(rater1_ratings + rater2_ratings)

    # Create the observed agreement matrix
    observed_agreement = sum(r1 == r2 for r1, r2 in zip(rater1_ratings, rater2_ratings)) / total_ratings

    # Calculate the probability of random agreement
    p_a = sum((rater1_ratings.count(label) / total_ratings) * (rater2_ratings.count(label) / total_ratings) for label in unique_labels)

    # Calculate the probability of agreement by chance
    p_e = sum((rater1_ratings.count(label) / total_ratings) ** 2 for label in unique_labels)

    # Calculate Cohen's kappa
    cohens_kappa = (observed_agreement - p_a) / (1 - p_e)

    return cohens_kappa

##############################
# Description: Extracts ratings from Excel sheets, computes Cohen's Kappa, and saves the results.
# Parameters: None
# Returns: None
##############################
def excel():
    long_list = []
    for word in word_list[1]:
        long_list.append(word)    
    for word in word_list[2]:
        long_list.append(word)

    #people = ['Ursa', 'Cole', 'Joyce','Ryley','Lulia','Isabel','Susan','Maria']
    two_people = ['Ursa', 'Cole']
    fleis = []
    for person in two_people:
        df = pd.read_excel('/Users/ayden/Desktop/responses.xlsx', sheet_name=person)

        condition_subset = df[df['Condition'] == 'N']
        desired = condition_subset['Normals Acc'].astype(int).to_numpy()  # Convert to integers
        fleis.append(desired)

    # Create DataFrame
    df_fleis = pd.DataFrame(fleis).T

    print(calculate_cohens_kappa(df_fleis))

    df_fleis.to_excel('/Users/ayden/Desktop/output.xlsx', index=False, sheet_name='temp')

    # df = pd.read_excel('/Users/ayden/Desktop/responses.xlsx', sheet_name='temp')
    # df['solution'] = [long_list[index] for index in df['Word']]
    # df.to_excel('/Users/ayden/Desktop/responses.xlsx', sheet_name='temp', index=False)
    # print(df)

##############################
# Description: Calculates and prints the percentage of agreement among raters.
# Parameters: None
# Returns: None
##############################
def percent_agreement():
    people = ['Ursa', 'Cole', 'Joyce','Ryley','Lulia','Isabel','Susan','Maria']

    df = pd.read_excel('/Users/ayden/Desktop/output_c.xlsx', sheet_name='c6000')
    first_eight_columns = df.iloc[1:50,:8]
    print(first_eight_columns)

    ave_agreemnt = []
    for index, row in first_eight_columns.iterrows():
        num_raters = len(row)
        agreement_count = row.sum()  # Assuming 1 indicates agreement
        
        if agreement_count == 0:
            percent_agreement = 100
        else:
            percent_agreement = max((agreement_count / num_raters) * 100, 50)
        
        print(f'Row {index}: {percent_agreement}% agreement')
        ave_agreemnt.append(percent_agreement)

    total_agreement = sum(ave_agreemnt)/50
    print(total_agreement)