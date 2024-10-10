# import pandas as pd
# import json

# # Define the person mapping
# person_mapping = {
#     1: 'aCauchi',
#     2: 'Claire',
#     3: 'Cole',
#     4: 'Daniel',
#     5: 'Hillary',
#     6: 'Lulia',
#     7: 'Melissa',
#     8: 'Micheal',
#     9: 'Polonenko',
#     10: 'rCauchi',
#     11: 'Robel',
#     12: 'Samsoor'
# }

# # Define the condition mapping
# condition_mapping = {
#     1: 'c1000',
#     2: 'c2000',
#     4: 'c4000',
#     6: 'c6000',
#     'N': 'Normal',
#     'F': 'dFirst',
#     'L': 'dLast'
# }

# # Load the Excel file
# file_path = '/Users/ayden/Desktop/SickKids/excel/interrater_assess.xlsx'
# df = pd.read_excel(file_path)

# # Display the column names for debugging
# print(df.columns)

# # Check for potential leading/trailing whitespace in column names
# df.columns = df.columns.str.strip()

# # Map person number to name using the 'Speaker' column
# df['speaker'] = df['Speaker'].map(person_mapping)

# # Map condition codes to full condition names
# df['condition'] = df['Condition'].map(condition_mapping)

# # Select and rename columns
# df = df[['speaker', 'condition', 'Word Number', 'Word']].rename(columns=str.lower)

# # Convert to list of dictionaries
# data = df.to_dict(orient='records')

# # Save to JSON
# output_path = 'data/inter_rater_assess.json'
# with open(output_path, 'w') as f:
#     json.dump(data, f, indent=4)

# print(f"Data successfully written to {output_path}")





import json
import pandas as pd

# Load the existing JSON data
with open('data/inter_rater_assess.json', 'r') as file:
    data = json.load(file)

# Load the Excel file
excel_file = pd.ExcelFile('/Users/ayden/Desktop/SickKids/excel/responses.xlsx')

# List of names in the Excel sheets
sheet_names = ['Ursa', 'Cole', 'Joyce', 'Ryley', 'Lulia', 'Isabel', 'Susan', 'Maria']

# Access the second column for each sheet, starting from the second row
Ursa = excel_file.parse('Ursa', header=None).iloc[1:, 1]
Cole = excel_file.parse('Cole', header=None).iloc[1:, 1]
Joyce = excel_file.parse('Joyce', header=None).iloc[1:, 1]
Ryley = excel_file.parse('Ryley', header=None).iloc[1:, 1]
Lulia = excel_file.parse('Lulia', header=None).iloc[1:, 1]
Isabel = excel_file.parse('Isabel', header=None).iloc[1:, 1]
Susan = excel_file.parse('Susan', header=None).iloc[1:, 1]
Maria = excel_file.parse('Maria', header=None).iloc[1:, 1]

# Initialize responses field for each entry in JSON data
for entry in data:
    entry['responses'] = {
        'ursa': '', 
        'cole': '', 
        'joyce': '', 
        'ryley': '', 
        'lulia': '', 
        'isabel': '', 
        'susan': '', 
        'maria': ''
    }

# Add responses to JSON data
for row in range(350):  # Iterate through all entries
    responses = {
        'ursa': Ursa.iloc[row].lower() if row < len(Ursa) and isinstance(Ursa.iloc[row], str) else '',
        'cole': Cole.iloc[row].lower() if row < len(Cole) and isinstance(Cole.iloc[row], str) else '',
        'joyce': Joyce.iloc[row].lower() if row < len(Joyce) and isinstance(Joyce.iloc[row], str) else '',
        'ryley': Ryley.iloc[row].lower() if row < len(Ryley) and isinstance(Ryley.iloc[row], str) else '',
        'lulia': Lulia.iloc[row].lower() if row < len(Lulia) and isinstance(Lulia.iloc[row], str) else '',
        'isabel': Isabel.iloc[row].lower() if row < len(Isabel) and isinstance(Isabel.iloc[row], str) else '',
        'susan': Susan.iloc[row].lower() if row < len(Susan) and isinstance(Susan.iloc[row], str) else '',
        'maria': Maria.iloc[row].lower() if row < len(Maria) and isinstance(Maria.iloc[row], str) else ''
    }

    data[row]['responses'] = responses

# Save the updated JSON data
with open('data/inter_rater_assess.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Responses appended successfully.")