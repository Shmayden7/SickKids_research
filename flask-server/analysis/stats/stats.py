#########################
# Imports:
from analysis.json_helpers import dumpDictToJSON, loadDictFromJSON
# Constants:

file_name = 'list_1_assembly_t1.json'
directory = 'data/json/assembly'

#########################


data = loadDictFromJSON(file_name, directory)

