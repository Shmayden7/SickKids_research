#########################
# Parsing Raw Deepgram Data
# Desc: Removes un-necessary information from the returned api dict
# Params: raw data dict, single string of attributes, space separated
# Return: Deepgram data that we care about, dict
# Dependant on: N.A.
#########################
def parseRawDeepgram(data: dict, attributes: str):

    # splits up the single string by spaces into array of attributes
    attribute_array = attributes.split()

    important_data = {} # the attributes we acc want

    # appending important fields to the returned dict
    for i in range(len(attribute_array)):
        important_data.update({attribute_array[i]: data[attribute_array[i]]})

    return important_data