## Utility function to combine all the part based metadata.json .. ##
## into a single dictionary and extracting it out as a .json ##

import json
from glob import glob

def combine_json_to_dict(dir , out_name):
    '''(str , str) --> None
    Combines all the .json file in the dir,
    converts them into a single dictionary
    and outputs it as a .json with the name
    "out_name.json".

    params:
    dir (str) : The name of the directory containing
    all the .json files.
    out_name (str) : The name of the output file.
    '''

    meta_files = glob(dir + '/*.json')

    assert len(meta_files) > 0 , 'Sorry no .json files found!!'

    combined_d = {}

    for file in meta_files:
        with open(file) as json_file:
            each_dict = json.load(json_file)
            for each_key in each_dict.keys():
                combined_d[each_key] = each_dict[each_key]

    print('Length of the combined dict is :' , len(combined_d.keys()))


    output = out_name + '.json'

    with open(output , 'w') as empty_json:
        json.dump(combined_d , empty_json)

    print('Completed!!')

combine_json_to_dict('preprocessing_code' , 'metadata')

