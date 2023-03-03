import pandas as pd
import os
import numpy as np
import shutil


def remove_all_accurences_of_element(list, element):
    while element in list:
        list.remove(element)
    return list


def random_unique_uniform_generator(top, size):
    random_numbrs = []
    while len(set(random_numbrs)) != size:
        random_numbrs.append(np.random.randint(top) + 1)
    return list(set(random_numbrs))


filenames =  os.listdir('.')
cases_from_reports = [filename.split('.')[0] for filename in filenames]


#removing repetitive elemnts from filenames
def remove_all_accurences_of_element(list, element):
    while element in list:
        list.remove(element)
    return list

filenames_uniqued = []
repetitive_elements = set([x for x in cases_from_reports if cases_from_reports.count(x) > 1])
for filename in filenames:
    for repetitive_element in repetitive_elements:
        if filename.split('.')[0] == repetitive_element:
            filenames_uniqued = remove_all_accurences_of_element(filenames, filename)
cases_from_reports_uniqued = [filename.split('.')[0] for filename in filenames_uniqued]


brca_cases_dataframe = pd.read_csv("nationwidechildrens.org_clinical_patient_brca.txt", sep='\t')


cases_from_clinical_table = brca_cases_dataframe['bcr_patient_barcode'].to_numpy()[2:]


intersection = list(set(cases_from_reports_uniqued).intersection(set(cases_from_clinical_table)))


random_sample_numbers = random_unique_uniform_generator(top=len(intersection), size=50)


random_cases = [intersection[random_number] for random_number in random_sample_numbers]


# os.mkdir('random_reports')
case_file_name = []
for filename in filenames:
    for case in random_cases:
        if filename.split('.')[0] == case:
            case_file_name.append(filename)


os.mkdir('cases')


f = open("cases/case_names.txt", "w")
for case in random_cases:
    f.write(case)
    f.write('\n')
f.close()


for case in case_file_name:
    shutil.copy(case, 'cases')


