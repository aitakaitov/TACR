
Each directory in 'output' ~ 1 transformation of database. default = no transformation,
2_removed = 'potential ad' annotations removed, 1_to_0 = 'hidden ad' moved to 'declared ad'

Each directory contains report.txt (stats) and datasets. 0.5 and 0.7 = majority fraction acceptance 
threshold ~  how large the fraction of annotators agreeing on the most common website state should be
in order for said website to be included in the dataset. 
Example: annotators say [0, 0, 2] -> major class is 0, majority size = 2 (two zeroes), 
majority fraction = 2 / 3 (total annotators) = 0.66 -> excluded from 0.7 dataset,
but included in 0.5 dataset.

transformed.db = sqlite3 database which the dataset was built from (for reproduction/backup purposes)

Datasets (0.5.csv and 0.7.csv)
separators used:
    ';;;' - between things made by one annotator
    '///' - between annotators
columns:
    url = website url with location in original url list structure
    emails = '///'-separated emails of annotators (their unique ID) 
    states = '///'-separated annotated website state/class (declared/hidden/potential/no ad) from annotators
    majority_fraction = size of annotator majority
    span_counts = '///'-separated supporting span count
    texts = '///'-separated annotator ';;;'-separated span texts
    reasons = '///'-separated annotator ';;;'-separated span reasons (see label_explanation.txt)
    start_paths = '///'-separated annotator ';;;'-separated identifiers of website elements where spans start as displayed on the website
    start_offsets = '///'-separated annotator ';;;'-separated locations of span starts in the corresponding elements as displayed on the website 
    end_paths = ...
    end_offsets = ...
        