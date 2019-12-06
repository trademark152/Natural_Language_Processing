import sys
import glob
import os
import collections

## List all files, given the root of training data.
all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

# defaultdict is analogous to dict() [or {}], except that for keys that do not
# yet exist (i.e. first time access), the value gets contructed using the function
# pointer (in this case, list() i.e. initializing all keys to empty lists).
test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
train_by_class = collections.defaultdict(list)

for f in all_files:
  # Take only last 4 components of the path. The earlier components are useless
  # as they contain path to the classes directories.
  class1, class2, fold, fname = f.split('/')[-4:]
  if fold == 'fold1':
    # True-clause will not enter in Vocareum as fold1 wont exist, but useful for your own code.
    test_by_class[class1+class2].append(f)
  else:
    train_by_class[class1+class2].append(f)


### Print the file names (i.e. the dictionaries of classes and their filenames)
import json

print('\n\n *** Test data:')
print(json.dumps(test_by_class, indent=2))
print('\n\n *** Train data:')
print(json.dumps(train_by_class, indent=2))