import os
import re
if 'FILE' not in os.environ:
  print('Error. Your FILE environment variable is not set to a code path file')

contents = open(os.environ['FILE'], 'r').read()
num_replacements = 0
while '\n' + ('\t'*num_replacements) + '  ' in contents:
  contents = contents.replace('\n' + ('\t'*num_replacements) + '  ', '\n\t' + ('\t'*num_replacements))
  num_replacements+=1

while num_replacements > 0:
  contents = contents.replace('\n' + ('\t'*num_replacements), '\n' + ('    '*num_replacements))
  num_replacements -= 1

print(contents)
