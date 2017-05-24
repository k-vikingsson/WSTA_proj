import re
import json

with open('QA_train.json') as dev_file:
    dev = json.load(dev_file)

units = set()
for trial in dev:
    for question in trial['qa']:
        answer = question['answer']
        matches = re.findall(r'(\d+) (\S+)', answer)
        for match in matches:
            # print answer
            # print match[1]
            units.add(match[1].lower())

for unit in units:
    print unit.encode('utf-8')