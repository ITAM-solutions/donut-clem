import sys
import json

content = sys.argv[1]
content_json = json.loads(content)
training_instance = [i.get('id') for i in content_json if str(i.get('label')).lower() == 'donut'][0]
print(training_instance)   

