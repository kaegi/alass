#!/usr/bin/python3
import json
import os
import datetime
import copy

print('start load')
with open('generated-data/1-database/database.json', 'r') as f:
    database = json.load(f)
print('end load')

for movie_data in database['movies']:
    print()
    print('%s ref %s' % (movie_data['id'], movie_data['reference_subtitle']['id']))
    for sub_data in movie_data['subtitles']:
        print('%s %s' % (movie_data['id'], sub_data['id']))

#import time
#time.sleep(2)
#print(sub_data['id'])
#time.sleep(2)
#print(orig_sub_data['data'])
#time.sleep(2)


#walle['reference_subtitle']['data'] = new_movie_ref_sub_data
#walle['reference_subtitle']['id'] = '%s,%s' % (walle['reference_subtitle']['id'], move)
