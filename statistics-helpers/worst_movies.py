import json

with open('generated-data/2-statistics/statistics.json', 'r') as f:
    data = json.load(f)

data2 = [x for x in data['offset_by_subtitle']]

wo = sorted(data2, key=lambda x: x['video_sync_offsets']['perc99'])
print(json.dumps(wo, indent=3))
