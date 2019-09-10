#!/usr/bin/python3.7

import sys
import json
import os
import argparse
import subprocess

def format_srt_time(ms):
    subsec_ms = ms % 1000
    sec = (ms // 1000) % 60
    minutes = (ms // (1000 * 60)) % 60
    hours = (ms // (1000 * 60 * 60))
    return '{0:0>2}:{1:0>2}:{2:0>2},{3:0>3}'.format(hours, minutes, sec, subsec_ms)

def write_subtitle_data(subtitle, path):
    subtitle_data = subtitle['data']
    subtitle_data.sort(key = lambda line: line['start_ms'])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as srt_file:
        for i, line in enumerate(subtitle_data):
            time_header = '%s --> %s\n' % (format_srt_time(line['start_ms']), format_srt_time(line['end_ms']))
            srt_file.writelines(['%s\n' % (i + 1), time_header, line['text'], '\n\n'])

    #print('Wrote file `%s`!' % path)

def find_movie_and_sub(data, subtitle_id):
    for movie in data['movies']:
        for subtitle in movie['subtitles']:
            if subtitle['id'] == subtitle_id:
                return (movie, subtitle)
                
    return (None, None)

if len(sys.argv) < 4:
    print('Usage: program path/to/database.json path/to/output/dir subtitle_id1 subtitle_id2 subtitle_id3 ...', file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description='Export subtitle from database')
parser.add_argument('--database-dir', required=True, help='directory for database files (program input)')
parser.add_argument('--output-dir', required=True, help='directory for generated srt files (program output)')
parser.add_argument('--sub-ids', required=True, help='IDs of requested subtitles (comma separated)')
parser.add_argument('--open-mpv', action='store_true')

args = parser.parse_args()

database_path = os.path.join(args.database_dir, "database.json")
output_dir = args.output_dir
subtitle_ids = args.sub_ids.split(',')

with open(database_path) as json_file:
    data = json.load(json_file)

for subtitle_id in subtitle_ids:

    ref_movie, subtitle_data = find_movie_and_sub(data, subtitle_id)

    if subtitle_data == None:
        print('Subtitle with id `%s` not found in `%s`' % (subtitle_id, database_path))
        sys.exit(1)

    ref_sub_data = ref_movie['reference_subtitle']

    out_sub_path = os.path.join(output_dir, '%s.srt' % subtitle_id)
    out_ref_path = os.path.join(output_dir, '%s_ref.srt' % subtitle_id)

    write_subtitle_data(subtitle_data, out_sub_path)
    write_subtitle_data(ref_sub_data, out_ref_path)

    print("subtitle id: '%s' [%s lines]" % (subtitle_id, len(subtitle_data['data'])))
    print("reference subtitle id: '%s' [%s lines]" % (ref_sub_data['id'], len(ref_sub_data['data'])))
    print("reference movie id: '%s'" % ref_movie['id'])
    print("mpv '%s' --sub-file '%s' --sub-file '%s'" % (ref_movie['path'], out_sub_path, out_ref_path))

    if args.open_mpv:
        subprocess.run(['mpv', ref_movie['path'], '--sub-file', out_sub_path, '--sub-file', out_ref_path])