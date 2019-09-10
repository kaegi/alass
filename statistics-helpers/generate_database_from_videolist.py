#!/usr/bin/python3

from pythonopensubtitles.opensubtitles import OpenSubtitles
import sys
import json
import zlib
import base64
import os
import errno
import shutil
import pysubs2
import re
import argparse
import time

import pprint
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--videolist-file', required=True, help='a file containing one path to a video file on each line (program input)')
parser.add_argument('--database-dir', required=True, help='directory for generated database files (program output)')

args = parser.parse_args()

videolist_file_path = args.videolist_file
database_dir = args.database_dir

print(videolist_file_path)

with open(videolist_file_path) as f:
    video_paths = [line.strip() for line in f]
    video_paths = [x for x in video_paths if x]


def make_parents(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def decompress(data, encoding):
    """
    Convert a base64-compressed subtitles file back to a string.

    :param data: the compressed data
    :param encoding: the encoding of the original file (e.g. utf-8, latin1)
    """
    try:
        return zlib.decompress(base64.b64decode(data),
                               16 + zlib.MAX_WBITS).decode(encoding)
    except UnicodeDecodeError as e:
        print(e, file=sys.stderr)
        return

def download_subtitles(ost, ids, encoding, override_filenames=None,
                           output_directory='.', override_directories=None,
                           extension='srt',
                           return_decoded_data=False):
        override_filenames = override_filenames or {}
        override_directories = override_directories or {}
        successful = {}

        # OpenSubtitles will accept a maximum of 20 IDs for download
        if len(ids) > 20:
            print("Cannot download more than 20 files at once.", file=sys.stderr)
            ids = ids[:20]

        response = ost.xmlrpc.DownloadSubtitles(ost.token, ids)
        status = response.get('status').split()[0]
        encoded_data = response.get('data') if '200' == status else None

        if not encoded_data:
            return None

        for item in encoded_data:
            subfile_id = item['idsubtitlefile']

            decoded_data = decompress(item['data'], encoding)

            if not decoded_data:
                print("An error occurred while decoding subtitle "
                      "file ID {}.".format(subfile_id), file=sys.stderr)
            elif return_decoded_data:
                successful[subfile_id] = decoded_data
            else:
                fname = override_filenames.get(subfile_id,
                                               subfile_id + '.' + extension)
                directory = override_directories.get(subfile_id,
                                                     output_directory)
                fpath = os.path.join(directory, fname)
                make_parents(fpath)

                try:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(decoded_data)
                    successful[subfile_id] = fpath
                except IOError as e:
                    print("There was an error writing file {}.".format(fpath),
                          file=sys.stderr)
                    print(e, file=sys.stderr)

        return successful or None


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    exit_cmd = ['q', 'Q', 'Quit', 'quit', 'exit']
    if default is None:
        prompt = " [y/n/q] "
    elif default == "yes":
        prompt = " [Y/n/q] "
    elif default == "no":
        prompt = " [y/N/q] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        choice = input(question + prompt).lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in exit_cmd:
             sys.exit(0)
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def handle_subtitle(movie_id, opensubtitles_metadata):

    # find subtitle ending (srt, ass, ...)
    #if subtitle['SubFormat'] not in sub_format_to_ending:
    #    sub_info_json = json.dumps(subtitle, indent=4)
    #    print(sub_info_json)
    #    print('Unreckognized subtitle format \'%s\'! Skipping this subtitle!' % subtitle['SubFormat'])
    #    continue
    #sub_ending = sub_format_to_ending[sub_format_to_ending[subtitle['SubFormat']]]

    #sub_filename = '{}-{:0>04}.{}'.format(movie_name_normalized, subtitle_idx, sub_ending)
    #sub_data_filename = '{}-{:0>04}.{}'.format(movie_name_normalized, subtitle_idx, 'json')
    sub_id = opensubtitles_metadata['IDSubtitleFile']
    print('Downloading subtitle with id `%s`...' % sub_id, file=sys.stderr, end=' ')
    data = None
    try:
        time.sleep(0.1)
        data = download_subtitles(ost, [sub_id],
                opensubtitles_metadata['SubEncoding'],
                return_decoded_data=True
                )
    except:
        print('error occured')

    if data == None:
        print('Error getting data - skipping subtitle!', file=sys.stderr)
        return None

    print('Done!', file=sys.stderr)

        
    ssa_styling_pattern = re.compile(r'\s*#?{[^}]*}#?\s*') # remove SSA-styling info
    newline_whitespace = re.compile(r'\s*\n\s*') # remove unnecessary trailing space around newlines

    line_data = []

    decoded_sub_data = pysubs2.SSAFile.from_string(data[sub_id], encoding=opensubtitles_metadata['SubEncoding'])
    for line in decoded_sub_data:
        if 'www.opensubtitles.org' in line.text.lower():
            continue # remove ad as this throws of pairing/statistics (same text in different places)

        text = line.text.replace('\n', '').replace('\r', '')
        text = ssa_styling_pattern.sub('', text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = text.replace(r'\N', '\n')
        text = text.strip()
        text = newline_whitespace.sub('\n', text)

        if line.start < line.end: 
            line_data.append({
                "start_ms": line.start,
                "end_ms": line.end,
                "text": text
            })
        elif line.start > line.end:
            line_data.append({
                "start_ms": line.end,
                "end_ms": line.start,
                "text": text
            })
        else:
            # start == end
            pass

    line_data = sorted(line_data, key=lambda l: l['start_ms'])
    
    return {
        "id": opensubtitles_metadata['IDSubtitleFile'],
        "movie_id": movie_id,
        "opensubtitles_metadata": opensubtitles_metadata,
        "data": line_data
    }


def handle_subtitle_files(movie_id, reference_subtitle_metadata, opensubtitle_metadatas):

    reference_subtitle_entry = handle_subtitle(movie_id, reference_subtitle_metadata)
    if reference_subtitle_entry == None:
        print('failed to download reference subtitle...', file=sys.stderr)
        return None


    token = ost.login('', '')
    print('New OpenSubtitles token: %s' % token, file=sys.stderr)

    result_subtitles_list = []
    
    for opensubtitle_metadata in opensubtitle_metadatas:
        if opensubtitle_metadata['IDSubtitle'] == reference_subtitle_metadata['IDSubtitle']:
            print('skipping reference subtitle...', file=sys.stderr)
            continue


        subtitle_entry = handle_subtitle(movie_id, opensubtitle_metadata)
        if subtitle_entry == None: continue
        result_subtitles_list.append(subtitle_entry)

    return (reference_subtitle_entry, result_subtitles_list)

def ask_user_for_movie(movie_name, correct_subtitle_metadata, subtitle_files):

    movie_name_normalized = movie_name.lower().replace(" ", "-")

    data = ost.search_movies_on_imdb(movie_name)
    for film in data['data']:
        if 'from_redis' in film and film['from_redis'] == 'false':
            continue
        print('%s [IMDB-ID: %s]' % (film['title'], film['id']), file=sys.stderr)
        answer = query_yes_no('Download subtitles for this movie?')
        print(file=sys.stderr)
        if answer is True:
            imdb_id = film['id']
            subtitle_files = ost.search_subtitles([{'imdbid': imdb_id, 'sublanguageid':'eng'}])
            handle_subtitle_files(movie_name_normalized, correct_subtitle_metadata, subtitle_files)

            sys.exit(0)



def to_normalized_name(s):
    printable = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
    return ''.join(filter(lambda x: x in printable, s.lower().replace(" ", "-")))

ost = OpenSubtitles()
from pythonopensubtitles.utils import File as OstFile

token = ost.login('', '')
print('OpenSubtitles token: %s' % token, file=sys.stderr)


movies_list = []
movies_without_reference_sub_count = 0

for file_idx, file_path in enumerate(video_paths):
    f = OstFile(file_path)
    file_hash = f.get_hash()

    print(file=sys.stderr)
    print('-------------------------------------------------------', file=sys.stderr)
    print('Movie `%s` with hash `%s`:' % (file_path, file_hash), file=sys.stderr)

    subtitle_files = ost.search_subtitles([{'moviehash': file_hash, 'sublanguageid':'eng'}])
    if len(subtitle_files) == 0:
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        print('Video file `%s` not registered on OpenSubtitles' % file_path, file=sys.stderr)
        movies_without_reference_sub_count = movies_without_reference_sub_count + 1

        continue

    correct_subtitle_file = subtitle_files[0]

    movie_name = correct_subtitle_file['MovieName']

    movie_name_normalized = to_normalized_name(movie_name)
    print('moviename is `%s`' % movie_name, file=sys.stderr)

    subtitle_files = ost.search_subtitles([{'idmovie': correct_subtitle_file['IDMovie'], 'sublanguageid':'eng'}])
    movie_id = '%s#%s' % (movie_name_normalized, file_idx)

    reference_subtitle, subtitle_list = handle_subtitle_files(movie_id, correct_subtitle_file, subtitle_files)

    movies_list.append(
        {
            "id": movie_id,
            "name": movie_name,
            "path": file_path,
            "reference_subtitle": reference_subtitle,
            "subtitles": subtitle_list
        }
    )


database_object = {
    "movies": movies_list,
    "movies_without_reference_sub_count": movies_without_reference_sub_count
}

print(file=sys.stderr)
print('Writing database file...', file=sys.stderr, end='')

os.makedirs(database_dir, exist_ok=True)
with open(os.path.join(database_dir, "database.json"), 'w') as f:
    json.dump(database_object, f)

print('Done!', file=sys.stderr)

sys.exit(0)
