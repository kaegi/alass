#!/usr/bin/python3

from pythonopensubtitles.opensubtitles import OpenSubtitles
import sys
import json
import zlib
import base64
import os

import pprint
pp = pprint.PrettyPrinter(indent=4)


if len(sys.argv) < 2:
    print('Expected video name as command line argument!')
    sys.exit(1)

film_name = sys.argv[1]

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
        print(e)
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
            print("Cannot download more than 20 files at once.",
                  file=sys.stderr)
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
                    print(e)

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

film_name_normalized = film_name.lower().replace(" ", "-")

sub_format_to_ending = {
        'srt': 'srt',
        'ssa': 'ass',
        'ass': 'ass',
        }

ost = OpenSubtitles()
token = ost.login('', '')
print('OpenSubtitles token: %s' % token);
print()

data = ost.search_movies_on_imdb(film_name)
for film in data['data']:
    if 'from_redis' in film and film['from_redis'] == 'false':
        continue
    print('%s [IMDB-ID: %s]' % (film['title'], film['id']))
    answer = query_yes_no('Download subtitles for this movie?')
    print()
    if answer is True:
        imdb_id = film['id']
        subtitles = ost.search_subtitles([{'imdbid': imdb_id, 'sublanguageid':'eng'}])
        for subtitle_idx, subtitle in enumerate(subtitles):
            #pp.pprint(subtitle)
            # convert into JSON:

            # find subtitle ending (srt, ass, ...)
            if subtitle['SubFormat'] not in sub_format_to_ending:
                sub_info_json = json.dumps(subtitle, indent=4)
                print(sub_info_json)
                print('Unreckognized subtitle format \'%s\'! Skipping this subtitle!' % subtitle['SubFormat'])
                continue
            sub_ending = sub_format_to_ending[sub_format_to_ending[subtitle['SubFormat']]]

            sub_info_json = json.dumps(subtitle, indent=4)
            print(sub_info_json)

            sub_filename = '{}-{:0>04}.{}'.format(film_name_normalized, subtitle_idx, sub_ending)
            sub_id = subtitle['IDSubtitleFile']
            data = download_subtitles(ost, [sub_id],
                    subtitle['SubEncoding'],
                    override_filenames={sub_id: sub_filename},
                    output_directory='../database/%s' % film_name_normalized
                    )
            print(data)

        sys.exit(0)


