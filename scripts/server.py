import queue
import re
import sys
from pathlib import Path
from tzlocal import get_localzone
import datetime
import sqlite3
import requests
import json
import time
import math
import numpy as np
import librosa
import operator
import socket
import threading
import os
import gzip

from utils.notifications import sendAppriseNotifications
from utils.parse_settings import config_to_settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tflite_runtime.interpreter as tflite
except BaseException:
    from tensorflow import lite as tflite


HEADER = 64
PORT = 5050
SERVER = "localhost"
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

userDir = os.path.expanduser('~')
DB_PATH = userDir + '/BirdNET-Pi/scripts/birds.db'

PREDICTED_SPECIES_LIST = []

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# List of our BirdWeather submission threads
bw_worker_threads = list()
bw_submission_queue = queue.Queue()
# BirdWeather Soundscape ID caching
bw_soundscape_submission_id_cache = list()
bw_soundscape_submission_id_cache_limit = 40
#
# Retry on these response codes
bw_request_retry_on_status = [404, 429, 500, 502, 503, 504]
bw_default_request_timeout = 10
bw_default_post_timeout = 6 * bw_default_request_timeout
# Stop processing once we hit the max number of tries, API might be down so limit the retries, together with the
# bw_default_request_timeout the total retry period is 60 seconds without taking the backoff time between requests into account
bw_request_max_retries = 6
# Used in an exponential calculation to provide the number of seconds to wait before making the request again
bw_request_backoff_factor = 3.5

# DEBUG flag to enable debug output in select functions
debug_birdweather_submissions = False

try:
    server.bind(ADDR)
except BaseException:
    print("Waiting on socket")
    time.sleep(5)


# Open most recent Configuration and grab DB_PWD as a python variable
with open(userDir + '/BirdNET-Pi/scripts/thisrun.txt', 'r') as f:
    this_run = f.readlines()
    audiofmt = "." + str(str(str([i for i in this_run if i.startswith('AUDIOFMT')]).split('=')[1]).split('\\')[0])
    priv_thresh = float("." + str(str(str([i for i in this_run if i.startswith('PRIVACY_THRESHOLD')]).split('=')[1]).split('\\')[0])) / 10
    try:
        model = str(str(str([i for i in this_run if i.startswith('MODEL')]).split('=')[1]).split('\\')[0])
        sf_thresh = str(str(str([i for i in this_run if i.startswith('SF_THRESH')]).split('=')[1]).split('\\')[0])
    except Exception:
        model = "BirdNET_6K_GLOBAL_MODEL"
        sf_thresh = 0.03


def loadModel():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    # model will either be BirdNET_GLOBAL_6K_V2.4_Model_FP16 (new) or BirdNET_6K_GLOBAL_MODEL (old)
    modelpath = userDir + '/BirdNET-Pi/model/'+model+'.tflite'
    myinterpreter = tflite.Interpreter(model_path=modelpath, num_threads=2)
    myinterpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = myinterpreter.get_input_details()
    output_details = myinterpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    if model == "BirdNET_6K_GLOBAL_MODEL":
        MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    labelspath = userDir + '/BirdNET-Pi/model/labels.txt'
    with open(labelspath, 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')

    return myinterpreter


def loadMetaModel():

    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = tflite.Interpreter(model_path=userDir + '/BirdNET-Pi/model/BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite')
    M_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()

    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]['index']
    M_OUTPUT_LAYER_INDEX = output_details[0]['index']

    print("loaded META model")


def predictFilter(lat, lon, week):

    global M_INTERPRETER

    # Does interpreter exist?
    try:
        if M_INTERPRETER is None:
            loadMetaModel()
    except Exception:
        loadMetaModel()

    # Prepare mdata as sample
    sample = np.expand_dims(np.array([lat, lon, week], dtype='float32'), 0)

    # Run inference
    M_INTERPRETER.set_tensor(M_INPUT_LAYER_INDEX, sample)
    M_INTERPRETER.invoke()

    return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER_INDEX)[0]


def explore(lat, lon, week):

    # Make filter prediction
    l_filter = predictFilter(lat, lon, week)

    # Apply threshold
    l_filter = np.where(l_filter >= float(sf_thresh), l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, CLASSES))

    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

    return l_filter


def predictSpeciesList(lat, lon, week):

    l_filter = explore(lat, lon, week)
    for s in l_filter:
        if s[0] >= float(sf_thresh):
            # if there's a custom user-made include list, we only want to use the species in that
            if (len(INCLUDE_LIST) == 0):
                PREDICTED_SPECIES_LIST.append(s[1])


def loadCustomSpeciesList(path):

    slist = []
    if os.path.isfile(path):
        with open(path, 'r') as csfile:
            for line in csfile.readlines():
                slist.append(line.replace('\r', '').replace('\n', ''))

    return slist


def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp

        sig_splits.append(split)

    return sig_splits


def readAudioData(path, overlap, sample_rate=48000):

    print('READING AUDIO DATA...', end=' ', flush=True)

    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')

    # Split audio into 3-second chunks
    chunks = splitSignal(sig, rate, overlap)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks


def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])


def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))


def predict(sample, sensitivity):
    global INTERPRETER
    # Make a prediction
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    if model == "BirdNET_6K_GLOBAL_MODEL":
        INTERPRETER.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    INTERPRETER.invoke()
    prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

#     # print("DATABASE SIZE:", len(p_sorted))
#     # print("HUMAN-CUTOFF AT:", int(len(p_sorted)*priv_thresh)/10)
#
#     # Remove species that are on blacklist

    human_cutoff = max(10, int(len(p_sorted) * priv_thresh))

    for i in range(min(10, len(p_sorted))):
        if p_sorted[i][0] == 'Human_Human':
            with open(userDir + '/BirdNET-Pi/HUMAN.txt', 'a') as rfile:
                rfile.write(str(datetime.datetime.now()) + str(p_sorted[i]) + ' ' + str(human_cutoff) + '\n')

    return p_sorted[:human_cutoff]


def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap,):
    global INTERPRETER

    detections = {}
    start = time.time()
    print('ANALYZING AUDIO...', end=' ', flush=True)

    if model == "BirdNET_GLOBAL_6K_V2.4_Model_FP16":
        if len(PREDICTED_SPECIES_LIST) == 0 or len(INCLUDE_LIST) != 0:
            predictSpeciesList(lat, lon, week)

    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        sig = np.expand_dims(c, 0)

        # Make prediction
        p = predict([sig, mdata], sensitivity)
#        print("PPPPP",p)
        HUMAN_DETECTED = False

        # Catch if Human is recognized
        for x in range(len(p)):
            if "Human" in p[x][0]:
                HUMAN_DETECTED = True
                break

        # Save result and timestamp
        pred_end = pred_start + 3.0

        # If human detected set all detections to human to make sure voices are not saved
        if HUMAN_DETECTED is True:
            p = [('Human_Human', 0.0)] * 10

        detections[str(pred_start) + ';' + str(pred_end)] = p

        pred_start = pred_end - overlap

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')
#    print('DETECTIONS:::::',detections)
    return detections


def writeResultsToFile(detections, min_conf, path):

    print('WRITING RESULTS TO', path, '...', end=' ')
    rcnt = 0
    with open(path, 'w') as rfile:
        rfile.write('Start (s);End (s);Scientific name;Common name;Confidence\n')
        for d in detections:
            for entry in detections[d]:
                if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0)
                                             and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)
                                             and (entry[0] in PREDICTED_SPECIES_LIST or len(PREDICTED_SPECIES_LIST) == 0)):
                    rfile.write(d + ';' + entry[0].replace('_', ';').split("/")[0] + ';' + str(entry[1]) + '\n')
                    rcnt += 1
    print('DONE! WROTE', rcnt, 'RESULTS.')
    return


def handle_client(conn, addr):
    global INCLUDE_LIST
    global EXCLUDE_LIST
    # print(f"[NEW CONNECTION] {addr} connected.")

    while True:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if not msg_length:
            break

        msg_length = int(msg_length)
        msg = conn.recv(msg_length).decode(FORMAT)
        if not msg:
            break
        if msg == DISCONNECT_MESSAGE:
            break

        # print(f"[{addr}] {msg}")

        args = type('', (), {})()

        args.i = ''
        args.o = ''
        args.birdweather_id = '99999'
        args.include_list = 'null'
        args.exclude_list = 'null'
        args.overlap = 0.0
        args.week = -1
        args.sensitivity = 1.25
        args.min_conf = 0.70
        args.lat = -1
        args.lon = -1

        for line in msg.split('||'):
            inputvars = line.split('=')
            if inputvars[0] == 'i':
                args.i = inputvars[1]
            elif inputvars[0] == 'o':
                args.o = inputvars[1]
            elif inputvars[0] == 'birdweather_id':
                args.birdweather_id = inputvars[1]
            elif inputvars[0] == 'include_list':
                args.include_list = inputvars[1]
            elif inputvars[0] == 'exclude_list':
                args.exclude_list = inputvars[1]
            elif inputvars[0] == 'overlap':
                args.overlap = float(inputvars[1])
            elif inputvars[0] == 'week':
                args.week = int(inputvars[1])
            elif inputvars[0] == 'sensitivity':
                args.sensitivity = float(inputvars[1])
            elif inputvars[0] == 'min_conf':
                args.min_conf = float(inputvars[1])
            elif inputvars[0] == 'lat':
                args.lat = float(inputvars[1])
            elif inputvars[0] == 'lon':
                args.lon = float(inputvars[1])

        # Load custom species lists - INCLUDED and EXCLUDED
        if not args.include_list == 'null':
            INCLUDE_LIST = loadCustomSpeciesList(args.include_list)
        else:
            INCLUDE_LIST = []

        if not args.exclude_list == 'null':
            EXCLUDE_LIST = loadCustomSpeciesList(args.exclude_list)
        else:
            EXCLUDE_LIST = []

        birdweather_id = args.birdweather_id

        # Read audio data & handle errors
        try:
            audioData = readAudioData(args.i, args.overlap)

        except (NameError, TypeError) as e:
            print(f"Error with the following info: {e}")
            open('~/BirdNET-Pi/analyzing_now.txt', 'w').close()

        finally:
            pass

        # Get Date/Time from filename in case Pi gets behind
        # now = datetime.now()
        full_file_name = args.i
        # print('FULL FILENAME: -' + full_file_name + '-')
        file_name = Path(full_file_name).stem

        # Get the RSTP stream identifier from the filename if it exists
        RTSP_ident_for_fn = ""
        RTSP_ident = re.search("RTSP_[0-9]+-", file_name)
        if RTSP_ident is not None:
            RTSP_ident_for_fn = RTSP_ident.group()

        # Find and remove the identifier for the RSTP stream url it was from that is added when more than one
        # RSTP stream is recorded simultaneously, in order to make the filenames unique as filenames are all
        # generated at the same time
        file_name = re.sub("RTSP_[0-9]+-", "", file_name)

        # Now we can read the date and time as normal
        # First portion of the filename contaning the date in Y m d
        file_date = file_name.split('-birdnet-')[0]
        # Second portion of the filename containing the time in H:M:S
        file_time = file_name.split('-birdnet-')[1]
        # Join the date and time together to get a complete string representing when the audio was recorded
        date_time_str = file_date + ' ' + file_time
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        # print('Date:', date_time_obj.date())
        # print('Time:', date_time_obj.time())
        print('Date-time:', date_time_obj)
        now = date_time_obj
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        current_iso8601 = now.astimezone(get_localzone()).isoformat()

        week_number = int(now.strftime("%V"))
        week = max(1, min(week_number, 48))

        sensitivity = max(0.5, min(1.0 - (args.sensitivity - 1.0), 1.5))

        # Process audio data and get detections
        detections = analyzeAudioData(audioData, args.lat, args.lon, week, sensitivity, args.overlap)

        # Write detections to output file
        min_conf = max(0.01, min(args.min_conf, 0.99))
        writeResultsToFile(detections, min_conf, args.o)

    ###############################################################################
    ###############################################################################

        soundscape_uploaded = False

        # Write detections to Database
        myReturn = ''
        for i in detections:
            myReturn += str(i) + '-' + str(detections[i][0]) + '\n'

        with open(userDir + '/BirdNET-Pi/BirdDB.txt', 'a') as rfile:
            for d in detections:
                species_apprised_this_run = []
                for entry in detections[d]:
                    if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0)
                                                 and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)
                                                 and (entry[0] in PREDICTED_SPECIES_LIST or len(PREDICTED_SPECIES_LIST) == 0)):
                        # Write to text file.
                        rfile.write(str(current_date) + ';' + str(current_time) + ';' + entry[0].replace('_', ';').split("/")[0] + ';'
                                    + str(entry[1]) + ";" + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';'
                                    + str(args.sensitivity) + ';' + str(args.overlap) + '\n')

                        # Write to database
                        Date = str(current_date)
                        Time = str(current_time)
                        species = entry[0].split("/")[0]
                        Sci_Name, Com_Name = species.split('_')
                        score = entry[1]
                        Confidence = str(round(score * 100))
                        Lat = str(args.lat)
                        Lon = str(args.lon)
                        Cutoff = str(args.min_conf)
                        Week = str(args.week)
                        Sens = str(args.sensitivity)
                        Overlap = str(args.overlap)
                        Com_Name = Com_Name.replace("'", "")
                        File_Name = Com_Name.replace(" ", "_") + '-' + Confidence + '-' + \
                            Date.replace("/", "-") + '-birdnet-' + RTSP_ident_for_fn + Time + audiofmt

                        # Connect to SQLite Database
                        for attempt_number in range(3):
                            try:
                                con = sqlite3.connect(DB_PATH)
                                cur = con.cursor()
                                cur.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (Date, Time,
                                            Sci_Name, Com_Name, str(score), Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name))

                                con.commit()
                                con.close()
                                break
                            except BaseException:
                                print("Database busy")
                                time.sleep(2)

                        # Apprise of detection if not already alerted this run.
                        if not entry[0] in species_apprised_this_run:
                            settings_dict = config_to_settings(userDir + '/BirdNET-Pi/scripts/thisrun.txt')
                            sendAppriseNotifications(species,
                                                     str(score),
                                                     str(round(score * 100)),
                                                     File_Name,
                                                     Date,
                                                     Time,
                                                     Week,
                                                     Lat,
                                                     Lon,
                                                     Cutoff,
                                                     Sens,
                                                     Overlap,
                                                     settings_dict,
                                                     DB_PATH)
                            species_apprised_this_run.append(entry[0])

                        print(str(current_date) +
                              ';' +
                              str(current_time) +
                              ';' +
                              entry[0].replace('_', ';') +
                              ';' +
                              str(entry[1]) +
                              ';' +
                              str(args.lat) +
                              ';' +
                              str(args.lon) +
                              ';' +
                              str(min_conf) +
                              ';' +
                              str(week) +
                              ';' +
                              str(args.sensitivity) +
                              ';' +
                              str(args.overlap) +
                              ';' +
                              File_Name +
                              '\n')

                        if birdweather_id != "99999":
                            try:

                                # BirdWeather submissions are now added to a queue and processed in a seperate thread (via birdweather_submission_processor and birdweather_submit)
                                # We still collect and compile all the URLS and JSON to pass through

                                # POST soundscape to server
                                # Build the URL used when uploading the soundscape
                                soundscape_url = 'https://app.birdweather.com/api/v1/stations/' + \
                                                 birdweather_id + \
                                                 '/soundscapes' + \
                                                 '?timestamp=' + \
                                                 current_iso8601

                                # Get the filename for this soundscape file
                                soundscape_filename = Path(args.i).stem

                                # Extract the audio
                                with open(args.i, 'rb') as f:
                                    wav_data = f.read()

                                # Compress the audio data using gzip
                                gzip_wav_data = gzip.compress(wav_data)

                                # POST detection to server
                                # Build the URL and all other data (to be added to a json array used when uploading the detection data
                                detection_url = "https://app.birdweather.com/api/v1/stations/" + birdweather_id + "/detections"
                                start_time = d.split(';')[0]
                                end_time = d.split(';')[1]
                                #
                                now_p_start = now + datetime.timedelta(seconds=float(start_time))
                                current_iso8601 = now_p_start.astimezone(get_localzone()).isoformat()

                                # Build a dictionary that represents the JSON data that will be posted to BirdWeather
                                # Instead of hand crafting the JSON string, just to ease maintainability
                                post_data = dict()
                                post_data['timestamp'] = current_iso8601
                                post_data['lat'] = str(args.lat)
                                post_data['lon'] = str(args.lon)
                                post_data['soundscapeId'] = "{{soundscape_id}}"
                                post_data['soundscapeStartTime'] = start_time
                                post_data['soundscapeEndTime'] = end_time
                                post_data['commonName'] = entry[0].split('_')[1].split("/")[0]
                                post_data['scientificName'] = entry[0].split('_')[0]

                                # Determine the algorithm used
                                if model == "BirdNET_GLOBAL_6K_V2.4_Model_FP16":
                                    post_data['algorithm'] = "2p4"
                                else:
                                    post_data['algorithm'] = "alpha"

                                post_data['confidence'] = str(entry[1])
                                ####

                                # Convert the detection data dictionary into a JSON string
                                post_json = json.dumps(post_data)
                                # print(post_json)

                                if debug_birdweather_submissions:
                                    print(
                                        f'handle_client:: debug_birdweather_submissions:: Add Birdweather queue submission with, soundscape_url:{soundscape_url} - wave_data:{sys.getsizeof(gzip_wav_data) / 1000}KB - detection_url:{detection_url} - json_detection_data:{post_json}')

                                # Create a dictionary containing all the data we need to posts a submission to BirdWeather
                                submission_data = dict()
                                submission_data['soundscape_url'] = soundscape_url
                                submission_data['soundscape_filename'] = soundscape_filename
                                submission_data['gzip_wav_data'] = gzip_wav_data
                                submission_data['detection_url'] = detection_url
                                submission_data['detection_post_json'] = post_json

                                # Add it to the queue to be processed
                                bw_submission_queue.put_nowait(submission_data)

                            except BaseException as b_exec:
                                print(f"ERROR: Cannot POST to BirdWeather right now - {b_exec}")

        conn.send(myReturn.encode(FORMAT))

        # time.sleep(3)

    conn.close()


def birdweather_submit(bw_submission_data):
    soundscape_id = None
    #
    extra_debug_output = ''

    # Grab the URL and sound data form the supplied dictionary
    # Soundscape POST data
    soundscape_url = bw_submission_data.pop('soundscape_url')
    wave_sound_data = bw_submission_data.pop('gzip_wav_data')
    soundscape_filename = bw_submission_data.pop('soundscape_filename')
    # Detection post data
    detection_url = bw_submission_data.pop('detection_url')
    detection_post_json = bw_submission_data.pop('detection_post_json')

    if debug_birdweather_submissions:
        print(f'BirdWeather Submission:: DEBUG:: URL: {soundscape_url}', flush=True)

    ##################################
    # SOUNDSCAPE UPLOAD #############
    ##################################
    # Loop for the max number of retries
    for ss_p_rt in range(bw_request_max_retries):
        extra_debug_output = ''
        # Don't calculate or sleep on the first loop as this is the initial attempt
        if ss_p_rt > 0:
            # Calculate the backoff time before making another request
            request_backoff_time = bw_request_backoff_factor * (2 ** (ss_p_rt - 1))
            # We're retrying, retry after the calculated backoff time
            print(f'BirdWeather Submission Error:: Retrying after {request_backoff_time}s, Retry ({ss_p_rt} of {bw_request_max_retries})', flush=True)
            time.sleep(request_backoff_time)

        try:
            # First see if we've already submitted this wave file and get the soundscape id for it
            find_existing_soundscape = birdweather_soundscape_id_cache('search', soundscape_filename)

            # Didn't find a soundscape id for the file we're processing, upload it to Birdweather and cache it's soundscape id
            if not find_existing_soundscape['soundscape_found']:
                if debug_birdweather_submissions:
                    print(f'BirdWeather Submission:: Did not find soundscape {soundscape_filename} in cache, Posting soundscape to BirdWeather', flush=True)

                # Submit the soundscape submission
                soundscape_async_response = requests.post(url=soundscape_url, data=wave_sound_data,
                                                          headers={'Content-Type': 'application/octet-stream',
                                                                   'Content-Encoding': 'gzip'},
                                                          timeout=bw_default_post_timeout)
                # Raise a error if response is not 2XX
                soundscape_async_response.raise_for_status()

                # Spit out the whole dict response if debugging
                if debug_birdweather_submissions:
                    print(f'BirdWeather Submission:: DEBUG:: Soundscape POST - RESPONSE: {soundscape_async_response}', flush=True)

                # Extract some data
                soundscape_response_json = soundscape_async_response.json()
                # Get the soundscape id for the soundscape uploaded
                soundscape_id = soundscape_response_json['soundscape']['id']

                # Cache the soundscape id for corresponding to the filename the wave uploaded
                birdweather_soundscape_id_cache('add', soundscape_filename, soundscape_id)

                print(f"BirdWeather Submission:: Soundscape Successfully Uploaded - status:{soundscape_async_response.status_code} soundscape_id:{soundscape_id}", flush=True)
            else:
                # We found the soundscape filename and the Birdweather Soundscape ID for it, soundscape considered uploaded & use the ID for the detection
                soundscape_id = find_existing_soundscape['soundscape_id']
                if debug_birdweather_submissions:
                    print(f"BirdWeather Submission:: Found Existing Soundscape in cache for {soundscape_filename} - using soundscape_id:{soundscape_id}", flush=True)

            # Break the loop, if we reach here then were no exceptions and the detection posted successfully
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as conn_exec:
            print(f"BirdWeather Submission Error:: Soundscape POST - Connection Error! - {conn_exec}", flush=True)
            continue
        except requests.exceptions.RequestException as request_exc:
            # Check if the status code is one that we can try on
            if request_exc.response.status_code in bw_request_retry_on_status:

                if debug_birdweather_submissions:
                    extra_debug_output = f' - {request_exc.response.reason} - {request_exc}'

                print(f"BirdWeather Submission Error:: Soundscape POST - HTTP Request Exception! {extra_debug_output}", flush=True)
                continue
            else:
                print(f"BirdWeather Submission Error:: Soundscape POST - HTTP Request Exception! Cannot retry - {request_exc}", flush=True)
                # break the loop on non-retryable status
                break
        except (requests.exceptions.JSONDecodeError, requests.exceptions.InvalidJSONError) as json_error_exec:
            print(f'BirdWeather Submission Error:: Soundscape POST - Something went wrong decoding JSON data - {json_error_exec}', flush=True)
        except BaseException as ss_ex:
            print(f'BirdWeather Submission Error:: Soundscape POST - Something went wrong - {ss_ex}', flush=True)

    ##################################
    # DETECTION UPLOAD #############
    ##################################
    # Loop for the max number of retries
    for detect_p_rt in range(bw_request_max_retries):
        extra_debug_output = ''
        # Don't calculate or sleep on the first loop as this is the initial attempt
        if detect_p_rt > 0:
            # Calculate the backoff time before making another request
            request_backoff_time = bw_request_backoff_factor * (2 ** (detect_p_rt - 1))
            # We're retrying, retry after the calculated backoff time
            print(f'BirdWeather Submission Error:: Retrying after {request_backoff_time}s, Retry ({detect_p_rt} of {bw_request_max_retries})', flush=True)
            time.sleep(request_backoff_time)

        # We need to substitute in the soundscape_id into the detection_post_json data, since it the ID isn't available until the soundscape is uploaded
        # and because we submitted the full json data with a placeholder set for the soundscape_id
        detection_post_json = detection_post_json.replace("{{soundscape_id}}", str(soundscape_id))

        # Some debugging output if needed
        if debug_birdweather_submissions:
            print(f'BirdWeather Submission:: DEBUG:: Detection POST - detection_url: {detection_url} -  detection_json: {detection_post_json} - Soundscape_ID: {soundscape_id}', flush=True)

        # Submit the detection
        try:
            # POST detection to server
            detection_async_response = requests.post(detection_url,
                                                     json=json.loads(detection_post_json),
                                                     timeout=bw_default_request_timeout)
            # Raise a error if response is not 2XX
            detection_async_response.raise_for_status()

            # Spit out the whole dict response if debugging
            if debug_birdweather_submissions:
                print(f'BirdWeather Submission:: DEBUG:: Detection POST - RESPONSE: {detection_async_response}', flush=True)

            # Extract some data
            detection_response_status_json = detection_async_response.json()

            # Check the response
            # Extract the bird detection info to display in the output
            bird_detection_string = "N/A"
            if 'detection' in detection_response_status_json:
                bird_detection_name = detection_response_status_json['detection']['species']['commonName']
                bird_detection_confidence = detection_response_status_json['detection']['confidence']
                bird_detection_timestamp = datetime.datetime.fromisoformat(
                    detection_response_status_json['detection']['timestamp'])
                bird_detection_time = bird_detection_timestamp.time()
                bird_detection_string = f"- {bird_detection_time}/{bird_detection_name}/{bird_detection_confidence}"

            if debug_birdweather_submissions:
                # Add in the JSON response if debugging just in case we might want to view it
                extra_debug_output = f'- json:{detection_response_status_json}'

            print(f"BirdWeather Submission:: Detection Successfully Uploaded - status:{detection_async_response.status_code} {bird_detection_string} {extra_debug_output}", flush=True)

            # Break the loop, if we reach here then were no exceptions and the detection posted successfully
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as conn_exec:
            print(f"BirdWeather Submission Error:: Detection POST - Connection Error! - {conn_exec}", flush=True)
            continue
        except requests.exceptions.RequestException as request_exc:
            # Check if the status code is one that we can try on
            req_status_code = request_exc.response.status_code
            if req_status_code in bw_request_retry_on_status:

                if debug_birdweather_submissions:
                    extra_debug_output = f'- {request_exc.response.reason} - {request_exc}'

                print(f"BirdWeather Submission Error:: Detection POST - HTTP Request Exception! {extra_debug_output}", flush=True)
                continue
            else:
                print(
                    f"BirdWeather Submission Error:: Detection POST - HTTP Request Exception! Cannot retry - {request_exc}",
                    flush=True)
                # break the loop on non-retryable status
                break
        except (requests.exceptions.JSONDecodeError, requests.exceptions.InvalidJSONError) as json_error_exec:
            print(
                f'BirdWeather Submission Error:: Detection POST - Something went wrong decoding JSON data - {json_error_exec}',
                flush=True)
        except BaseException as dp_ex:
            print(f'BirdWeather Submission Error:: Detection POST - Something went wrong - {dp_ex}', flush=True)


def birdweather_submission_processor():
    if debug_birdweather_submissions:
        print("Starting: birdweather_submission_processor thread")

    # Loop over the queue containing the data used for the BirdWeather submissions
    while True:
        # Get the BirdWeather submission data, this is a dictionary containing the necessary data
        bw_submission_data = bw_submission_queue.get()

        if debug_birdweather_submissions:
            print(
                f"Processing: Soundscape:{bw_submission_data['soundscape_url']}, Detection:{bw_submission_data['detection_url']}, Detection_Data:{bw_submission_data['detection_post_json']}")

        # Perform the submission
        birdweather_submit(bw_submission_data)
        # Processing finished so the task is now done
        bw_submission_queue.task_done()


def birdweather_soundscape_id_cache(mode, soundscape_filename, bw_soundscape_id=None):
    global bw_soundscape_submission_id_cache
    ss_id_was_found = False
    ss_id_to_return = 0

    if debug_birdweather_submissions:
        print(f"birdweather_soundscape_id_cache - mode:{mode} - for {soundscape_filename}")

    if mode == 'search':
        # Search the list for the filename
        for soundscape_submission in bw_soundscape_submission_id_cache:
            this_ss_filename = soundscape_submission['soundscape_filename']
            this_ss_id = soundscape_submission['soundscape_id']
            # If this soundscape filename matches the one we're searching for, return the bw soundscape id
            if this_ss_filename == soundscape_filename:
                ss_id_was_found = True
                ss_id_to_return = this_ss_id
                if debug_birdweather_submissions:
                    print(
                        f"birdweather_soundscape_id_cache - Found {soundscape_filename} with soundscape_id:{ss_id_to_return}")
                break

        return {'soundscape_found': ss_id_was_found, 'soundscape_id': ss_id_to_return}
    elif mode == 'add':
        # If a filename AND soundscape ID has been supplied then we want to store it in the list
        if soundscape_filename is not None and bw_soundscape_id is not None:
            # Check the length of the list first
            if len(bw_soundscape_submission_id_cache) >= bw_soundscape_submission_id_cache_limit:
                # Remove the first item in the list before inserting a new item
                del bw_soundscape_submission_id_cache[0]

            # Create a new dict containing the appropriate data
            new_ss_submission = dict()
            new_ss_submission['soundscape_filename'] = soundscape_filename
            new_ss_submission['soundscape_id'] = bw_soundscape_id
            # Append it to the list
            bw_soundscape_submission_id_cache.append(new_ss_submission)

            if debug_birdweather_submissions:
                print(
                    f"birdweather_soundscape_id_cache - Inserting entry soundscape_filename:{soundscape_filename} - soundscape_id:{bw_soundscape_id}")


def start():
    # Load model
    global INTERPRETER, INCLUDE_LIST, EXCLUDE_LIST
    INTERPRETER = loadModel()

    # Run the BirdWeather submission queue processor in a thread
    bw_submission = threading.Thread(target=birdweather_submission_processor)
    bw_worker_threads.append(bw_submission)
    bw_submission.start()

    server.listen()
    # print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


# print("[STARTING] server is starting...")
start()
