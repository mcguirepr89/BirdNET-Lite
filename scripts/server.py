import apprise
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

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server.bind(ADDR)
except BaseException:
    print("Waiting on socket")
    time.sleep(5)


# Open most recent Configuration and grab DB_PWD as a python variable
userDir = os.path.expanduser('~')
with open(userDir + '/BirdNET-Pi/scripts/thisrun.txt', 'r') as f:
    this_run = f.readlines()
    audiofmt = "." + str(str(str([i for i in this_run if i.startswith('AUDIOFMT')]).split('=')[1]).split('\\')[0])
    priv_thresh = float("." + str(str(str([i for i in this_run if i.startswith('PRIVACY_THRESHOLD')]).split('=')[1]).split('\\')[0])) / 10


def loadModel():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    modelpath = userDir + '/BirdNET-Pi/model/BirdNET_6K_GLOBAL_MODEL.tflite'
    myinterpreter = tflite.Interpreter(model_path=modelpath, num_threads=2)
    myinterpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = myinterpreter.get_input_details()
    output_details = myinterpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
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


def sendAppriseNotifications(species, confidence, path, lat, lon, date, time, week, sens, cutoff, overlap):
    if os.path.exists(userDir + '/BirdNET-Pi/apprise.txt') and os.path.getsize(userDir + '/BirdNET-Pi/apprise.txt') > 0:
        with open(userDir + '/BirdNET-Pi/scripts/thisrun.txt', 'r') as f:
            this_run = f.readlines()
            title = str(str(str([i for i in this_run if i.startswith('APPRISE_NOTIFICATION_TITLE')]).split('=')[1]).split('\\')[0]).replace('"', '')
            body = str(str(str([i for i in this_run if i.startswith('APPRISE_NOTIFICATION_BODY')]).split('=')[1]).split('\\')[1]).replace("'", '')
            print("Notification Body " + body)
            
            try:
                websiteurl = str(str(str([i for i in this_run if i.startswith('BIRDNETPI_URL')]).split('=')[1]).split('\\')[0]).replace('"', '')
                if len(websiteurl) == 0:
                    raise ValueError('Blank URL')
            except Exception as e:
                websiteurl = "http://"+socket.gethostname()+".local"

            listenurl = websiteurl+"?filename="+path

        if str(str(str([i for i in this_run if i.startswith('APPRISE_NOTIFY_EACH_DETECTION')]).split('=')[1]).split('\\')[0]) == "1":

            apobj = apprise.Apprise()
            config = apprise.AppriseConfig()
            config.add(userDir + '/BirdNET-Pi/apprise.txt')
            apobj.add(config)

            apobj.notify(
                body=body.replace("$sciname", species.split("_")[0]).replace("$comname", species.split("_")[1]).replace("$confidence", confidence).replace("$listenurl", listenurl).replace("$latitude", lat).replace("$longitude", lon).replace("$date", date).replace("$time", time).replace("$week", week).replace("$cutoff", cutoff).replace("$overlap", overlap).replace("$sens", sens),
                title=title.replace("$sciname", species.split("_")[0]).replace("$comname", species.split("_")[1]).replace("$confidence", confidence).replace("$listenurl", listenurl).replace("$latitude", lat).replace("$longitude", lon).replace("$date", date).replace("$time", time).replace("$week", week).replace("$cutoff", cutoff).replace("$overlap", overlap).replace("$sens", sens)
            )

        if str(str(str([i for i in this_run if i.startswith('APPRISE_NOTIFY_NEW_SPECIES')]).split('=')[1]).split('\\')[0]) == "1":
            try:
                con = sqlite3.connect(userDir + '/BirdNET-Pi/scripts/birds.db')
                cur = con.cursor()
                cur.execute("SELECT DISTINCT(Com_Name), count(Com_Name) FROM detections WHERE date > (SELECT DATETIME('now', '-7 day')) GROUP BY Com_Name")
                known_species = cur.fetchall()
                sciName, comName = species.split("_")
                numberDetections = [d[1] for d in known_species if d[0] == comName.replace("'","")][0]

                if numberDetections <= 5:
                    apobj = apprise.Apprise()
                    config = apprise.AppriseConfig()
                    config.add(userDir + '/BirdNET-Pi/apprise.txt')
                    apobj.add(config)
                    apobj.notify(
                        body=body.replace("$sciname", species.split("_")[0]).replace("$comname", species.split("_")[1]).replace("$confidence", confidence).replace("$listenurl", listenurl).replace("$latitude", lat).replace("$longitude", lon).replace("$date", date).replace("$time", time).replace("$week", week).replace("$cutoff", cutoff).replace("$overlap", overlap).replace("$sens", sens) + " (only seen "+str(int(numberDetections))+" times in last 7d)",
                        title=title.replace("$sciname", species.split("_")[0]).replace("$comname", species.split("_")[1]).replace("$confidence", confidence).replace("$listenurl", listenurl).replace("$latitude", lat).replace("$longitude", lon).replace("$date", date).replace("$time", time).replace("$week", week).replace("$cutoff", cutoff).replace("$overlap", overlap).replace("$sens", sens) + " (only seen "+str(int(numberDetections))+" times in last 7d)",
                    )

                con.close()
            except BaseException:
                print("Database busy")
                time.sleep(2)


def writeResultsToFile(detections, min_conf, path):

    print('WRITING RESULTS TO', path, '...', end=' ')
    rcnt = 0
    with open(path, 'w') as rfile:
        rfile.write('Start (s);End (s);Scientific name;Common name;Confidence\n')
        for d in detections:
            for entry in detections[d]:
                if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)):
                    rfile.write(d + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) + '\n')
                    rcnt += 1
    print('DONE! WROTE', rcnt, 'RESULTS.')
    return


def handle_client(conn, addr):
    global INCLUDE_LIST
    global EXCLUDE_LIST
    # print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False
            else:
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

                # Read audio data
                audioData = readAudioData(args.i, args.overlap)

                # Get Date/Time from filename in case Pi gets behind
                # now = datetime.now()
                full_file_name = args.i
                # print('FULL FILENAME: -' + full_file_name + '-')
                file_name = Path(full_file_name).stem
                file_date = file_name.split('-birdnet-')[0]
                file_time = file_name.split('-birdnet-')[1]
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
                        for entry in detections[d]:
                            if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0)
                                                         and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)):
                                rfile.write(str(current_date) + ';' + str(current_time) + ';' + entry[0].replace('_', ';') + ';'
                                            + str(entry[1]) + ";" + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';'
                                            + str(args.sensitivity) + ';' + str(args.overlap) + '\n')

                                Date = str(current_date)
                                Time = str(current_time)
                                species = entry[0]
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
                                    Date.replace("/", "-") + '-birdnet-' + Time + audiofmt

                                # Connect to SQLite Database
                                for attempt_number in range(3):
                                    try:
                                        con = sqlite3.connect(userDir + '/BirdNET-Pi/scripts/birds.db')
                                        cur = con.cursor()
                                        cur.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (Date, Time,
                                                    Sci_Name, Com_Name, str(score), Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name, "UNVERIFIED"))

                                        con.commit()
                                        con.close()
                                        break
                                    except BaseException:
                                        print("Database busy")
                                        time.sleep(2)
                                        
                                sendAppriseNotifications(species, str(score), File_Name, Lat, Lon, Date, Time, Week, Sens, Cutoff, Overlap)

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

                                        if soundscape_uploaded is False:
                                            # POST soundscape to server
                                            soundscape_url = 'https://app.birdweather.com/api/v1/stations/' + \
                                                birdweather_id + \
                                                '/soundscapes' + \
                                                '?timestamp=' + \
                                                current_iso8601

                                            with open(args.i, 'rb') as f:
                                                wav_data = f.read()
                                            response = requests.post(url=soundscape_url, data=wav_data, headers={'Content-Type': 'application/octet-stream'})
                                            print("Soundscape POST Response Status - ", response.status_code)
                                            sdata = response.json()
                                            soundscape_id = sdata['soundscape']['id']
                                            soundscape_uploaded = True

                                        # POST detection to server
                                        detection_url = "https://app.birdweather.com/api/v1/stations/" + birdweather_id + "/detections"
                                        start_time = d.split(';')[0]
                                        end_time = d.split(';')[1]
                                        post_begin = "{ "
                                        now_p_start = now + datetime.timedelta(seconds=float(start_time))
                                        current_iso8601 = now_p_start.astimezone(get_localzone()).isoformat()
                                        post_timestamp = "\"timestamp\": \"" + current_iso8601 + "\","
                                        post_lat = "\"lat\": " + str(args.lat) + ","
                                        post_lon = "\"lon\": " + str(args.lon) + ","
                                        post_soundscape_id = "\"soundscapeId\": " + str(soundscape_id) + ","
                                        post_soundscape_start_time = "\"soundscapeStartTime\": " + start_time + ","
                                        post_soundscape_end_time = "\"soundscapeEndTime\": " + end_time + ","
                                        post_commonName = "\"commonName\": \"" + entry[0].split('_')[1] + "\","
                                        post_scientificName = "\"scientificName\": \"" + entry[0].split('_')[0] + "\","
                                        post_algorithm = "\"algorithm\": " + "\"alpha\"" + ","
                                        post_confidence = "\"confidence\": " + str(entry[1])
                                        post_end = " }"

                                        post_json = post_begin + post_timestamp + post_lat + post_lon + post_soundscape_id + post_soundscape_start_time + \
                                            post_soundscape_end_time + post_commonName + post_scientificName + post_algorithm + post_confidence + post_end
                                        print(post_json)
                                        response = requests.post(detection_url, json=json.loads(post_json))
                                        print("Detection POST Response Status - ", response.status_code)
                                    except BaseException:
                                        print("Cannot POST right now")
                conn.send(myReturn.encode(FORMAT))

                # time.sleep(3)

    conn.close()


def start():
    # Load model
    global INTERPRETER, INCLUDE_LIST, EXCLUDE_LIST
    INTERPRETER = loadModel()
    server.listen()
    # print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


# print("[STARTING] server is starting...")
start()
