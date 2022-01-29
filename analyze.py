# BirdWeather edits by @timsterc
# Other edits by @CaiusX and @mcguirepr89
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite

import argparse
import operator
import librosa
import numpy as np
import math
import time
from decimal import Decimal
import json
###############################################################################    
import requests
import mysql.connector
###############################################################################
import datetime
import pytz
from tzlocal import get_localzone
from pathlib import Path

def loadModel():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path='model/BirdNET_6K_GLOBAL_MODEL.tflite',num_threads=2)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    with open('model/labels.txt', 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')

    return interpreter

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

def predict(sample, interpreter, sensitivity):

    # Make a prediction
    interpreter.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    interpreter.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    # Remove species that are on blacklist
    for i in range(min(10, len(p_sorted))):
        if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
            p_sorted[i] = (p_sorted[i][0], 0.0)

    # Only return first the top ten results
    return p_sorted[:10]

def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap, interpreter):

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
        p = predict([sig, mdata], interpreter, sensitivity)

        # Save result and timestamp
        pred_end = pred_start + 3.0
        detections[str(pred_start) + ';' + str(pred_end)] = p
        pred_start = pred_end - overlap

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')

    return detections

def writeResultsToFile(detections, min_conf, path):

    print('WRITING RESULTS TO', path, '...', end=' ')
    rcnt = 0
    with open(path, 'w') as rfile:
        rfile.write('Start (s);End (s);Scientific name;Common name;Confidence\n')
        for d in detections:
            for entry in detections[d]:
                if entry[1] >= min_conf and (entry[0] in WHITE_LIST or len(WHITE_LIST) == 0):
                    rfile.write(d + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) + '\n')
                    rcnt += 1
    print('DONE! WROTE', rcnt, 'RESULTS.')

def main():

    global WHITE_LIST

    # Parse passed arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Path to input file.')
    parser.add_argument('--o', default='result.csv', help='Path to output file. Defaults to result.csv.')
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 to ignore.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap in seconds between extracted spectrograms. Values in [0.0, 2.9]. Defaults tp 0.0.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')   
    parser.add_argument('--custom_list', default='', help='Path to text file containing a list of species. Not used if not provided.')
    parser.add_argument('--birdweather_id', default='99999', help='Private Station ID for BirdWeather.')    

    args = parser.parse_args()

    # Load model
    interpreter = loadModel()

    # Load custom species list
    if not args.custom_list == '':
        WHITE_LIST = loadCustomSpeciesList(args.custom_list)
    else:
        WHITE_LIST = []

    birdweather_id = args.birdweather_id

    # Read audio data
    audioData = readAudioData(args.i, args.overlap)

    # Get Date/Time from filename in case Pi gets behind
    #now = datetime.now()
    full_file_name = args.i
    file_name = Path(full_file_name).stem
    file_date = file_name.split('-birdnet-')[0]
    file_time = file_name.split('-birdnet-')[1]
    date_time_str = file_date + ' ' + file_time
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    #print('Date:', date_time_obj.date())
    #print('Time:', date_time_obj.time())
    print('Date-time:', date_time_obj)
    now = date_time_obj
    current_date = now.strftime("%Y/%m/%d")
    current_time = now.strftime("%H:%M:%S")
    current_iso8601 = now.astimezone(get_localzone()).isoformat()
    
    week_number = int(now.strftime("%V"))
    week = max(1, min(week_number, 48))

    sensitivity = max(0.5, min(1.0 - (args.sensitivity - 1.0), 1.5))

    # Process audio data and get detections
    detections = analyzeAudioData(audioData, args.lat, args.lon, week, sensitivity, args.overlap, interpreter)

    # Write detections to output file
    min_conf = max(0.01, min(args.min_conf, 0.99))
    writeResultsToFile(detections, min_conf, args.o)
    
###############################################################################    
###############################################################################    
    
    soundscape_uploaded = False

    # Write detections to Database
    for i in detections:
      print("\n", detections[i][0],"\n")
    with open('BirdDB.txt', 'a') as rfile:
        for d in detections:
            print("\n", "Database Entry", "\n")
            for entry in detections[d]:
                if entry[1] >= min_conf and (entry[0] in WHITE_LIST or len(WHITE_LIST) == 0):
                    rfile.write(str(current_date) + ';' + str(current_time) + ';' + entry[0].replace('_', ';') + ';' \
                    + str(entry[1]) +";" + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';' \
                    + str(sensitivity) +';' + str(args.overlap) + '\n')

                    def insert_variables_into_table(Date, Time, Sci_Name, Com_Name, Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap):
                        try:
                            connection = mysql.connector.connect(host='localhost',
                                                                 database='birds',
                                                                 user='birder',
                                                                 password='birdnet')
                            cursor = connection.cursor()
                            mySql_insert_query = """INSERT INTO detections (Date, Time, Sci_Name, Com_Name, Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap)
                                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) """
                    
                            record = (Date, Time, Sci_Name, Com_Name, Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap)

                            cursor.execute(mySql_insert_query, record)
                            connection.commit()
                            print("Record inserted successfully into detections table")

                    
                        except mysql.connector.Error as error:
                            print("Failed to insert record into detections table {}".format(error))
                        
                        finally:
                            if connection.is_connected():
                                connection.close()
                                print("MySQL connection is closed")

                    species = entry[0]
                    sci_name,com_name = species.split('_')
                    insert_variables_into_table(str(current_date), str(current_time), sci_name, com_name, \
                    str(entry[1]), str(args.lat), str(args.lon), str(min_conf), str(week), \
                    str(args.sensitivity), str(args.overlap))

                    print(str(current_date) + ';' + str(current_time) + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) +";" + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';' + str(args.sensitivity) +';' + str(args.overlap) + '\n')

                    if birdweather_id != "99999":

                        if soundscape_uploaded is False:
                            # POST soundscape to server
                            soundscape_url = "https://app.birdweather.com/api/v1/stations/" + birdweather_id +  "/soundscapes" + "?timestamp=" + current_iso8601

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
                        post_timestamp =  "\"timestamp\": \"" + current_iso8601 + "\","
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

                        post_json = post_begin + post_timestamp + post_lat + post_lon + post_soundscape_id + post_soundscape_start_time + post_soundscape_end_time + post_commonName + post_scientificName + post_algorithm + post_confidence + post_end
                        print(post_json)
                        response = requests.post(detection_url, json=json.loads(post_json))
                        print("Detection POST Response Status - ", response.status_code)

                    #time.sleep(3)

###############################################################################    
###############################################################################    

if __name__ == '__main__':

    main()

    # Example calls
    # python3 analyze.py --i 'example/XC558716 - Soundscape.mp3' --lat 35.4244 --lon -120.7463 --week 18
    # python3 analyze.py --i 'example/XC563936 - Soundscape.mp3' --lat 47.6766 --lon -122.294 --week 11 --overlap 1.5 --min_conf 0.25 --sensitivity 1.25 --custom_list 'example/custom_species_list.txt'
