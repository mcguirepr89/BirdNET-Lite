import numpy as np
import os
import argparse
import datetime

try:
    import tflite_runtime.interpreter as tflite
except BaseException:
    from tensorflow import lite as tflite


def loadMetaModel():

    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX
    global CLASSES

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = tflite.Interpreter(model_path=userDir + '/BirdNET-Pi/model/BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite')
    M_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()

    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]['index']
    M_OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    labelspath = userDir + '/BirdNET-Pi/model/labels.txt'
    with open(labelspath, 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

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


def explore(lat, lon, week, threshold):

    # Make filter prediction
    l_filter = predictFilter(lat, lon, week)

    # Apply threshold
    l_filter = np.where(l_filter >= threshold, l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, CLASSES))

    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

    return l_filter


def getSpeciesList(lat, lon, week, threshold=0.05, sort=False):

    print('Getting species list for {}/{}, Week {}...'.format(lat, lon, week), end='', flush=True)

    # Extract species from model
    pred = explore(lat, lon, week, threshold)

    # Make species list
    slist = []
    for p in pred:
        if p[0] >= threshold:
            slist.append([p[1], p[0]])

    return slist


userDir = os.path.expanduser('~')
DB_PATH = userDir + '/BirdNET-Pi/scripts/birds.db'
with open(userDir + '/BirdNET-Pi/scripts/thisrun.txt', 'r') as f:

    this_run = f.readlines()
    lat = str(str(str([i for i in this_run if i.startswith('LATITUDE')]).split('=')[1]).split('\\')[0])
    lon = str(str(str([i for i in this_run if i.startswith('LONGITUDE')]).split('=')[1]).split('\\')[0])

weekofyear = datetime.datetime.today().isocalendar()[1]
if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Get list of species for a given location with BirdNET. Sorted by occurrence frequency.'
    )
    parser.add_argument('--threshold', type=float, default=0.05, help='Occurrence frequency threshold. Defaults to 0.05.')

    args = parser.parse_args()

    LOCATION_FILTER_THRESHOLD = args.threshold

    # Get species list
    species_list = getSpeciesList(lat, lon, weekofyear, LOCATION_FILTER_THRESHOLD, False)
    for x in range(len(species_list)):
        print(species_list[x][0] + " - " + str(species_list[x][1]))

    print("\nThe above species list describes all the species that the model will attempt to detect. \
          If you don't see a species you want detected on this list, decrease your threshold.")
    print("\nNOTE: no actual changes to your BirdNET-Pi species list were made by running this command. \
          To set your desired frequency threshold, do it through the BirdNET-Pi web interface (Tools -> Settings -> Model)")
