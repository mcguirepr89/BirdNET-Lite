#!/usr/bin/env bash
BINDIR_COMMON=$(cd $(dirname $0) && pwd)
# Load config and settings
source /etc/birdnet/birdnet.conf
# $HOME here is home of the user executing the script
EXPANDED_HOME=$HOME

# Get the user and home directory of the user with id 1000 (normally pi)
USER=$(awk -F: '/1000/ {print $1}' /etc/passwd)
HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)

# If script is being run as sudo get the home path for the user that's running sudo
#if [ $SUDO_USER ]; then HOME=$(getent passwd $SUDO_USER | cut -d: -f6); fi

filePathMap_data=""
# Test some paths to see where the config file is accessible
# First try find the config folder under the current working directory where this file is executing, else find it the current working directory BUT up 1 level
# else find it under the current working directory
if [ -f "${BINDIR_COMMON}/config/filepath_map.json" ]; then
  filePathMap_json_path="${BINDIR_COMMON}/config/filepath_map.json"
elif [ -f "${PWD%/*}/config/filepath_map.json" ]; then
  filePathMap_json_path="${PWD%/*}/config/filepath_map.json"
else
  filePathMap_json_path="${PWD%}/config/filepath_map.json"
fi

loadFilePathMap() {
  #Loads in the JSON file containing data on directory and file paths
  if [ -z "$filePathMap_data" ]; then
    # Test if jq is installed
    test_json_str=$(echo "{ }" | jq)
    if [ "$test_json_str" != "{}" ]; then
      echo "jq command-line JSON processor is not installed, Install it with; sudo apt-get install jq"
      exit 1
    fi

    # Test if the path to the JSON file is valud
    if [ ! -f "$filePathMap_json_path" ]; then
      echo "Specified JSON file $filePathMap_json_path does not exist"
      exit 2
    fi

    # Read in the JSON file
    filePathMap_data=$(cat "$filePathMap_json_path")
  fi
}

readJsonConfig() {
  # Reads the filepath_map JSON file and processes it with jq
  loadFilePathMap
  echo $filePathMap_data | jq -r "$1"
}

readJsonElement() {
  # Read a specific element from the supplied JSON string
  # $1 is the JSON string and $2 is the value to read
  # e.g $1 = {"description": "The home path for the user with User ID of 1000, this is normally the 'pi' user",  "read_setting": null,  "return_var": "home" }
  # e.g $2 = '.description'
  # will return the description from the JSON array
  echo "$1" | jq -r "$2"
}

getDirectory() {
  DIR_NAME=$1
  DIR_NAME=$(echo "$DIR_NAME" | tr '[:upper:]' '[:lower:]')

  filePathMap_directories=$(readJsonConfig ".directories")

  if [ "${filePathMap_directories[$DIR_NAME]}" ]; then
    filePathMap_directories_selected=$(readJsonElement "$filePathMap_directories" ".$DIR_NAME")

    # Check to see if directory is an alias for another
    test_dir_alias="$(readJsonElement "$filePathMap_directories_selected" ".alias_for")"
    if [ -n "$test_dir_alias" ] && [ "$test_dir_alias" != "null" ]; then
      # If so load in the data for that directory
      filePathMap_directories_selected=$(readJsonElement "$filePathMap_directories" ".$test_dir_alias")
    fi

    # Read in all the options from the JSON string at once
    filePathMap_directory_options="$(readJsonElement "$filePathMap_directories_selected" ".read_setting , .lives_under , .replace_setting_text , .replace_setting_text_with , .append , .return_var")"
    # Split the string by newlines into an array so we can access individual elements
    readarray -t ElementArray <<<"$filePathMap_directory_options"

    # Gather all the options into variables
    test_read_setting="${ElementArray[0]}"
    test_lives_under="${ElementArray[1]}"
    test_replace_setting_text="${ElementArray[2]}"
    test_replace_setting_text_with="${ElementArray[3]}"
    test_append="${ElementArray[4]}"
    test_return_var="${ElementArray[5]}"
    #
    if [ -n "$test_read_setting" ] && [ "$test_read_setting" != "null" ]; then setting_value="$test_read_setting"; else setting_value=""; fi
    if [ -n "$test_lives_under" ] && [ "$test_lives_under" != "null" ]; then lives_under="$test_lives_under"; else lives_under=""; fi
    if [ -n "$test_replace_setting_text" ] && [ "$test_replace_setting_text" != "null" ]; then replace_text="$test_replace_setting_text"; else replace_text=""; fi
    if [ -n "$test_replace_setting_text_with" ] && [ "$test_replace_setting_text_with" != "null" ]; then replace_test_with="$test_replace_setting_text_with"; else replace_test_with=""; fi
    if [ -n "$test_append" ] && [ "$test_append" != "null" ]; then append="$test_append"; else append=""; fi
    if [ -n "$test_return_var" ] && [ "$test_return_var" != "null" ]; then return_val="$test_return_var"; else return_val=""; fi
    #
    under_directory=''

    # Get the directory which the directory we're processing lives under
    if [ -n "$lives_under" ]; then
      under_directory=$(getDirectory "$lives_under")
    fi

    # Read the specified config file setting if any
    if [ -n "$setting_value" ]; then
      setting_value="${!setting_value}"
    fi

    # Replace value in setting, like ${RECS_DIR} etc as they are not expanded
    if [ -n "$replace_text" ]; then
      # remove characters that represent a pre-expanded variable, eg ${RECS_DIR} in the ini file will expand correctly to the e.g /home/pi/BirdSongs
      # but in other languages like  it remains as ${RECS_DIR} as no expansion takes place, The pre-expanded variable is used in the filepat map
      replace_text=${replace_text//"\$"/""}
      replace_text=${replace_text//"{"/""}
      replace_text=${replace_text//"}"/""}

      # Now we have the actual variable name, expand it to read it's value
      replace_val=${!replace_text}
      #replace the entire ${RECS_DIR} expansion as it is expanded on $HOME of the user executing the script
      #this may or may not be correct, so we want to be able to replace it and put in place the calculated recordings dir
      return_value=${setting_value//"$replace_val"/"$replace_test_with"}

      # Replace the home path part of the setting value as it was expanded using ENV variables for the user executing this script
      # This allows the under_directory to be prepended to whatever is left
      return_value=${return_value//"$EXPANDED_HOME"/""}
    else
      return_value=$setting_value
    fi

    # If a variable is specified for return, return it first and directly
    if [ -n "$return_val" ]; then
      #test if trying to read the variable works in it's current form. e.g 'home'
      if [ -z "${!return_val}" ]; then
        # uppercase the to be variable name so it matches a variable in the current session 'home' => 'HOME'
        # e.g which will match the users HOME location specified at the start of this script
        return_val="$(echo "$return_val" | tr '[:lower:]' '[:upper:]')"
      fi

      # return the dynamic variable, currently this is just the users home path in $home
      echo "${!return_val}"
    elif [ -n "$append" ]; then
      # Append this to the end of the path, (models, scripts etc) do this as they reside under BirdNET-pi
      echo "$under_directory$append"
    else
      # Else return the directory and result of the setting manipulation
      echo "$under_directory$return_value"
    fi
  else
    echo
  fi
}

getFilePath() {
  FILENAME=$1
  # Manipulate the filename to accommodate filenames which inherently have periods in the filename
  # https://stedolan.github.io/jq/manual/#Basicfilters
  FILENAME=".\"$FILENAME\""

  filePathMap_files=$(readJsonConfig ".files")
  # Check and make sure the supplied filename exists before further processing
  test_file_key_exists="$(readJsonElement "$filePathMap_files" $FILENAME)"
  if [ -n "$test_file_key_exists" ] && [ "$test_file_key_exists" != "null" ]; then
    filePathMap_file_selected=$test_file_key_exists

    # Read in all the options from the JSON string at once
    filePathMap_file_options="$(readJsonElement "$filePathMap_file_selected" ".read_setting , .lives_under , .replace_setting_text , .replace_setting_text_with , .append , .return_var")"
    # Split the string by newlines into an array so we can access individual elements
    readarray -t ElementArray <<<"$filePathMap_file_options"

    # Gather all the options into variables, array indexes are in order of how the values were read by readJsonElement
    test_read_setting="${ElementArray[0]}"
    test_lives_under="${ElementArray[1]}"
    test_replace_setting_text="${ElementArray[2]}"
    test_replace_setting_text_with="${ElementArray[3]}"
    test_append="${ElementArray[4]}"
    test_return_var="${ElementArray[5]}"
    #
    if [ -n "$test_read_setting" ] && [ "$test_read_setting" != "null" ]; then setting_value="$test_read_setting"; else setting_value=""; fi
    if [ -n "$test_lives_under" ] && [ "$test_lives_under" != "null" ]; then lives_under="$test_lives_under"; else lives_under=""; fi
    if [ -n "$test_replace_setting_text" ] && [ "$test_replace_setting_text" != "null" ]; then replace_text="$test_replace_setting_text"; else replace_text=""; fi
    if [ -n "$test_replace_setting_text_with" ] && [ "$test_replace_setting_text_with" != "null" ]; then replace_test_with="$test_replace_setting_text_with"; else replace_test_with=""; fi
    if [ -n "$test_append" ] && [ "$test_append" != "null" ]; then append="$test_append"; else append=""; fi
    if [ -n "$test_return_var" ] && [ "$test_return_var" != "null" ]; then return_val="$test_return_var"; else return_val=""; fi
    #
    under_directory=''

    # Get the directory which the directory we're processing lives under
    if [ -n "$lives_under" ]; then
      under_directory=$(getDirectory "$lives_under")
    fi

    # Read the specified config file setting if any
    if [ -n "$setting_value" ]; then
      setting_value="${!setting_value}"
    fi

    # Replace value in setting, like ${RECS_DIR} etc as they are not expanded
    if [ -n "$replace_text" ]; then
      # remove characters that represent a pre-expanded variable, eg ${RECS_DIR} in the ini file will expand correctly to the e.g /home/pi/BirdSongs
      # but in other languages like  it remains as ${RECS_DIR} as no expansion takes place, The pre-expanded variable is used in the filepat map
      replace_text=${replace_text//"\$"/""}
      replace_text=${replace_text//"{"/""}
      replace_text=${replace_text//"}"/""}

      # Now we have the actual variable name, expand it to read it's value
      replace_val=${!replace_text}
      #replace the entire ${RECS_DIR} expansion as it is expanded on $HOME of the user executing the script
      #this may or may not be correct, so we want to be able to replace it and put in place the calculated recordings dir
      return_value=${setting_value//"$replace_val"/"$replace_test_with"}

      # Replace the home path part of the setting value as it was expanded using ENV variables for the user executing this script
      # This allows the under_directory to be prepended to whatever is left
      return_value=${return_value//"$EXPANDED_HOME"/""}
    else
      return_value=$setting_value
    fi

    if [ -n "$return_val" ]; then
      # If a value has been specified for return, return it first
      echo "$return_val"
    elif [ -n "$append" ]; then
      # Append this to the end of the path, (models, scripts etc) do this as they reside under BirdNET-pi
      echo "$under_directory$append"
    else
      # Else return the directory and result of the setting manipulation
      echo "$under_directory$return_value"
    fi
  else
    echo
  fi
}
