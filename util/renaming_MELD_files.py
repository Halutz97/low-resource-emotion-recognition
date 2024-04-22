import os
import re

def rename_numbering(directory):
    max_utt_id = 0
    num_wrong_format = 0
    num_renamed = 0
    for filename in os.listdir(directory):
        if filename.endswith('.wav') and filename.startswith('dia'):
            # old_filename = filename
            dia_index = filename.find("dia")
            _utt_index = filename.find("_utt")
            _wav_index = filename.find(".wav")

            dialogue_id_len = _utt_index - (dia_index + 3)
            dialogue_id = filename[dia_index + 3:dia_index + 3 + dialogue_id_len]

            utterance_id_len = _wav_index - (_utt_index + 4)
            utterance_id = filename[_utt_index + 4:_utt_index + 4 + utterance_id_len]
        
            if int(dialogue_id) >= 1000:
                dialogue_id = dialogue_id
            elif int(dialogue_id) >= 100:
                dialogue_id = "0" + dialogue_id
            elif int(dialogue_id) >= 10:
                dialogue_id = "00" + dialogue_id
            else:
                dialogue_id = "000" + dialogue_id

            if int(utterance_id) >= 10:
                utterance_id = utterance_id
            else:
                utterance_id = "0" + utterance_id

            if int(utterance_id) > max_utt_id:
                max_utt_id = int(utterance_id)

            new_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"

            # print(filename)
            # print(new_filename)
            # print()

        
            # print(dia_index)
            # print(_utt_index)
            # print(_wav_index)
            # print(dialogue_id_len)
            # print(dialogue_id)
            # print()
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            num_renamed += 1
            # print(f"Renamed {filename} to {new_filename}")
        else:
            num_wrong_format += 1

    print("MAX UTT_ID = " + str(max_utt_id))
    print("Number of files with wrong format: " + str(num_wrong_format))
    print("Renamed " + str(num_renamed) + " files.")

def check_correct_format(directory):
    files_checked = 0
    num_correct_format = 0
    pattern = re.compile(r'dia\d{4}_utt\d{2}\.wav$')
    for filename in os.listdir(directory):
        if filename.endswith('.wav') and filename.startswith('dia'):
            # Use Reg ex to check if the filename is in the correct format: "dia1234_utt12.wav"
             if pattern.match(filename):
                num_correct_format += 1
        files_checked += 1
    print("Number of files in correct format: " + str(num_correct_format) + " out of " + str(files_checked))

def check_original_format(directory,filetype=".wav"):
    files_checked = 0
    num_correct_format = 0
    if filetype == ".wav":
        pattern = re.compile(r'dia\d{1,4}_utt\d{1,2}\.wav$')
    elif filetype == ".mp4":
        pattern = re.compile(r'dia\d{1,4}_utt\d{1,2}\.mp4$')

    for filename in os.listdir(directory):
        if (filename.endswith('.wav') or filename.endswith('.mp4')) and filename.startswith('dia'):
            # Use Reg ex to check if the filename is in the correct format: "dia1234_utt12.wav"
             if pattern.match(filename):
                num_correct_format += 1
        files_checked += 1
    print("Number of files in original format: " + str(num_correct_format) + " out of " + str(files_checked))

def restore_to_original_format(directory):
    num_restored = 0
    for filename in os.listdir(directory):
        if filename.endswith('.wav') and filename.startswith('dia'):
            # old_filename = filename
            dia_index = filename.find("dia")
            _utt_index = filename.find("_utt")
            _wav_index = filename.find(".wav")

            dialogue_id_len = _utt_index - (dia_index + 3)
            dialogue_id = filename[dia_index + 3:dia_index + 3 + dialogue_id_len]
            dialogue_id_int = int(dialogue_id)

            utterance_id_len = _wav_index - (_utt_index + 4)
            utterance_id = filename[_utt_index + 4:_utt_index + 4 + utterance_id_len]
            utterance_id_int = int(utterance_id)
            new_filename = "dia" + str(dialogue_id_int) + "_utt" + str(utterance_id_int) + ".wav"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            num_restored += 1
    print("Restored " + str(num_restored) + " files to original format.")

def cleanup_files(directory):
    # Remove files named like '._dia0_utt0.mp4'
    num_removed_files = 0
    for filename in os.listdir(directory):
        if filename.startswith("._"):
            os.remove(os.path.join(directory, filename))
            # print(f"Removed {filename}")
            num_removed_files += 1
        elif filename.startswith("final"):
            os.remove(os.path.join(directory, filename))
            num_removed_files += 1
    print(f"Removed {num_removed_files} files")

def main():
    directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_dataset\train\train_audio"
    # correct_filename_format_MELD(directory)
    
    # count how many files start with "._"
    # count = 0
    # for filename in os.listdir(directory):
        # if filename.startswith("._"):
            # count += 1
        # if filename.startswith("final"):
            # count += 1
    # print("Files starting with ... :")
    # print(count)

    # cleanup_files(directory)
    # rename_numbering(directory)
    check_correct_format(directory)
    # check_original_format(directory, ".wav")
    # restore_to_original_format(directory)

if __name__ == "__main__":
    main()