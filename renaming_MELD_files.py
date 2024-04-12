import os

directory = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_rename_files_test"

max_utt_id = 0

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
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
        # print(f"Renamed {filename} to {new_filename}")

print("MAX UTT_ID = " + str(max_utt_id))