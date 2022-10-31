import re


class AudiomothAudioFileName:
    def __init__(self, filename: str):

        match_obj = re.match(r'(\d{8})_(\d{6})\.WAV', filename, re.IGNORECASE)
        if not  match_obj:
            raise NameError(filename + " is not a valid Song Meter recording filename")
        self.date = match_obj.groups()[0][0:4]+"-"+match_obj.groups()[0][4:6]+"-"+match_obj.groups()[0][6:8]
        self.time = match_obj.groups()[1][0:2] + ":" + match_obj.groups()[1][2:4] + ":" + match_obj.groups()[1][4:6]
        self.filename = filename
