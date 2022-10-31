import re


class SongMeterAudioFileName:
    def __init__(self, filename: str):

        match_obj = re.match(r'(.*)_(\d{8})_(\d{6})\.wav', filename, re.IGNORECASE)
        if not  match_obj:
            raise NameError(filename + " is not a valid Song Meter recording filename")
        self.rec_name = re.sub(r'[-_]', "", str(match_obj.groups()[0]))
        self.date = match_obj.groups()[1][0:4]+"-"+match_obj.groups()[1][4:6]+"-"+match_obj.groups()[1][6:8]
        self.time = match_obj.groups()[2][0:2]+":"+match_obj.groups()[2][2:4]+":"+match_obj.groups()[2][4:6]
        self.filename = filename
