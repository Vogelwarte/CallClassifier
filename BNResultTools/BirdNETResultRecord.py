from BNResultTools.RelativeTimeSegment import RelativeTimeSegment


class BirdNETResultRecord(RelativeTimeSegment):
    def __init__(self, t0: float, t1: float, confidence: float, species_code: str, common_name: str):
        super().__init__(t0, t1)
        self.confidence = confidence        # float from the range [0; 1.0]
        self.species_code = species_code    # species code used in BirdNET-Analyzer
        self.common_name = common_name      # in English, possibly with spaces

    def to_mmss_string(self, separator: str = "-") -> str:
        return super().to_mm_ss(separator) \
               + " " + self.species_code \
               + " (" + self.common_name+") " \
               + str(self.confidence)

    def to_string(self, separator: str = "-") -> str:
        return super().to_strindstr(separator) \
               + " " + self.species_code \
               + " ("+self.common_name+") " \
               + str(self.confidence)

