import math
import re
from typing import List

from BNResultTools.BirdNETResultRecord import BirdNETResultRecord
from BNResultTools.BirdNETSelectionRecord import BirdNETSelectionRecord

from BNResultTools.SynonymeChecker import SynonymeChecker
from BNResultTools.RelativeTimeSegment import RelativeTimeSegment



class SingleVocalisationSegment(RelativeTimeSegment):
    """

    """

    def __init__(self, bn_results: List[BirdNETResultRecord]):
        super().__init__(bn_results[0].begin_time, bn_results[-1].end_time);
        self.bn_results = bn_results

    def min_max_confidence(self) -> (float, float):
        mmin = mmax = self.bn_results[0].confidence
        for rn in self.bn_results:
            if rn.confidence < mmin:
                mmin = rn.confidence
            else:
                if rn.confidence > mmax:
                    mmax = rn.confidence
        return (mmin, mmax)

    def to_string(self) -> str:
        mmin, mmax = self.min_max_confidence()
        return super().to_mmss_string() + ", " \
               + str(self.duration()) + "s, " \
               + "@ " + str(mmin) + " - " + str(mmax)



def group_overlapping(records: List[BirdNETResultRecord], max_offset: float) -> List[List[BirdNETResultRecord]]:
    """
       :param records: the list of BirdNet records (lines), will be sorted BY begin_time ASC
       :param max_offset: time in seconds, measured from the begin, tolerance 1ms
       :rtype: List[List[BnCsvResultRecord]]
    """
    records.sort(key=lambda r: r.begin_time)

    groups: list[list[BirdNETResultRecord]] = []
    curr_group: list[BirdNETResultRecord] = []
    for record in records:
        if len(curr_group) == 0:
            curr_group.append(record)
        else:
            if curr_group[-1].begin_time + max_offset + 1E-3 >= record.begin_time:
                curr_group.append(record)
            else:
                groups.append(curr_group)
                curr_group = [record]
    if len(curr_group) > 0:
        groups.append(curr_group)
    return groups


def split_into_vocalisations(records: List[BirdNETResultRecord], max_duration: float) -> List[SingleVocalisationSegment]:
    """

    :param records:
    :return:
    """
    vocs: list[SingleVocalisationSegment] = []
    one = SingleVocalisationSegment(records);
    if one.duration() > max_duration + 1E-3:  # be tolerant for 1ms, just in case

        n_vocs = math.ceil(one.duration() / max_duration)
        n_rec = math.ceil(len(records) / n_vocs)
        # print("too long vocalisation: "
        #       , one.to_string()
        #       , "(",one.duration(),"s,", len(records),  " records )"
        #       , ",\n splitting into "
        #       , n_vocs, " shorter, each with max "
        #       , n_rec, " records")
        i = 0
        while i < len(records):
            sv = SingleVocalisationSegment(records[i:i + n_rec])
            # print( "\t ",i,") ",sv.to_string())
            vocs.append(sv)
            i += n_rec
    else:
        vocs = [one]

    return vocs


def extract_vocalistions(file:str, species_code:str, max_voc_duration: float = 7.0) -> List[SingleVocalisationSegment]:
    sc = SynonymeChecker(species_code, [re.compile(species_code, re.IGNORECASE)])
    return extract_vocalistions_ext(file, sc,max_voc_duration)

def extract_vocalistions_ext(file:str, species_code_checker:SynonymeChecker , max_voc_duration: float = 7.0 ) -> List[SingleVocalisationSegment]:
    #data = BnCsvResultRecord.parse_file(file)
    data: List[BirdNETResultRecord] = BirdNETSelectionRecord.parse_file(file)

    data = [r for r in data if species_code_checker.is_synonyme_ext(r.species_code)]
    data.sort(key=lambda r: r.begin_time)
    continuous_segments = group_overlapping(data, 1.0)
    nv: list[SingleVocalisationSegment] = []
    for segment in continuous_segments:
        nv.extend(split_into_vocalisations(segment, max_voc_duration))
    return nv
