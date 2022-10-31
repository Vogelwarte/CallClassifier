import copy
from pathlib import Path
from typing import List, Dict

from BNResultTools.RelativeTimeSegment import RelativeTimeSegment
from BNResultTools.SingleVocalisationSegment import extract_vocalistions, SingleVocalisationSegment, \
    extract_vocalistions_ext
from BNResultTools.SynonymeChecker import SynonymeChecker
from BNResultTools.Table1SelectionRecord import Table1SelectionRecord

class BenchmarkUnitResult:
    def __init__(self, correct_detections: int = 0, not_detected: int = 0, false_positives: int = 0):
        self.correct_detections = correct_detections
        self.not_detected = not_detected
        self.false_positives = false_positives

    def add(self, bur: 'BenchmarkUnitResult'):
        self.correct_detections = self.correct_detections + bur.correct_detections
        self.not_detected = self.not_detected + bur.not_detected
        self.false_positives = self.false_positives + bur.false_positives

    def FP_ratio(self) -> float:
        all_detections: int = self.correct_detections + self.false_positives
        if all_detections == 0:
            return 0.0
        return self.false_positives / all_detections

    def ND_ratio(self) -> float:
        all_vocs: int = self.correct_detections + self.not_detected
        if all_vocs == 0:
            return 0.0
        return self.not_detected / all_vocs

    def TP_ratio(self) -> float:
        all_vocs: int = self.correct_detections + self.not_detected
        if all_vocs == 0:
            return 0.0
        return self.correct_detections / all_vocs

    @staticmethod
    def get_csv_headers(separator: str = ";") -> str:
        return f'"Annotated calls"{separator} ' \
               f'"Correct Detections"{separator} "CD-ratio"{separator} '\
               f'"Not Detected"{separator} "ND-ratio"{separator} '\
               f'"False Positives"{separator} "FP-ratio" '

    def as_csv_row(self, separator: str = ";") -> str:
        return f"{self.not_detected+self.correct_detections}{separator} "\
               f"{self.correct_detections}{separator} {self.TP_ratio()}{separator} "\
               f"{self.not_detected}{separator} {self.ND_ratio()}{separator} "\
               f"{self.false_positives}{separator} {self.FP_ratio()}"


class BenchmarkUnitResultSet:

    def __init__(self):
        self.results: List[BenchmarkUnitResult] = [BenchmarkUnitResult() for _ in range(100)]

    def set(self, index: int, bur: BenchmarkUnitResult ):
        self.results[index] = copy.deepcopy(bur)

    def add(self, rhs: 'BenchmarkUnitResultSet'):
        for i in range(len(self.results)):
            self.results[i].add(rhs.results[i])

    def get(self, minimal_confidence: int):
        if minimal_confidence > len(self.results):
            return self.results[-1]
        if minimal_confidence <= 0:
            return self.results[0]
        return self.results[minimal_confidence]


class BenchmarkUnit:
    def __init__(self, reference_file: Path, fn_suffix: str, label_checker: SynonymeChecker, all_result_files: Dict[str, Path]):
        self.ref_file = reference_file
        self.label_checker = label_checker
        if not reference_file.name.endswith(fn_suffix):
            raise Exception(f"reference file name [{reference_file.name}] "
                            f"doesn't have the expected suffix [{fn_suffix}]")
        self.ref_file_stem = self.ref_file.name[0:-(len(fn_suffix))]
        self.result_file = all_result_files.get(self.ref_file_stem, None)
        self.active_records: List[RelativeTimeSegment] = []
        self.other_records: List[RelativeTimeSegment] = []
        records: List[Table1SelectionRecord] = Table1SelectionRecord.parse_file(str(self.ref_file))
        for r in records:
            # print( "\t" + r.to_string() + " / " + r.to_mmss_string() + ", " + str(r.duration()) + "s :" + r.annotation)
            key = r.annotation.strip().lower()
            matched = label_checker.is_synonyme(key)
            if matched:
                self.active_records.append(r)
            else:
                self.other_records.append(r)



    def get_benchmark(self) -> BenchmarkUnitResult:
        if self.result_file == None:
            raise Exception( "Cannot find result file for ["+self.ref_file_stem+"]")
        vocs_to_check = extract_vocalistions_ext(self.result_file, self.label_checker)
        vtc_number = len(vocs_to_check)

        ref_detected : Dict[RelativeTimeSegment, SingleVocalisationSegment] = {}
        for ref_voc in self.active_records:
            for checked in vocs_to_check:
                #assume the detection result is valid if it overlaps an annotation it at least 1/3
                #or alt least for 1second (for longer annotations)
                if ref_voc.overlaping_time(checked) >= min(1, ref_voc.duration()/3.0 ):
                    ref_detected[ref_voc] = checked
                    vocs_to_check.remove(checked)
                    break
        burs: BenchmarkUnitResultSet = BenchmarkUnitResultSet()
        for cs in range(100):
            threshold: float = cs / 100.0
            cs_NotDetected: int = 0
            cs_TruePositives:int = 0
            for ref,tested in ref_detected.items():
                if tested.min_max_confidence()[1] >= threshold:
                    cs_TruePositives += 1
                else:
                    cs_NotDetected += 1
            cs_FalsPositives:int = 0
            for v in vocs_to_check:
                if v.min_max_confidence()[1] >= threshold:
                    cs_FalsPositives += 1
            cs_NotDetected += len(self.active_records) - len(ref_detected.items())
            burs.set(cs, BenchmarkUnitResult(cs_TruePositives, cs_NotDetected, cs_FalsPositives))
        return burs



