import os
import sys
from pathlib import Path
from typing import List, Dict


def list_of_files(dir: Path, extension: str) -> List[Path]:
    return list_of_files_ex(dir, [extension])


def list_of_files_ex(dir: Path, extensions: List[str]) -> List[Path]:
    all_files: List[Path] = []
    for dirPath, dirNames, fileNames in os.walk(dir):
        for f in fileNames:
            for ex in extensions:
                if f.lower().endswith(ex.lower()):
                    all_files.append(Path(dirPath) / f)
    return all_files



def dictionary_by_bare_name(files: List[Path], suffix_to_remove: str) -> Dict[str, Path]:
    #    print("checking the files"+str(files))
    d: Dict[str, Path] = {}
    for f in files:
        if f.name.lower().endswith(suffix_to_remove.lower()):
            bn = f.name[0:-len(suffix_to_remove)]
            #print("[" + bn + "] -> " + str(f))
            if not d.get(bn, False):
                d[bn] = f
            else:
                sys.stderr.write("[" + bn + "] is not a unique name: \n" + str(f) + "\nor\n" +  str(d[bn]) + " - kept as first")
        else:
            sys.stderr.write(str(f) + " doesn't have the expecetd suffix [" + suffix_to_remove + "]")
    return d
