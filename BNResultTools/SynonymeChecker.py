import re
import struct
from re import Pattern
from typing import TextIO, Dict, List


class SynonymeChecker:

    def __init__(self, name: str, regexps: List['Pattern']):
        self.name = name
        self.regexps = regexps
        self.is_empty_a_synonyme = False

    def empty_is_synonym(self,  is_empty_a_synonym: bool):
        self.is_empty_a_synonyme = is_empty_a_synonym


    def is_synonyme(self, input: str) -> bool:
        """
        Check if the input string matches any of the synonym regular expressions

        :param input: string to be checked, if it maches any of the synonyms
        :return: True, the input matches a synonym, False else
        """
        m,p = self.is_synonyme_ext(input)
        return m

    def is_synonyme_ext(self, input: str) -> (bool, Pattern):
        """
        Check if the input string matches any of the synonym regular expressions

        :param input: string to be checked, if it maches any of the synonyms
        :return: (True, pattern), if the input matches a synonym, (False,None) else
        """
        if self.is_empty_a_synonyme and len(input.strip()) == 0:
            return True, None

        for r in self.regexps:
            if r.fullmatch(input):
                return True, r.pattern
        return False, None

    @staticmethod
    def parse_file(filename: str) -> 'SynonymeChecker':
        regexps: List[Pattern] = []
        file = open(filename, "r")
        for line in file.readlines():
            regexps.append(re.compile(line.strip().lower(), re.IGNORECASE))
        if len(regexps) == 0:
            raise ImportError(filename + " doesn't contain any synonym or regular expression")
        return SynonymeChecker(filename, regexps)
