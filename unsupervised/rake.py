import re
from collections import Counter
from itertools import groupby, repeat
from string import punctuation
from typing import NamedTuple

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Candidate(NamedTuple):
    elements: list[str]
    text: str


class RAKE:
    """
    Implementation of Rapid Automatic Keyword Extraction (RAKE) algorithm.

    Originally described in "Automatic Keyword Extraction from Individual Documents"
    by Stuart Rose, Dave Engel, Nick Cramer and Wendy Cowley (2010).
    """
    def __init__(self, language='english', min_length=1, max_length=4, num_keywords=None):
        self.language = language
        self.max_length = max_length
        self.min_length = min_length
        self.num_keywords = num_keywords
        self.stopwords = stopwords.words(language.lower())

        self.word_frequency = None
        self.word_degree = None
        self.ratio = None

    def extract_keywords(self, text):
        """
        Public method to run RAKE algorithm on given text.

        :param text: string(text) -> original text
        :return: string(keywords) -> highest scoring keywords/phrases (amount controlled by num_keywords on __init__)
        """
        self.word_frequency = None
        self.word_degree = None
        self.ratio = None

        candidates = self._generate_candidate_keywords(text)
        self._keyword_scores(candidates)
        return self._rank_keyword_scores(candidates)

    def _generate_candidate_keywords(self, text):
        """
        # 1.2.1 Candidate keywords

        Generate a list of potential candidates based on defined delimiters.
        Keyword/keyphrase delimiters are punctuation and stopwords. hyphenated words are allowed as long as NLTK's
        word_tokenize does not break them up.

        :param text: string(text) -> original text
        :return: list[Candidate]
        """

        # Reduce text into list of words
        word_list = word_tokenize(text.lower(), language=self.language)

        # Split list of words based on delimiters (punctuation or stopwords), returning list of lists of relevant words
        split_on_punctuation = [
            list(word_group)
            for is_delimiter, word_group in groupby(word_list, lambda x: x in self.stopwords + list(punctuation))
            if not is_delimiter
        ]

        # Break candidates that exceed max amount of keywords into smaller ones, left-to-right (Note: not in paper)
        split_on_max = [
            reduced_candidate
            for candidate in split_on_punctuation
            for reduced_candidate in [candidate[i:i + self.max_length]
                                      for i in range(0, len(candidate), self.max_length)]
            ]

        # Generate candidates that might have interior stopwords but are important (appear multiple times) nonetheless
        adjoined_candidates = self._adjoining_keywords(split_on_max, text.lower())

        # Remove candidates that do not have minimum amount of keywords
        candidates = [
            Candidate(candidate, ' '.join(candidate))
            for candidate in split_on_max
            if len(candidate) >= self.min_length
        ]

        # Adjoined candidates might have more than the minimum necessary, but individually would not
        candidates.extend(adjoined_candidates)

        return candidates

    def _keyword_scores(self, candidates):
        """
        # 1.2.2 Keyword scores

        Calculates score for individual keywords. RAKE object retains information on different metrics:
            - word_frequency: total amount of times word occurs in text
            - degree: amount of times word appears in relation to others (co-occurrence matrix), including to itself
            - ratio: ratio between word degree and word frequency, i.e. deg(e)/freq(w)

        :param candidates: list[Candidate] -> list of candidates
        :return: None
        """

        # Dictionary<word: int(count)> -> Total amount of times word appears in text
        self.word_frequency = Counter(word for phrase in candidates for word in phrase.elements)

        # Dictionary<word: int(degree)> -> Sum of co-occurrences, including to itself
        self.degree = dict(zip(self.word_frequency.keys(), repeat(0)))

        for candidate in candidates:
            # This is equivalent of computing the co-occurrence matrix and immediately summing values
            for word in candidate.elements:
                self.degree[word] += len(candidate.elements)

        # Dictionary<word: float(ratio)> -> Divide each word's degree by its frequency
        self.ratio = {word: self.degree[word] / freq for word, freq in self.word_frequency.items()}

    def _adjoining_keywords(self, keywords, original_text):
        """
        # 1.2.3 Adjoining keywords

        Allow some leeway in terms of keyphrases that might contain interior stopwords.
        RAKE looks for pairs of keywords that adjoin one another at least twice in the same document.
        A new candidate keyword is then created as a combination of those keywords and their interior stop words.

        This step is done prior to calculating scores (paper is confusing whether it should be before or after).
        It is also possible to group keywords that would otherwise not have the minimum amount of keywords according to
        min_length upon algorithm initialization.

        :param keywords: list[string(word)] -> List of strings containing keywords
        :param original_text: full text (lower-cased)
        :return: list[Candidate] -> list of new keyphrase candidates that obey to imposed restrictions
        """

        adjoining = []

        # Sliding window on pairs of keywords
        for _former, _latter in zip(keywords, keywords[1:]):

            # Total keywords if both elements are joined
            total_keywords = len(_former) + len(_latter)

            # Final keyword amount must not exceed the maximum or be lower than the minimum allowed
            if total_keywords > self.max_length or total_keywords < self.min_length:
                continue

            # Join elements in string form (they have not been converted to Candidate yet)
            former, latter = (" ".join(keywords) for keywords in (_former, _latter))

            # Find potential interior stopwords (this has the possibility of finding weird matches as well)
            all_matches = re.findall(rf'{former}([\w\s]+?){latter}', original_text)

            # Ensure that matches are exclusively stopwords
            filter_matches = [match for match in all_matches if set(word_tokenize(match)).issubset(rake.stopwords)]

            # The connecting stopwords must be found in the text at least twice
            if len(filter_matches) > 1:
                adjoining.extend([
                    Candidate([former, latter], f'{former}{interior_keywords}{latter}')
                    for interior_keywords, count in Counter(filter_matches).items()
                    if count > 1
                ])

        return adjoining

    def _rank_keyword_scores(self, candidates):
        """
        Compute score for entire candidate, i.e. sum score of individual keywords contained in the candidate.
        Only returns specified number of candidates:
            - default: 1/3 of total candidates
            - user defined using num_keywords param during algorithm initialization

        :param candidates: list[Candidate] -> list of candidates
        :return: best scoring candidates
        """

        # Candidate score = sum of candidate's keyword scores
        candidate_scores = {candidate.text: sum(self.ratio[word]
                                                for word in candidate.elements if word not in self.stopwords)
                            for candidate in candidates}

        # Sort candidates in decreasing order of score
        sorted_candidate_keywords = sorted(candidate_scores, key=candidate_scores.get, reverse=True)

        # Only return specified amount of candidates
        return sorted_candidate_keywords[:self.num_keywords] if self.num_keywords\
            else sorted_candidate_keywords[:round(len(candidate_scores)/3)]


if __name__ == "__main__":
    test_text = ("Compatibility of systems of linear constraints over the set of natural numbers.\nCriteria of "
                "compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict "
                "inequations are considered. Upper bounds for components of a minimal set of solutions "
                "and algorithms of construction of minimal generating sets of solutions for all types of systems are "
                "given. These criteria and the corresponding algorithms for constructing a minimal supporting set of "
                "solutions can be used in solving all the considered types of systems and systems of mixed types")

    rake = RAKE('english')
    result = rake.extract_keywords(test_text)

    if result != ['minimal generating sets', 'linear diophantine equations', 'minimal supporting set',
                      'minimal set', 'linear constraints', 'natural numbers', 'strict inequations',
                      'nonstrict inequations', 'upper bounds']:
        raise Exception("Algorithm changed")
