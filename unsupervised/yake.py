import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from copy import deepcopy
from collections import Counter, OrderedDict
from utils import levenshtein_distance


class YAKE:
    """
    Implementation of Yet Another Keyword Extractor (YAKE) algorithm.

    Originally described in "A Text Feature Based Automatic Keyword Extraction Method for Single Documents" by
    Ricardo Campos, Vitor Mangaravite, Arian Pasquali, Alipio Mario Jorge, Celia Nunes, Adam Jatowt. (ECIR)

    The algorithm was also described in-depth in "YAKE! Keyword extraction from single documents using multiple
    local features" by the same authors. (Information Sciences Journal)

    This implementation follows the algorithm as described in ECIR 2018 paper and official implementation
    (https://github.com/LIAAD/yake) as reference.
    """
    def __init__(self,
                 stoplist=None,
                 reproduce=False,
                 n_grams=(1, 3),
                 window=1,
                 language='english',
                 threshold=0.8):

        self.stopwords = stopwords.words(language.lower()) if not stoplist else stoplist
        self.reproduce = reproduce
        self.window = window
        self.threshold = threshold
        self.n_grams = n_grams
        self.term_weights = None
        self.keyword_weights = None

    def _preprocess_text(self, text):
        """
        2.1 Text Pre-processing

        This is the workhorse of the algorithm, which computes most of what is necessary for YAKE to function.
        It is responsible for generating keyword candidates and computing some of the core statistical elements
        used by YAKE.

        :param text: string(text) -> unprocessed text
        :return:
        """

        # Tokenize each sentence into words, organized in a list of lists.
        # <punct> is used to avoid generating n_grams that crossed punctuation marks.
        sentence_list = [
            [word if word not in punctuation else '<punct>' for word in word_tokenize(sentence)]
            for sentence in sent_tokenize(text)
        ]

        # This is mostly necessary due to an annoying preprocessing immediately step below.
        # This allows me to keep the original representation of keywords without messing with the statistics.
        original_sentences = deepcopy(self._remove_starting_capitalization(sentence_list))

        if self.reproduce:
            # For some reason, YAKE originally removed the letter 's' from every string
            for sentence in sentence_list:
                for i, word in enumerate(sentence):
                    # Also filter for stopwords, otherwise words such as 'this' would become 'thi' and mess up the stats
                    if word.endswith('s') and len(word) > 3 and word not in self.stopwords:
                        sentence[i] = word[:-1]

        # While the algorithm does give importance to capitalized words, beginning of sentences are not
        self.sentence_list = self._remove_starting_capitalization(sentence_list)

        # Complete list of all words in the text
        self.terms = [word for sentence in self.sentence_list for word in sentence if word != '<punct>']

        # Term Frequency (TF) is the amount of times a term occurs within the text
        self.term_frequency = Counter(self.terms)

        # Use context window to find left and right contexts for each word
        self.term_contexts = {word: self._get_contexts(word) for word in self.term_frequency.keys()}

        # Count number of total sentences
        self.sentence_count = len(self.sentence_list)

        # The most frequent term in the text, also known as MaxTF in the paper
        self.max_term_frequency = np.max(list(self.term_frequency.values()))

        # Mean of frequencies (MeanTF), which although not mentioned in the paper, does not consider stopword TFs
        self.mean_term_frequency = np.mean([count for word, count in self.term_frequency.items()
                                            if word not in self.stopwords])

        # Standard Deviation of frequencies (stdTF), as with MeanTF, does not consider stopword TFs
        self.std_term_frequency = np.std([count for word, count in self.term_frequency.items()
                                          if word not in self.stopwords])

        # Neat little trick to gather up all valid keywords according to n_gram sizes (defaults to {1,2,3}-grams)
        # Valid keywords at this stage must not start or end with stopwords, nor have punctuation mark within them
        self.keywords = [
            n_grams
            for sentence in original_sentences
            for step in range(self.n_grams[0], self.n_grams[1] + 1)
            for n_grams in self._generate_n_grams(sentence, step)
            if n_grams[0] not in self.stopwords and n_grams[-1] not in self.stopwords and '<punct>' not in n_grams
        ]

        # Similar to term frequencies, keyword frequencies are the amount of times a keyword has occurred
        self.keyword_frequency = Counter(" ".join(n_gram) for n_gram in self.keywords)

    def _score_casing(self, word):
        """
        2.2.1 Casing

        Weight of a word in relation to the existence of uppercase letters.
        Disregards words starting sentences unless they also occur within the text (_remove_starting_capitalization).

        Remark: although stated in the paper that the denominator should be log2(TF(word)), this is incorrect as it
        would raise a ZeroDivisionError when TF(word) = 1, i.e. word occurred once in text. Furthermore, as with many
        other equations in the paper, the official implementations uses base e logarithm (math.log, np.log, ln, ...)
        To avoid the aforementioned ZeroDivisionError, the denominator is incremented by one.

        :param word: string(word) -> word being considered
        :return: float(casing_weight) -> weight of word in regards to its casing
        """
        score = 0
        if word[0].isupper() and word in self.terms:
            # It doesn't matter if the word is only capitalized or an acronym if everything is lowercase.
            # No real example is shown in the paper where this is necessary.
            tmp_word = word.lower()

            # Equation (1) -> max[TF(U(w)), TF(A(w))] / (1 + log(TF(w)))
            score = max(
                self.term_frequency[tmp_word.title()],
                self.term_frequency[tmp_word.upper()]
            ) / (1.0 + np.log(self.term_frequency[word]))
        return score

    def _score_position(self, word):
        """
        2.2.2 Word Position

        Weight of a word in relation to its position within the text.
        Assumes that the most relevant keywords are most likely at the beginning of the text.

        Remarks: it seems each sentence can be counted at most once, with repeat occurrences being disregarded.
        Once again, the paper diverges from the original implementation, by using base e logarithm and having a constant
        of value 3 added to the median of the sentences (to avoid negative numbers).

        :param word: string(word) -> word being considered
        :return: float(position_weight) -> weight of word in regards to its position within text
        """

        # Equation (2) -> log(log(3 + median(Sen_w))), where Sen_w is index of the sentence where word occurs
        return np.log(
            np.log(
                3 + np.median([idx for idx, sentence in enumerate(self.sentence_list) if word in sentence])
            )
        )

    def _score_frequency(self, word):
        """
        2.2.3 Word Frequency

        Weight word according to its frequency.
        The more it occurs, the more it important is must be, which is why MeanTF and stdTF are calculated without
        consideration for stopwords.

        Remarks: It seems stdTF might be weighted in Equation 3 of the paper (by multiplication of a constant value),
        however, the equation/algorithm does not seem to reflect much of this as the value is shown to be the neutral
        element of multiplication (1).

        :param word: string(word) -> word being considered
        :return: float(frequency_weight) -> weight of word in regards to its frequency within text
        """

        # Equation (3) -> TF(w) / (MeanTF + stdTF)
        return self.term_frequency[word] / (self.mean_term_frequency + self.std_term_frequency)

    def _score_relatedness(self, word):
        """
        2.2.4 Word Relatedness to Context

        Weights word in relation its surrounding context.
        Words that are not necessarily stopwords, but behave similar to them will be heavily weighted.
        Uses a context window of variable size (default: 1, i.e. adjacent words).

        Remarks: it seems that the authors of the algorithm decided to disregard PL/PR from Equation (4)

        :param word: string(word) -> word being considered
        :return: float(relatedness_weight) -> weight of word in regards to its collective context
        """

        # Get pre-computed context of word
        left_context, right_context = self.term_contexts[word]

        # Ratio between the amount of unique words occurring in context window and the total amount of words in context
        left_ratio = len(set(left_context)) / len(left_context) if left_context else 0
        right_ratio = len(set(right_context)) / len(right_context) if right_context else 0

        # https://github.com/boudinfl/pke/issues/90 as opposed to Equation (4) which is written in long form.
        return 1 + (left_ratio + right_ratio) * (self.term_frequency[word] / self.max_term_frequency)

    def _score_sentences(self, word):
        """
        2.2.5 Word DifSentence

        Weights word according to how often it appears within different sentences, regardless of where it occurs.
        Ratio between the amount of sentences where word appears and the total amount of sentences.
        Not to be confused with _score_position, which weights a word based on the position it occupies in the text.

        :param word: string(word) -> word being considered
        :return: float(sentence_weight) -> weight of word in regards to its existence in text's sentences
        """

        # Equation (5) -> SF(w) / #Sentences, where SF(w) is the number of sentences where word appears
        return sum(1 for sentence in self.sentence_list if word in sentence) / len(self.sentence_list)

    def _word_weight(self, word):
        """
        2.3 Individual Term Weighting

        Combines every feature weight (_score_relatedness, _score_casing, _score_frequency, _score_sentences) of an
        individual word/term.
        Word relatedness is used in many cases to penalize words that appear too often but are of little importance.

        :param word: string(word) -> word being considered
        :return: float(term_score) -> individual word weight considering all features
        """

        rel = self._score_relatedness(word)

        # Equation (6) -> WREL * WPOS / [WCASE + (WFREQ / WREL) + (WSENT / WFREQ)]
        return (rel * self._score_position(word)) / sum([
            self._score_casing(word),
            (self._score_frequency(word) / rel),
            (self._score_sentences(word) / rel)
        ])

    def _keyword_weight_none(self, keyword):
        """
        2.4 Candidate Keyword List Generation w/ stopword weight being disregarded

        Scores candidate keyword, that may contain several words using Equation (7).
        Using the stopword_weighting 'none' parameter within extract_keywords method, stopwords are not factored into
        the total score of keywords that contain them.

        :param keyword: list[words] -> list of words contained within the keyword
        :return: float(keyword_score) -> candidate keyword score
        """

        # Similar to _keyword_weight_same with the added restriction that stopwords are not factored
        weight_prod = np.prod([self.term_weights[word] for word in keyword if word not in self.stopwords])
        weight_sum = sum([self.term_weights[word] for word in keyword if keyword not in self.stopwords])

        # Equation (7) -> prod(S(w)) / [TF(w) * (1 + sum(S(w)))]
        return weight_prod / (self.keyword_frequency[" ".join(keyword)] * (1 + weight_sum))

    def _keyword_weight_same(self, keyword):
        """
        2.4 Candidate Keyword List Generation w/ all words considered equal

        Scores candidate keyword, that may contain several words using Equation (7).
        Assumes all words, including stopwords, have equal weight regarding final keyword score.

        :param keyword: list[words] -> list of words contained within the keyword
        :return: float(keyword_score) -> candidate keyword score
        """

        # Equation (7) -> prod(S(w)) / [TF(w) * (1 + sum(S(w)))]
        return np.prod([self.term_weights[word] for word in keyword])\
               / (self.keyword_frequency[" ".join(keyword)] * (1 + sum([self.term_weights[word] for word in keyword])))

    def _keyword_weight_penalize(self, keyword):
        """
        2.4 Candidate Keyword List Generation w/ penalization of stopwords

        Scores candidate keyword, that may contain several words using Equation (7).
        Considers that stopword weights must be penalized.

        Remark: I found no mention to this in the paper itself, but it is the keyword scoring technique utilized in the
        original/official YAKE implementation.

        :param keyword: list[words] -> list of words contained within the keyword
        :return: float(keyword_score) -> candidate keyword score
        """

        # Initializes cummulative product and sum with their respective neutral element
        weight_prod = 1.0
        weight_sum = 0.0

        for idx, word in enumerate(keyword):

            # Original YAKE algorithm removed the letter 's' from word endings
            word = self._original_yake_quirk(word)

            if word in self.stopwords:

                # Immediate adjacent words
                term_left, term_right = [self._original_yake_quirk(kw) for kw in (keyword[idx - 1], keyword[idx + 1])]

                # Get context windows
                context_left_term = self.term_contexts[term_left]
                context_right_term = self.term_contexts[term_right]

                # Ratio between the number of times word occurs after term_left and total frequency of term_left
                prob_left = context_left_term[1].count(word) / self.term_frequency[term_left]

                # Ratio between the number of times word occurs before term_right and total frequency of term_right
                prob_right = context_right_term[0].count(word) / self.term_frequency[term_right]

                # P(L ^ R) = P(L) * P(R), probability of stopword being always surrounded by same words
                prob = prob_left * prob_right

                # Technically (1 + (1 - prob)) as the added stopword weight on the keyword product
                weight_prod *= (2 - prob)

                # Not sure if this should be increment (we want to minimize S(w) for better keywords), or if we are
                # actively penalizing compensating for the product
                weight_sum -= (1 - prob)
            else:
                # If word is not a stopword, then business as usual
                weight_prod *= self.term_weights[word]
                weight_sum += self.term_weights[word]

        # Equation (7) -> prod(S(w)) / [TF(w) * (1 + sum(S(w)))]
        return weight_prod / (self.keyword_frequency[" ".join(keyword)] * (1 + weight_sum))

    def _get_contexts(self, word):
        """
        Get surrounding context of a given word according to a context window (self.window)

        Disallows the existence of punctuation marks within context.
            Ex:
                "In this work, we propose a novel", with a context window of 2, focusing on the word "propose":
                    left_context = [we]
                    right_context = [a, novel]

        Resulting lists contain context for word considering entire document.
            Ex:
                "propose" occurs in "In this work, we propose a novel" and "and they propose a framework",
                considering a context window of 2:
                    left_context = [we, and, they]
                    right_context = [a, novel, a, framework]

        :param word: string(word) -> word being considered
        :return: tuple(list[left_context], list[right_context]) -> tuple containing left and right contexts
        """

        left_context, right_context = [], []

        for sentence in self.sentence_list:

            # Find word index where word occurs in sentence
            word_indices = [idx for idx, tmp in enumerate(sentence) if word == tmp]

            # Get adjacent left context according to window size
            left_context.extend(sentence[max(0, lidx - self.window): lidx] for lidx in word_indices)

            # Get adjacent right context according to window size
            right_context.extend(
                sentence[ridx + 1: ridx + self.window + 1] for ridx in word_indices if ridx < len(sentence) - 1)

        clean_left, clean_right = [], []

        # Only keep context words up to the first punctuation mark (by reversing left_context we can use same method)
        for l_context in left_context:
            clean_left.extend(self._prune_at_punctuation(l_context[::-1]) if '<punct>' in l_context else l_context)

        for r_context in right_context:
            clean_right.extend(self._prune_at_punctuation(r_context) if '<punct>' in r_context else r_context)

        return clean_left, clean_right

    def _original_yake_quirk(self, word):
        """
        This is necessary to avoid KeyError when original YAKE 's' letter pruning is done.

        :param word: string(original word) -> new word depending if we need to take the last letter ('s')
        :return:
        """
        if not self.reproduce:
            return word
        return word if word in self.term_weights.keys() else word[:-1]

    @staticmethod
    def _prune_at_punctuation(context):
        """
        Find index of first occurrence of punctuation and return the context up to it.
        If not mistaken, order should not matter beyond returning the correct sequence up to punctuation.

        :param context: list[words] -> context
        :return: list[words] -> context up to first punctuation (excluding)
        """
        return context[:context.index('<punct>')]

    @staticmethod
    def _remove_starting_capitalization(sentences):
        """
        Removes capitalization of first word in sentence.
        Might maintain capitalization if and only if it finds another capitalized occurrence within the text, excluding
        other sentence starting words.

        :param sentences: list[list[words]] -> list of sentences that have been tokenized into words
        :return: list[list[word]] -> same list as input but with sentence staring words in lowercase
        """

        # Beginning word of each sentence
        starting = [sentence[0] for sentence in sentences]

        for starting_word in starting:

            # Starting word is an acronym with at least two uppercase letters
            if starting_word.isupper() and len(starting_word) > 1:
                continue

            # Check if there is at least one occurrence of the capitalized word within any sentence other than at start
            if not any(starting_word in sentence[1:] for sentence in sentences):

                # If there are no occurrences that satisfy condition then lowercase the starting word
                for sentence in sentences:
                    if sentence[0] == starting_word:
                        sentence[0] = sentence[0].lower()

        return sentences

    @staticmethod
    def _too_similar(candidate, already_considered, threshold=0.8):
        """
        Computes ratio between 1 - Levenshtein_Distance and the longest keyword being considered.
        It runs this for every new candidate, comparing to every previously considered candidate.

        Returns True if, for any previous candidate, the ratio falls above a certain threshold.
        This threshold indicates that the pair of words are too similar.

        The candidates have previously been ordered in ascending weight.

        :param candidate: string(word) -> current keyword candidate being considered
        :param already_considered: list[words] -> every keyword already considered
        :param threshold: float(0-1) -> maximum threshold (similarity ratio) allowed
        :return: boolean
        """

        return any((1.0 - levenshtein_distance(candidate, keyword)[0] / max(len(candidate), len(keyword)) > threshold)
                   for keyword, _ in already_considered)

    @staticmethod
    def _generate_n_grams(sentence, step):
        """
        Naively run a sliding window that captures n-grams of a given step.

        :param sentence: list[words] -> list of words in a sentence
        :param step: int() -> size of n-grams we want to obtain
        :return: list[list[words]] -> result of sliding window
        """

        # len(sentence) + 1 - step allows the last iteration to be perfectly within the final n_gram
        return [sentence[start:start+step] for start in range(len(sentence) + 1 - step)]

    def extract_keywords(self, text, num_keywords=20, stopword_weighting='penalize'):
        """
        Extract keywords according to given parameters and weighing method.

        :param text: string(original text) -> text we want to extract keywords from
        :param num_keywords: total number of keywords we want
        :param stopword_weighting: stopword weighing method
        :return: list[tuple(keyword, score)] -> best scoring keywords
        """

        # Prepare text and gather core statistics
        self._preprocess_text(text)

        # Weight individual terms
        self.term_weights = {word: self._word_weight(word) for word in self.term_frequency.keys()}

        keyword_weighting = {'penalize': self._keyword_weight_penalize,
                             'same': self._keyword_weight_same,
                             'none': self._keyword_weight_none}

        # Weight keyword (obtained by sliding window during preprocessing) according to weighting technique
        keyword_weights = {" ".join(keyword): keyword_weighting[stopword_weighting](keyword)
                           for keyword in self.keywords}

        # Sort keywords in ascending order according to their weight score
        self.keyword_weights = OrderedDict(sorted(keyword_weights.items(), key=lambda item: item[1]))

        best_keywords = []

        for keyword, score in self.keyword_weights.items():

            # Remove keywords too similar to ones already considered, keeping those with lesser score
            if not self._too_similar(keyword, best_keywords, self.threshold):
                best_keywords.append((keyword, round(score, 6)))

            if len(best_keywords) >= num_keywords:
                break

        return best_keywords


text_content = "In this work, we propose a lightweight approach for keyword extraction and ranking based on an " \
       "unsupervised methodology to select the most important keywords of a single document. To understand " \
       "the merits of our proposal, we compare it against RAKE, textrank and singlerank methods (three well-known " \
       "unsupervised approaches) and the baseline tf-idf, over four different collections to illustrate the " \
       "generality of our approach. The experimental results suggest that extracting keywords from documents using " \
       "our method results in a superior effectiveness when compared to similar approaches."

reproduce_yake = True

# Technically I can also just install yake in this project as well
# Install official YAKE algorithm via pip -> pip install git+https://github.com/LIAAD/yake
# Tested with this exact string (abstract to YAKE paper from ECIR 2018) using version 0.4.4
if reproduce_yake:

    # Since I do not own stopword list, I won't include it in this project
    with open('../data/stopwords_en.txt') as f:
        stop_lst = f.read().splitlines()

    assert len(stop_lst) == 575

    yake = YAKE(stoplist=stop_lst, reproduce=reproduce_yake, window=1)
    response = yake.extract_keywords(text_content, stopword_weighting='penalize', num_keywords=50)

    # This list was obtained in the following manner, using official YAKE algorithm
    #
    # import yake
    # kw_extractor = yake.KeywordExtractor(top=50, dedupLim=0.8)
    # actual_yake_response = [(k, round(v, 6))for k,v in kw_extractor.extract_keywords(text)]

    actual_yake_response = [
        ('propose a lightweight', 0.028374), ('extraction and ranking', 0.028374), ('ranking based', 0.028374),
        ('methodology to select', 0.028374), ('well-known unsupervised approaches', 0.046841),
        ('lightweight approach', 0.049308), ('unsupervised methodology', 0.049308), ('single document', 0.064314),
        ('keyword extraction', 0.069818), ('important keywords', 0.069818), ('unsupervised approaches', 0.093349),
        ('work', 0.127663), ('well-known unsupervised', 0.130814), ('approach', 0.141448), ('unsupervised', 0.141448),
        ('approaches', 0.150902), ('textrank and singlerank', 0.159825), ('baseline tf-idf', 0.159825),
        ('propose', 0.166106), ('lightweight', 0.166106), ('extraction', 0.166106), ('ranking', 0.166106),
        ('based', 0.166106), ('methodology', 0.166106), ('select', 0.166106), ('important', 0.166106),
        ('single', 0.166106), ('similar approaches', 0.178547), ('rake', 0.181201), ('keywords', 0.196446),
        ('understand the merits', 0.200788), ('collections to illustrate', 0.200788),
        ('illustrate the generality', 0.200788), ('singlerank methods', 0.205429),
        ('extracting keywords', 0.245673), ('results', 0.251994),
        ('experimental results suggest', 0.256018), ('method results', 0.265889), ('suggest that extracting', 0.336858),
        ('superior effectiveness', 0.336858), ('effectiveness when compared', 0.336858),
        ('compared to similar', 0.336858), ('proposal', 0.336989), ('textrank', 0.336989), ('tf-idf', 0.336989),
        ('experimental results', 0.338667), ('results suggest', 0.338667), ('document', 0.363788),
        ('approach for keyword', 0.387358), ('understand', 0.408918)
    ]

    assert all((keyword.lower(), score) in actual_yake_response for keyword, score in response)

else:
    yake = YAKE(window=1)
    response = yake.extract_keywords(text_content, num_keywords=10)
    print(response)
