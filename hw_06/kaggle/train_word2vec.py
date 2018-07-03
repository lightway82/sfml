from gensim.models.word2vec import Word2Vec
from bs4 import BeautifulSoup
import numpy as np
import gensim, logging
import pymystem3
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def text_to_vec(words, model, size):
    text_vec = np.zeros((size,), dtype="float32")
    n_words = 0

    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            n_words = n_words + 1
            text_vec = np.add(text_vec, model[word])

    if n_words != 0:
        text_vec /= n_words
    return text_vec


def texts_to_vecs(texts, model, size):
    texts_vecs = np.zeros((len(texts), size), dtype="float32")

    for i, text in enumerate(texts):
        texts_vecs[i] = text_to_vec(text, model, size)

    return texts_vecs




class Word2VecDataIterator(object):
    def __init__(self, file: str):
        self.__file = file
        self.__count=0


    def __iter__(self):
        for line in open(self.__file, "rt"):
            res = self._prepare_string(line)
            if len(res) != 2:
                continue

            name, description = res

            if self.__count == 0:
                continue#пропуск первой строки c заголовками

            yield name.split()
            yield description.split()
            self.__count+=1
            #print(self.count)


    def _prepare_string(self, string_from_file: str) -> (str, str):
        return string_from_file.split(sep="\t")




class Word2VecDataIteratorWithPreprocess(Word2VecDataIterator):
    def __init__(self, file: str):
        super(Word2VecDataIteratorWithPreprocess, self).__init__(file)
        self.__mystem = pymystem3.Mystem()

    def _prepare_string(self, string_from_file: str) -> (str, str):
        res = super()._prepare_string(string_from_file)
        if len(res) != 2:
            return []
        name, description = res
        description = BeautifulSoup(description.replace("<", " <"), "lxml").get_text()
        return self.__sanitize_str(name), self.__sanitize_str(description)





    def __sanitize_str(self, str):
        return Word2VecDataIteratorWithPreprocess.__remove_digits(
            " ".join(filter(lambda word: len(word) > 2,
                            map(lambda w: re.sub(r'[\s+")(]', '', w).lower(),
                                self.__mystem.lemmatize(str))))
        )

    @staticmethod
    def __remove_digits(str):
        return re.sub(r'[^\w\s]+|[\d]+', r'', str).strip()


class Word2VecDataIterator_(Word2VecDataIterator):
    def __init__(self, file: str):
        super(Word2VecDataIterator_, self).__init__(file)

    def _prepare_string(self, string_from_file: str) -> (str, str):
        _, name, description, _ = super()._prepare_string(string_from_file)
        return name, description


# симортируем соответствующую функцию из модуля gensim, который должен быть установлен



# список параметров, которые можно менять по вашему желанию
num_features = 300  # итоговая размерность вектора каждого слова
min_word_count = 5  # минимальная частотность слова, чтобы оно попало в модель
num_workers = 4  # количество ядер вашего процессора, чтоб запустить обучение в несколько потоков
context = 10  # размер окна
downsampling = 1e-3  # внутренняя метрика модели

model = Word2Vec(Word2VecDataIteratorWithPreprocess("other.csv"), workers=num_workers, size=num_features,
                 min_count=min_word_count, window=context, sample=downsampling)

# Финализируем нашу модель. Ее нельзя будет доучить теперь, но она начнет занимать гораздо меньше места
model.init_sims(replace=True)
print("save")
model.save(fname_or_handle="trained_full.vec")
