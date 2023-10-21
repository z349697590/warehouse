import jieba
from gensim.models import Word2Vec
from sklearn import metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
res = []
with open("./lemon_plus.txt", "r", encoding="utf-8") as f:
    res = f.readlines()

res_remove_answer = []
for x in res:
    res_remove_answer.append(x.replace("柠檬", ""))

res_remove_answer = sorted(res_remove_answer, key = lambda x: len(x))

print(f"len(prompt): {[len(s) for s in res_remove_answer]}")

# 使用 jieba 进行分词
sentences = [list(jieba.cut(sentence)) for sentence in res]

# 训练 Word2Vec 模型
model = Word2Vec(sentences, vector_size=200, window=10, min_count=1, workers=4)

# 获取词汇表中的词
words = list(model.wv.key_to_index.keys())

def get_sentence_embedding(sentence, model):
    # 对句子进行分词
    words = list(jieba.cut(sentence))

    # 获取每个词的向量，并将它们存储在一个列表中
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    # 计算词向量的平均值
    sentence_vector = np.mean(word_vectors, axis=0)

    return sentence_vector

sentenc_vector1 = get_sentence_embedding(res_remove_answer[0], model)
sentenc_vector2 = get_sentence_embedding(res_remove_answer[-1], model)

vector = model.wv['柠檬']

# print("method1: 互信息")
# print(f"{res_remove_answer[0]} - 柠檬：{metrics.mutual_info_score(vector, sentenc_vector1)}")
# print(f"{res_remove_answer[-1]} - 柠檬：{metrics.mutual_info_score(vector, sentenc_vector2)}")

print("method1： 互信息 ")
all_sim = []
for i in range(len(res_remove_answer)):
    # 计算两个句子相对于"柠檬"的信息量
    similarity = metrics.mutual_info_score(vector, get_sentence_embedding(res_remove_answer[i], model))
    # print(f"{res_remove_answer[i]} - 柠檬：{similarity}")
    all_sim.append(similarity)
print(all_sim, sep = " ")
print()


def get_sentence_similarity_to_apple(sentence, model):
    # 对句子进行分词
    words = list(jieba.cut(sentence))

    # 获取每个词的向量，并将它们存储在一个列表中
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    # 获取"柠檬"的词向量
    apple_vector = model.wv['柠檬']

    # 计算每个词向量与"柠檬"的余弦相似度，并取平均
    similarity_scores = [cosine_similarity([vec], [apple_vector]) for vec in word_vectors]
    mean_similarity = np.mean(similarity_scores)

    return mean_similarity

# 计算两个句子相对于"柠檬"的信息量
similarity1 = get_sentence_similarity_to_apple(res_remove_answer[0], model)
similarity2 = get_sentence_similarity_to_apple(res_remove_answer[-1], model)

# print("method2： 余弦相似度")
# print(f"{res_remove_answer[0]} - 柠檬：{similarity1}")
# print(f"{res_remove_answer[-1]} - 柠檬：{similarity2}")


print("method2： 余弦相似度 ")
all_sim = []
for i in range(len(res_remove_answer)):
    # 计算两个句子相对于"柠檬"的信息量
    similarity = get_sentence_similarity_to_apple(res_remove_answer[i], model)
    # print(f"{res_remove_answer[i]} - 柠檬：{similarity}")
    all_sim.append(similarity)
print(all_sim, sep = " ")
print()



print("method3： jaccard_similarity ")
def jaccard_similarity(sentence1, sentence2):
    set1 = set(jieba.cut(sentence1))
    set2 = set(jieba.cut(sentence2))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)
all_sim = []
for i in range(len(res_remove_answer)):
    # 计算两个句子相对于"柠檬"的信息量
    similarity = jaccard_similarity(res_remove_answer[i], "这是一个柠檬")
    # print(f"{res_remove_answer[i]} - 柠檬：{similarity}")
    all_sim.append(similarity)
print(all_sim, sep = " ")
print()


print("method4： TF-IDF ")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer_tfidf = TfidfVectorizer()
sentences_with_target = res_remove_answer + ["这是一个柠檬"]
tfidf = vectorizer_tfidf.fit_transform(sentences_with_target)
tfidf = tfidf.toarray()

def tfidf_similarity(tfidf, index):
    target_vector = tfidf[-1].reshape(1, -1)
    sentence_vector = tfidf[index].reshape(1, -1)
    similarity = cosine_similarity(target_vector, sentence_vector)
    return similarity

all_sim = [tfidf_similarity(tfidf, i) for i in range(len(res_remove_answer))]
print(all_sim, sep = " ")
print()


print("method5： Bag of Words ")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer_bow = CountVectorizer()
sentences_with_target = res_remove_answer + ["这是一个柠檬"]
bow = vectorizer_bow.fit_transform(sentences_with_target)
bow = bow.toarray()

def bag_of_words_similarity(bow, index):
    target_vector = bow[-1].reshape(1, -1)
    sentence_vector = bow[index].reshape(1, -1)
    similarity = cosine_similarity(target_vector, sentence_vector)
    return similarity

all_sim = [bag_of_words_similarity(bow, i) for i in range(len(res_remove_answer))]
print(all_sim, sep = " ")
print()


print("method6： 词移距离（Word Mover's Distance, WMD） ")
all_sim = [model.wv.wmdistance(res_remove_answer[i], "柠檬") for i in range(len(res_remove_answer))]
print(all_sim, sep = " ")
print()
